"""Alert dispatcher for sending real-time alerts to the web app.

Dispatches alerts (SWITCH_ITEM, SELL_NOW, ADJUST_PRICE) to the web application
when the ML engine detects opportunities or risks for active trades.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx

from .config import config
from .logging_config import get_logger
from .webhook import generate_webhook_signature

logger = get_logger(__name__)

# Request timeout in seconds
REQUEST_TIMEOUT = 5.0

# Maximum retries on failure
MAX_RETRIES = 1


class AlertType(str, Enum):
    """Types of alerts that can be sent to users."""

    SWITCH_ITEM = "SWITCH_ITEM"
    SELL_NOW = "SELL_NOW"
    ADJUST_PRICE = "ADJUST_PRICE"
    HOLD = "HOLD"


class AlertUrgency(str, Enum):
    """Urgency levels for alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class NewItemRecommendation:
    """A new item recommendation for SWITCH_ITEM alerts."""

    itemId: int
    item: str
    buyPrice: int
    sellPrice: int
    quantity: int
    expectedProfit: int
    confidence: float


@dataclass
class Alert:
    """An alert to be sent to a user."""

    id: str
    tradeId: str
    type: AlertType
    reason: str
    confidence: float
    urgency: AlertUrgency
    timestamp: str

    # For SWITCH_ITEM - new item recommendation
    newItem: Optional[NewItemRecommendation] = None

    # For SELL_NOW - optional adjusted sell price
    adjustedSellPrice: Optional[int] = None

    # For ADJUST_PRICE - price reduction details
    newSellPrice: Optional[int] = None
    originalSellPrice: Optional[int] = None

    # Computed impact
    profitDelta: Optional[int] = None


@dataclass
class AlertPayload:
    """Payload for alert webhook request."""

    userId: str
    alert: Alert


class AlertDispatcher:
    """Dispatches alerts to the web application via webhook.

    Uses fire-and-forget pattern with single retry on failure.
    Alerts are sent asynchronously to avoid blocking the main thread.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        webhook_secret: Optional[str] = None,
    ):
        """Initialize the alert dispatcher.

        Args:
            webhook_url: URL to send alerts to (defaults to config)
            webhook_secret: Shared secret for signing (defaults to config)
        """
        self.webhook_url = webhook_url or config.web_app_webhook_url
        self.webhook_secret = webhook_secret or config.webhook_secret
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _serialize_alert(self, user_id: str, alert: Alert) -> str:
        """Serialize alert payload to JSON string.

        Args:
            user_id: User identifier
            alert: Alert to serialize

        Returns:
            JSON string of the payload
        """
        # Convert alert to dict, handling nested dataclasses
        alert_dict = {
            "id": alert.id,
            "tradeId": alert.tradeId,
            "type": (
                alert.type.value if isinstance(alert.type, AlertType) else alert.type
            ),
            "reason": alert.reason,
            "confidence": alert.confidence,
            "urgency": (
                alert.urgency.value
                if isinstance(alert.urgency, AlertUrgency)
                else alert.urgency
            ),
            "timestamp": alert.timestamp,
        }

        # Add optional fields
        if alert.newItem is not None:
            alert_dict["newItem"] = asdict(alert.newItem)
        if alert.adjustedSellPrice is not None:
            alert_dict["adjustedSellPrice"] = alert.adjustedSellPrice
        if alert.newSellPrice is not None:
            alert_dict["newSellPrice"] = alert.newSellPrice
        if alert.originalSellPrice is not None:
            alert_dict["originalSellPrice"] = alert.originalSellPrice
        if alert.profitDelta is not None:
            alert_dict["profitDelta"] = alert.profitDelta

        payload = {"userId": user_id, "alert": alert_dict}
        return json.dumps(payload)

    async def _send_request(self, body: str) -> bool:
        """Send a single webhook request.

        Args:
            body: JSON body to send

        Returns:
            True if request succeeded, False otherwise
        """
        if not self.webhook_url:
            logger.warning("Alert dispatch skipped: no webhook URL configured")
            return False

        if not self.webhook_secret:
            logger.warning("Alert dispatch skipped: no webhook secret configured")
            return False

        # Generate signature
        timestamp, signature = generate_webhook_signature(body, self.webhook_secret)

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Signature": signature,
        }

        try:
            client = await self._get_client()
            response = await client.post(
                self.webhook_url,
                content=body,
                headers=headers,
            )
            response.raise_for_status()

            logger.debug(
                "Alert dispatched successfully",
                status_code=response.status_code,
            )
            return True

        except httpx.TimeoutException:
            logger.warning("Alert dispatch timeout")
            return False
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Alert dispatch HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
            )
            return False
        except Exception as e:
            logger.warning(f"Alert dispatch error: {e}")
            return False

    async def dispatch(self, user_id: str, alert: Alert) -> bool:
        """Dispatch an alert to the web application.

        Uses fire-and-forget pattern with single retry on failure.

        Args:
            user_id: User identifier
            alert: Alert to dispatch

        Returns:
            True if dispatch succeeded, False otherwise
        """
        body = self._serialize_alert(user_id, alert)

        # First attempt
        success = await self._send_request(body)
        if success:
            logger.info(
                "Alert dispatched",
                user_id=user_id[:8] + "..." if len(user_id) > 8 else user_id,
                alert_type=(
                    alert.type.value
                    if isinstance(alert.type, AlertType)
                    else alert.type
                ),
                trade_id=alert.tradeId,
            )
            return True

        # Single retry
        logger.debug("Retrying alert dispatch")
        success = await self._send_request(body)
        if success:
            logger.info(
                "Alert dispatched (retry)",
                user_id=user_id[:8] + "..." if len(user_id) > 8 else user_id,
                alert_type=(
                    alert.type.value
                    if isinstance(alert.type, AlertType)
                    else alert.type
                ),
                trade_id=alert.tradeId,
            )
            return True

        logger.warning(
            "Alert dispatch failed after retry",
            user_id=user_id[:8] + "..." if len(user_id) > 8 else user_id,
            alert_type=(
                alert.type.value if isinstance(alert.type, AlertType) else alert.type
            ),
            trade_id=alert.tradeId,
        )
        return False

    def dispatch_async(self, user_id: str, alert: Alert) -> None:
        """Dispatch an alert asynchronously (fire-and-forget).

        Creates a background task to dispatch the alert without blocking.

        Args:
            user_id: User identifier
            alert: Alert to dispatch
        """
        asyncio.create_task(self._dispatch_and_log(user_id, alert))

    async def _dispatch_and_log(self, user_id: str, alert: Alert) -> None:
        """Dispatch an alert and log the result.

        Args:
            user_id: User identifier
            alert: Alert to dispatch
        """
        try:
            await self.dispatch(user_id, alert)
        except Exception as e:
            logger.error(f"Unexpected error in alert dispatch: {e}", exc_info=True)


# Factory functions for creating common alerts


def create_switch_item_alert(
    alert_id: str,
    trade_id: str,
    reason: str,
    confidence: float,
    urgency: AlertUrgency,
    new_item: NewItemRecommendation,
    profit_delta: Optional[int] = None,
) -> Alert:
    """Create a SWITCH_ITEM alert.

    Args:
        alert_id: Unique alert identifier
        trade_id: Trade being alerted about
        reason: Human-readable reason for the alert
        confidence: Model confidence (0-1)
        urgency: Alert urgency level
        new_item: The new item recommendation
        profit_delta: Change in expected profit

    Returns:
        Alert configured for SWITCH_ITEM
    """
    return Alert(
        id=alert_id,
        tradeId=trade_id,
        type=AlertType.SWITCH_ITEM,
        reason=reason,
        confidence=confidence,
        urgency=urgency,
        timestamp=datetime.now(timezone.utc).isoformat(),
        newItem=new_item,
        profitDelta=profit_delta,
    )


def create_sell_now_alert(
    alert_id: str,
    trade_id: str,
    reason: str,
    confidence: float,
    urgency: AlertUrgency,
    adjusted_sell_price: Optional[int] = None,
    profit_delta: Optional[int] = None,
) -> Alert:
    """Create a SELL_NOW alert.

    Args:
        alert_id: Unique alert identifier
        trade_id: Trade being alerted about
        reason: Human-readable reason for the alert
        confidence: Model confidence (0-1)
        urgency: Alert urgency level
        adjusted_sell_price: Optional adjusted sell price
        profit_delta: Change in expected profit

    Returns:
        Alert configured for SELL_NOW
    """
    return Alert(
        id=alert_id,
        tradeId=trade_id,
        type=AlertType.SELL_NOW,
        reason=reason,
        confidence=confidence,
        urgency=urgency,
        timestamp=datetime.now(timezone.utc).isoformat(),
        adjustedSellPrice=adjusted_sell_price,
        profitDelta=profit_delta,
    )


def create_adjust_price_alert(
    alert_id: str,
    trade_id: str,
    reason: str,
    confidence: float,
    urgency: AlertUrgency,
    new_sell_price: int,
    original_sell_price: int,
    profit_delta: Optional[int] = None,
) -> Alert:
    """Create an ADJUST_PRICE alert.

    Args:
        alert_id: Unique alert identifier
        trade_id: Trade being alerted about
        reason: Human-readable reason for the alert
        confidence: Model confidence (0-1)
        urgency: Alert urgency level
        new_sell_price: Suggested new sell price
        original_sell_price: Original sell price
        profit_delta: Change in expected profit

    Returns:
        Alert configured for ADJUST_PRICE
    """
    return Alert(
        id=alert_id,
        tradeId=trade_id,
        type=AlertType.ADJUST_PRICE,
        reason=reason,
        confidence=confidence,
        urgency=urgency,
        timestamp=datetime.now(timezone.utc).isoformat(),
        newSellPrice=new_sell_price,
        originalSellPrice=original_sell_price,
        profitDelta=profit_delta,
    )


def create_hold_alert(
    alert_id: str,
    trade_id: str,
    reason: str,
    confidence: float,
    urgency: AlertUrgency,
) -> Alert:
    """Create a HOLD alert.

    Used when the engine detects risk but cannot recommend an action without
    violating constraints (e.g., "never sell at a loss").
    """
    return Alert(
        id=alert_id,
        tradeId=trade_id,
        type=AlertType.HOLD,
        reason=reason,
        confidence=confidence,
        urgency=urgency,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
