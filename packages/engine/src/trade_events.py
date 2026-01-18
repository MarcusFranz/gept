"""Trade event handling for web app webhook integration."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from .logging_config import get_logger

logger = get_logger(__name__)


class TradeEventType(str, Enum):
    """Types of trade lifecycle events."""

    TRADE_CREATED = "TRADE_CREATED"
    TRADE_COMPLETED = "TRADE_COMPLETED"
    TRADE_CANCELLED = "TRADE_CANCELLED"
    TRADE_UPDATED = "TRADE_UPDATED"


@dataclass
class TradePayload:
    """Payload data for a trade event."""

    item_id: int
    item_name: str
    buy_price: int
    sell_price: int
    quantity: int
    profit: Optional[int] = None  # Only on TRADE_COMPLETED
    rec_id: Optional[str] = None  # Links to original recommendation
    model_id: Optional[str] = None  # Links to model that made recommendation


@dataclass
class TradeEvent:
    """A trade lifecycle event from the web app."""

    event_type: TradeEventType
    timestamp: datetime
    user_id: str
    trade_id: str
    payload: TradePayload


class TradeEventHandler:
    """Handles incoming trade events from the web app.

    This class processes trade lifecycle events and integrates with:
    - Crowding tracker (add/remove positions)
    - Trade monitoring (start/stop monitoring for alerts)
    - Outcome recording (record completed trades for ML feedback)
    """

    def __init__(self, crowding_tracker=None, recommendation_engine=None):
        """Initialize the trade event handler.

        Args:
            crowding_tracker: Optional crowding tracker for position tracking
            recommendation_engine: Optional engine for trade monitoring
        """
        self.crowding_tracker = crowding_tracker
        self.recommendation_engine = recommendation_engine
        self._active_trades: dict[str, TradeEvent] = {}

    async def handle_event(self, event: TradeEvent) -> None:
        """Process a trade event.

        Args:
            event: The trade event to process
        """
        logger.info(
            "Processing trade event",
            event_type=event.event_type.value,
            trade_id=event.trade_id,
            user_id=(
                event.user_id[:8] + "..." if len(event.user_id) > 8 else event.user_id
            ),
            item_id=event.payload.item_id,
            item_name=event.payload.item_name,
        )

        if event.event_type == TradeEventType.TRADE_CREATED:
            await self._handle_trade_created(event)
        elif event.event_type == TradeEventType.TRADE_COMPLETED:
            await self._handle_trade_completed(event)
        elif event.event_type == TradeEventType.TRADE_CANCELLED:
            await self._handle_trade_cancelled(event)
        elif event.event_type == TradeEventType.TRADE_UPDATED:
            await self._handle_trade_updated(event)

    async def _handle_trade_created(self, event: TradeEvent) -> None:
        """Handle a new trade being created.

        - Register position with crowding tracker
        - Start monitoring for price alerts
        """
        # Store active trade
        self._active_trades[event.trade_id] = event

        # Register with crowding tracker
        if self.crowding_tracker:
            self.crowding_tracker.record_delivery(
                item_id=event.payload.item_id,
                user_id=event.user_id,
            )
            logger.debug(
                "Registered trade with crowding tracker",
                trade_id=event.trade_id,
                item_id=event.payload.item_id,
            )

        logger.info(
            "Trade created - monitoring started",
            trade_id=event.trade_id,
            item_id=event.payload.item_id,
            buy_price=event.payload.buy_price,
            sell_price=event.payload.sell_price,
        )

    async def _handle_trade_completed(self, event: TradeEvent) -> None:
        """Handle a trade being completed.

        - Remove from active trades
        - Release crowding position
        - Record outcome for ML feedback
        """
        # Remove from active trades
        self._active_trades.pop(event.trade_id, None)

        # Release crowding position
        if self.crowding_tracker:
            self.crowding_tracker.release_position(
                item_id=event.payload.item_id,
                user_id=event.user_id,
            )
            logger.debug(
                "Released crowding position",
                trade_id=event.trade_id,
                item_id=event.payload.item_id,
            )

        logger.info(
            "Trade completed",
            trade_id=event.trade_id,
            item_id=event.payload.item_id,
            profit=event.payload.profit,
            rec_id=event.payload.rec_id,
        )

    async def _handle_trade_cancelled(self, event: TradeEvent) -> None:
        """Handle a trade being cancelled.

        - Remove from active trades
        - Release crowding position
        """
        # Remove from active trades
        self._active_trades.pop(event.trade_id, None)

        # Release crowding position
        if self.crowding_tracker:
            self.crowding_tracker.release_position(
                item_id=event.payload.item_id,
                user_id=event.user_id,
            )
            logger.debug(
                "Released crowding position (cancelled)",
                trade_id=event.trade_id,
                item_id=event.payload.item_id,
            )

        logger.info(
            "Trade cancelled",
            trade_id=event.trade_id,
            item_id=event.payload.item_id,
        )

    async def _handle_trade_updated(self, event: TradeEvent) -> None:
        """Handle a trade being updated.

        - Update stored trade data
        - Adjust monitoring parameters if needed
        """
        # Update stored trade
        self._active_trades[event.trade_id] = event

        logger.info(
            "Trade updated",
            trade_id=event.trade_id,
            item_id=event.payload.item_id,
            quantity=event.payload.quantity,
        )

    def get_active_trades(self) -> dict[str, TradeEvent]:
        """Get all active trades being monitored."""
        return self._active_trades.copy()

    def get_active_trades_for_user(self, user_id: str) -> list[TradeEvent]:
        """Get active trades for a specific user."""
        return [t for t in self._active_trades.values() if t.user_id == user_id]

    def get_active_trades_for_item(self, item_id: int) -> list[TradeEvent]:
        """Get active trades for a specific item."""
        return [t for t in self._active_trades.values() if t.payload.item_id == item_id]
