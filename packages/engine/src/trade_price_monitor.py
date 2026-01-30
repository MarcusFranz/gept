"""Trade price monitor for detecting price drops on active trades.

Runs as an async background loop, periodically checking predictions against
active trade sell prices and dispatching alerts when significant drops are detected.
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from .alert_dispatcher import (
    AlertDispatcher,
    AlertUrgency,
    create_adjust_price_alert,
    create_sell_now_alert,
)
from .config import Config
from .logging_config import get_logger
from .prediction_loader import PredictionLoader
from .trade_events import TradeEvent, TradeEventHandler, TradeEventType

logger = get_logger(__name__)


@dataclass
class CooldownEntry:
    """Tracks alert cooldown state for a trade."""

    last_alert_time: float
    last_suggested_price: int


class TradePriceMonitor:
    """Monitors active trades for price drops and dispatches alerts.

    Periodically checks predictions against active trade sell prices.
    When a significant predicted price drop is detected, an appropriate
    alert (ADJUST_PRICE or SELL_NOW) is dispatched to the user.
    """

    def __init__(
        self,
        trade_event_handler: TradeEventHandler,
        prediction_loader: PredictionLoader,
        alert_dispatcher: AlertDispatcher,
        config: Config,
    ):
        """Initialize the trade price monitor.

        Args:
            trade_event_handler: Handler providing active trade data
            prediction_loader: Loader for fetching prediction data
            alert_dispatcher: Dispatcher for sending alerts to users
            config: Application configuration
        """
        self.trade_event_handler = trade_event_handler
        self.prediction_loader = prediction_loader
        self.alert_dispatcher = alert_dispatcher
        self.config = config

        # Cooldown tracking: trade_id -> CooldownEntry
        self._cooldowns: dict[str, CooldownEntry] = {}

        # Register callback with trade event handler
        self.trade_event_handler.register_callback(self._on_trade_event)

    def _on_trade_event(self, event_type: TradeEventType, trade_id: str) -> None:
        """Handle trade lifecycle events to manage cooldown state.

        Args:
            event_type: Type of trade event
            trade_id: Identifier of the affected trade
        """
        if event_type == TradeEventType.TRADE_UPDATED:
            # Clear cooldown so trade is re-evaluated next cycle
            self._cooldowns.pop(trade_id, None)
            logger.debug(
                "Cleared price monitor cooldown for updated trade",
                trade_id=trade_id,
            )
        elif event_type in (
            TradeEventType.TRADE_COMPLETED,
            TradeEventType.TRADE_CANCELLED,
        ):
            # Remove cooldown entry entirely
            self._cooldowns.pop(trade_id, None)
            logger.debug(
                "Removed price monitor cooldown for closed trade",
                trade_id=trade_id,
                event_type=event_type.value,
            )

    async def run(self) -> None:
        """Run the monitoring loop indefinitely.

        Sleeps for config.price_drop_monitor_interval between cycles.
        Catches and logs exceptions to prevent the background task from dying.
        """
        logger.info(
            "Trade price monitor loop started",
            interval=self.config.price_drop_monitor_interval,
        )

        while True:
            try:
                await asyncio.sleep(self.config.price_drop_monitor_interval)
                await self._monitor_cycle()
            except asyncio.CancelledError:
                logger.info("Trade price monitor loop cancelled")
                raise
            except Exception:
                logger.exception("Error in trade price monitor cycle")

    async def _monitor_cycle(self) -> None:
        """Execute a single monitoring cycle.

        Checks prediction staleness, gathers active trades, fetches predictions,
        and evaluates each trade for potential price drop alerts.
        """
        # Check prediction freshness
        prediction_age = self.prediction_loader.get_prediction_age_seconds()
        if prediction_age > self.config.data_stale_seconds:
            logger.warning(
                "Skipping price monitor cycle: predictions are stale",
                prediction_age_seconds=prediction_age,
                stale_threshold=self.config.data_stale_seconds,
            )
            return

        # Get active trades
        active_trades = self.trade_event_handler.get_active_trades()
        if not active_trades:
            logger.debug("No active trades to monitor")
            return

        # Group by item_id for batched prediction queries
        item_ids = list(
            {trade.payload.item_id for trade in active_trades.values()}
        )

        logger.debug(
            "Monitoring active trades",
            trade_count=len(active_trades),
            unique_items=len(item_ids),
        )

        # Fetch predictions for all monitored items in one batch
        predictions_df = self.prediction_loader.get_predictions_for_items(item_ids)
        if predictions_df.empty:
            logger.warning("No predictions found for monitored items")
            return

        # Evaluate each trade
        now = datetime.now(timezone.utc).timestamp()
        for trade_id, trade in active_trades.items():
            try:
                self._evaluate_trade(trade_id, trade, predictions_df, now)
            except Exception:
                logger.exception(
                    "Error evaluating trade for price drop",
                    trade_id=trade_id,
                    item_id=trade.payload.item_id,
                )

    def _evaluate_trade(
        self,
        trade_id: str,
        trade: TradeEvent,
        predictions_df,
        now: float,
    ) -> None:
        """Evaluate a single trade against predictions for price drop alerts.

        Args:
            trade_id: Trade identifier
            trade: Trade event data
            predictions_df: DataFrame of predictions for monitored items
            now: Current timestamp (epoch seconds)
        """
        item_id = trade.payload.item_id
        sell_price = trade.payload.sell_price
        buy_price = trade.payload.buy_price
        quantity = trade.payload.quantity

        # Compute remaining hours in the trade window.
        # If the trade has expected_hours and created_at, use the remaining
        # time to cap the prediction horizon. Otherwise fall back to 12h.
        max_hour_offset = 12  # default
        if trade.payload.expected_hours and trade.payload.created_at:
            elapsed_hours = (
                datetime.now(timezone.utc) - trade.payload.created_at
            ).total_seconds() / 3600
            remaining = trade.payload.expected_hours - elapsed_hours
            # Clamp to at least 1h and at most 48h (predictions table range)
            max_hour_offset = max(1, min(48, math.ceil(remaining)))

        # Filter predictions for this item within the remaining trade window,
        # using the lowest offset_pct (most conservative margin) per hour_offset.
        # This gives the most realistic sell price the model expects to fill.
        item_preds = predictions_df[
            (predictions_df["item_id"] == item_id)
            & (predictions_df["hour_offset"] >= 1)
            & (predictions_df["hour_offset"] <= max_hour_offset)
        ]

        if item_preds.empty:
            return

        # For each hour_offset, take the row with the highest offset_pct
        # (= widest margin = most conservative/lowest-risk sell price)
        conservative_preds = item_preds.loc[
            item_preds.groupby("hour_offset")["offset_pct"].idxmax()
        ]

        # Best predicted sell = max across all hour offsets at conservative margin
        best_predicted_sell = conservative_preds["sell_price"].max()

        if best_predicted_sell is None or best_predicted_sell <= 0:
            return

        # Compute drop percentage
        if sell_price <= 0:
            return

        drop_pct = (sell_price - best_predicted_sell) / sell_price

        # Check if drop exceeds minimum threshold
        if drop_pct < self.config.price_drop_min_pct:
            return

        # Determine alert severity
        if (
            drop_pct > self.config.price_drop_high_pct
            or best_predicted_sell <= buy_price
        ):
            urgency = AlertUrgency.HIGH
            cooldown = self.config.price_drop_cooldown_high
            is_sell_now = True
        elif drop_pct >= self.config.price_drop_medium_pct:
            urgency = AlertUrgency.MEDIUM
            cooldown = self.config.price_drop_cooldown_medium
            is_sell_now = False
        else:
            urgency = AlertUrgency.LOW
            cooldown = self.config.price_drop_cooldown_low
            is_sell_now = False

        # Suggested price: 0.5% above predicted sell
        suggested_sell = round(best_predicted_sell * 1.005)

        # Check cooldown and price change deduplication
        existing = self._cooldowns.get(trade_id)
        if existing is not None:
            time_since_last = now - existing.last_alert_time
            if time_since_last < cooldown:
                # Within cooldown window — check if price changed significantly
                if existing.last_suggested_price > 0:
                    price_change_pct = abs(
                        suggested_sell - existing.last_suggested_price
                    ) / existing.last_suggested_price
                    if price_change_pct <= 0.01:
                        # Price hasn't changed by more than 1%, skip
                        return
                else:
                    return

        # Compute profit delta
        original_profit = (sell_price - buy_price) * quantity
        suggested_profit = (suggested_sell - buy_price) * quantity
        profit_delta = suggested_profit - original_profit

        # Build alert
        alert_id = f"pda_{trade_id}_{int(datetime.now(timezone.utc).timestamp())}"

        # Get best row's fill probability for confidence
        best_row = conservative_preds.loc[conservative_preds["sell_price"].idxmax()]
        confidence = float(best_row["fill_probability"])

        if is_sell_now:
            reason = (
                f"Price predicted to drop {drop_pct:.1%}. "
                f"Predicted sell {int(best_predicted_sell)}gp "
                f"{'is at or below' if best_predicted_sell <= buy_price else 'vs'} "
                f"buy price {buy_price}gp."
            )
            alert = create_sell_now_alert(
                alert_id=alert_id,
                trade_id=trade_id,
                reason=reason,
                confidence=confidence,
                urgency=urgency,
                adjusted_sell_price=suggested_sell,
                profit_delta=profit_delta,
            )
        else:
            reason = (
                f"Price predicted to drop {drop_pct:.1%}. "
                f"Consider adjusting sell price from {sell_price}gp "
                f"to {suggested_sell}gp."
            )
            alert = create_adjust_price_alert(
                alert_id=alert_id,
                trade_id=trade_id,
                reason=reason,
                confidence=confidence,
                urgency=urgency,
                new_sell_price=suggested_sell,
                original_sell_price=sell_price,
                profit_delta=profit_delta,
            )

        # Update cooldown
        self._cooldowns[trade_id] = CooldownEntry(
            last_alert_time=now,
            last_suggested_price=suggested_sell,
        )

        # Dispatch alert (fire-and-forget)
        self.alert_dispatcher.dispatch_async(trade.user_id, alert)

        logger.info(
            "Price drop alert dispatched",
            trade_id=trade_id,
            item_id=item_id,
            drop_pct=f"{drop_pct:.2%}",
            urgency=urgency.value,
            alert_type="SELL_NOW" if is_sell_now else "ADJUST_PRICE",
            suggested_sell=suggested_sell,
            original_sell=sell_price,
        )
