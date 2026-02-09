from datetime import datetime, timezone

import pandas as pd

from src.config import Config
from src.trade_events import TradeEvent, TradeEventType, TradePayload
from src.trade_price_monitor import TradePriceMonitor


class _DummyEngine:
    """Only provides the attributes TradePriceMonitor touches in __init__."""

    def __init__(self):
        self.loader = object()
        self._beta_loader = None


class _DummyDispatcher:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def dispatch_async(self, user_id: str, alert) -> None:
        self.calls.append((user_id, alert))


class _DummyTradeEventHandler:
    def register_callback(self, _callback) -> None:
        # TradePriceMonitor registers a callback during init; tests call
        # _evaluate_trade directly so we don't need to store it.
        return None


def test_price_monitor_uses_lowest_offset_pct_for_conservative_sell():
    """
    Regression test:
    The monitor should use the *lowest* offset_pct per hour_offset as the
    conservative sell target. Using the highest offset makes the predicted sell
    unrealistically optimistic and suppresses alerts.
    """
    cfg = Config()
    # Keep defaults; just ensure we trigger an alert deterministically.
    cfg.price_drop_min_pct = 0.02
    cfg.price_drop_medium_pct = 0.05
    cfg.price_drop_high_pct = 0.10

    dispatcher = _DummyDispatcher()
    handler = _DummyTradeEventHandler()
    monitor = TradePriceMonitor(
        trade_event_handler=handler,  # type: ignore[arg-type]
        recommendation_engine=_DummyEngine(),  # type: ignore[arg-type]
        alert_dispatcher=dispatcher,  # type: ignore[arg-type]
        config=cfg,
    )

    # Trade is currently trying to sell at 110.
    trade = TradeEvent(
        event_type=TradeEventType.TRADE_CREATED,
        timestamp=datetime.now(timezone.utc),
        user_id="user123",
        trade_id="t1",
        payload=TradePayload(
            item_id=1,
            item_name="Test item",
            buy_price=90,
            sell_price=110,
            quantity=10,
            expected_hours=12,
            created_at=datetime.now(timezone.utc),
        ),
    )

    # Two offsets for each hour. Higher offset => higher sell_price (harder to fill).
    # Lowest offsets predict ~100-101; highest offsets predict ~110-111.
    preds = pd.DataFrame(
        [
            {"item_id": 1, "hour_offset": 1, "offset_pct": 0.0125, "sell_price": 100, "fill_probability": 0.70},
            {"item_id": 1, "hour_offset": 1, "offset_pct": 0.0250, "sell_price": 110, "fill_probability": 0.40},
            {"item_id": 1, "hour_offset": 2, "offset_pct": 0.0125, "sell_price": 101, "fill_probability": 0.68},
            {"item_id": 1, "hour_offset": 2, "offset_pct": 0.0250, "sell_price": 111, "fill_probability": 0.38},
        ]
    )

    monitor._evaluate_trade(  # pylint: disable=protected-access
        trade_id=trade.trade_id,
        trade=trade,
        predictions_df=preds,
        now=1000.0,
    )

    # With the correct lowest-offset selection, best_predicted_sell becomes 101
    # and drop_pct = (110-101)/110 ~= 8.2% -> triggers an alert.
    assert len(dispatcher.calls) == 1
    user_id, alert = dispatcher.calls[0]
    assert user_id == "user123"
    assert getattr(alert, "type").value == "ADJUST_PRICE"
    assert getattr(alert, "newSellPrice") is not None
    assert getattr(alert, "newSellPrice") < 110
