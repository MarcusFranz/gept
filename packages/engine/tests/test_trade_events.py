"""Tests for trade event handling."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.trade_events import (
    TradeEvent,
    TradeEventHandler,
    TradeEventType,
    TradePayload,
)


@pytest.fixture
def mock_crowding_tracker():
    """Create a mock crowding tracker."""
    tracker = MagicMock()
    tracker.record_delivery = MagicMock()
    tracker.release_position = MagicMock(return_value=True)
    return tracker


@pytest.fixture
def handler(mock_crowding_tracker):
    """Create a trade event handler with mocked dependencies."""
    return TradeEventHandler(
        crowding_tracker=mock_crowding_tracker,
        recommendation_engine=None,
    )


@pytest.fixture
def sample_payload():
    """Create a sample trade payload."""
    return TradePayload(
        item_id=4151,
        item_name="Abyssal whip",
        buy_price=2500000,
        sell_price=2600000,
        quantity=1,
        rec_id="rec_4151_2024011512",
        model_id="model_v3",
    )


@pytest.fixture
def sample_event(sample_payload):
    """Create a sample trade event."""
    return TradeEvent(
        event_type=TradeEventType.TRADE_CREATED,
        timestamp=datetime.now(timezone.utc),
        user_id="user_abc123",
        trade_id="trade_xyz789",
        payload=sample_payload,
    )


class TestTradeEventHandler:
    """Tests for TradeEventHandler."""

    @pytest.mark.asyncio
    async def test_handle_trade_created(
        self, handler, sample_event, mock_crowding_tracker
    ):
        """Test handling TRADE_CREATED event."""
        await handler.handle_event(sample_event)

        # Should register with crowding tracker
        mock_crowding_tracker.record_delivery.assert_called_once_with(
            item_id=4151,
            user_id="user_abc123",
        )

        # Should store active trade
        assert "trade_xyz789" in handler._active_trades
        assert handler._active_trades["trade_xyz789"] == sample_event

    @pytest.mark.asyncio
    async def test_handle_trade_completed(
        self, handler, sample_event, mock_crowding_tracker
    ):
        """Test handling TRADE_COMPLETED event."""
        # First create the trade
        await handler.handle_event(sample_event)

        # Now complete it
        completed_event = TradeEvent(
            event_type=TradeEventType.TRADE_COMPLETED,
            timestamp=datetime.now(timezone.utc),
            user_id="user_abc123",
            trade_id="trade_xyz789",
            payload=TradePayload(
                item_id=4151,
                item_name="Abyssal whip",
                buy_price=2500000,
                sell_price=2600000,
                quantity=1,
                profit=80000,  # Net profit after tax
            ),
        )
        await handler.handle_event(completed_event)

        # Should release crowding position
        mock_crowding_tracker.release_position.assert_called_once_with(
            item_id=4151,
            user_id="user_abc123",
        )

        # Should remove from active trades
        assert "trade_xyz789" not in handler._active_trades

    @pytest.mark.asyncio
    async def test_handle_trade_cancelled(
        self, handler, sample_event, mock_crowding_tracker
    ):
        """Test handling TRADE_CANCELLED event."""
        # First create the trade
        await handler.handle_event(sample_event)

        # Now cancel it
        cancelled_event = TradeEvent(
            event_type=TradeEventType.TRADE_CANCELLED,
            timestamp=datetime.now(timezone.utc),
            user_id="user_abc123",
            trade_id="trade_xyz789",
            payload=sample_event.payload,
        )
        await handler.handle_event(cancelled_event)

        # Should release crowding position
        mock_crowding_tracker.release_position.assert_called_once()

        # Should remove from active trades
        assert "trade_xyz789" not in handler._active_trades

    @pytest.mark.asyncio
    async def test_handle_trade_updated(self, handler, sample_event):
        """Test handling TRADE_UPDATED event."""
        # First create the trade
        await handler.handle_event(sample_event)

        # Now update it with new quantity
        updated_payload = TradePayload(
            item_id=4151,
            item_name="Abyssal whip",
            buy_price=2500000,
            sell_price=2600000,
            quantity=2,  # Changed quantity
        )
        updated_event = TradeEvent(
            event_type=TradeEventType.TRADE_UPDATED,
            timestamp=datetime.now(timezone.utc),
            user_id="user_abc123",
            trade_id="trade_xyz789",
            payload=updated_payload,
        )
        await handler.handle_event(updated_event)

        # Should update the stored trade
        assert handler._active_trades["trade_xyz789"].payload.quantity == 2

    def test_get_active_trades(self, handler, sample_event):
        """Test getting all active trades."""
        # Add a trade manually
        handler._active_trades["trade_1"] = sample_event

        trades = handler.get_active_trades()
        assert "trade_1" in trades
        assert trades["trade_1"] == sample_event

    def test_get_active_trades_for_user(self, handler, sample_event):
        """Test getting active trades for a specific user."""
        handler._active_trades["trade_1"] = sample_event

        trades = handler.get_active_trades_for_user("user_abc123")
        assert len(trades) == 1
        assert trades[0] == sample_event

        # Different user should return empty
        trades = handler.get_active_trades_for_user("other_user")
        assert len(trades) == 0

    def test_get_active_trades_for_item(self, handler, sample_event):
        """Test getting active trades for a specific item."""
        handler._active_trades["trade_1"] = sample_event

        trades = handler.get_active_trades_for_item(4151)
        assert len(trades) == 1
        assert trades[0] == sample_event

        # Different item should return empty
        trades = handler.get_active_trades_for_item(12345)
        assert len(trades) == 0


class TestTradeEventType:
    """Tests for TradeEventType enum."""

    def test_event_types(self):
        """Test that all event types are defined."""
        assert TradeEventType.TRADE_CREATED.value == "TRADE_CREATED"
        assert TradeEventType.TRADE_COMPLETED.value == "TRADE_COMPLETED"
        assert TradeEventType.TRADE_CANCELLED.value == "TRADE_CANCELLED"
        assert TradeEventType.TRADE_UPDATED.value == "TRADE_UPDATED"

    def test_event_type_from_string(self):
        """Test creating event type from string."""
        assert TradeEventType("TRADE_CREATED") == TradeEventType.TRADE_CREATED


class TestTradePayload:
    """Tests for TradePayload dataclass."""

    def test_create_payload(self):
        """Test creating a trade payload."""
        payload = TradePayload(
            item_id=4151,
            item_name="Abyssal whip",
            buy_price=2500000,
            sell_price=2600000,
            quantity=1,
        )
        assert payload.item_id == 4151
        assert payload.profit is None
        assert payload.rec_id is None

    def test_create_payload_with_optional_fields(self):
        """Test creating a payload with optional fields."""
        payload = TradePayload(
            item_id=4151,
            item_name="Abyssal whip",
            buy_price=2500000,
            sell_price=2600000,
            quantity=1,
            profit=80000,
            rec_id="rec_123",
            model_id="model_v3",
        )
        assert payload.profit == 80000
        assert payload.rec_id == "rec_123"
        assert payload.model_id == "model_v3"
