"""Tests for alert dispatcher."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.alert_dispatcher import (
    Alert,
    AlertDispatcher,
    AlertType,
    AlertUrgency,
    NewItemRecommendation,
    create_adjust_price_alert,
    create_sell_now_alert,
    create_switch_item_alert,
)


@pytest.fixture
def sample_alert():
    """Create a sample alert."""
    return Alert(
        id="alert_123",
        tradeId="trade_xyz",
        type=AlertType.SELL_NOW,
        reason="Market price dropping, sell now to minimize loss",
        confidence=0.85,
        urgency=AlertUrgency.HIGH,
        timestamp="2024-01-15T12:00:00Z",
        adjustedSellPrice=1180000,
        profitDelta=-20000,
    )


@pytest.fixture
def dispatcher():
    """Create an alert dispatcher with test config."""
    return AlertDispatcher(
        webhook_url="https://example.com/api/webhooks/alerts",
        webhook_secret="test_secret_key_12345",
    )


class TestAlertDispatcher:
    """Tests for AlertDispatcher class."""

    @pytest.mark.asyncio
    async def test_dispatch_success(self, dispatcher, sample_alert):
        """Test successful alert dispatch."""
        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await dispatcher.dispatch("user_abc123", sample_alert)

            assert result is True
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_with_retry(self, dispatcher, sample_alert):
        """Test alert dispatch with retry on first failure."""
        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()

            # First call fails, second succeeds
            mock_response_fail = MagicMock()
            mock_response_fail.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )

            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            mock_response_success.raise_for_status = MagicMock()

            mock_client.post = AsyncMock(
                side_effect=[mock_response_fail, mock_response_success]
            )
            mock_get_client.return_value = mock_client

            result = await dispatcher.dispatch("user_abc123", sample_alert)

            assert result is True
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_dispatch_fails_after_retry(self, dispatcher, sample_alert):
        """Test alert dispatch fails after retry exhausted."""
        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()

            mock_response_fail = MagicMock()
            mock_response_fail.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )

            mock_client.post = AsyncMock(return_value=mock_response_fail)
            mock_get_client.return_value = mock_client

            result = await dispatcher.dispatch("user_abc123", sample_alert)

            assert result is False
            assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_dispatch_timeout(self, dispatcher, sample_alert):
        """Test alert dispatch handles timeout."""
        with patch.object(dispatcher, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
            mock_get_client.return_value = mock_client

            result = await dispatcher.dispatch("user_abc123", sample_alert)

            assert result is False

    @pytest.mark.asyncio
    async def test_dispatch_no_webhook_url(self, sample_alert):
        """Test dispatch skips when no webhook URL configured."""
        dispatcher = AlertDispatcher(webhook_url="", webhook_secret="secret")

        result = await dispatcher.dispatch("user_abc123", sample_alert)

        assert result is False

    @pytest.mark.asyncio
    async def test_dispatch_no_webhook_secret(self, sample_alert):
        """Test dispatch skips when no webhook secret configured."""
        dispatcher = AlertDispatcher(
            webhook_url="https://example.com/webhook", webhook_secret=""
        )

        result = await dispatcher.dispatch("user_abc123", sample_alert)

        assert result is False

    def test_serialize_alert_basic(self, dispatcher, sample_alert):
        """Test alert serialization."""
        body = dispatcher._serialize_alert("user_123", sample_alert)
        data = json.loads(body)

        assert data["userId"] == "user_123"
        assert data["alert"]["id"] == "alert_123"
        assert data["alert"]["tradeId"] == "trade_xyz"
        assert data["alert"]["type"] == "SELL_NOW"
        assert data["alert"]["confidence"] == 0.85
        assert data["alert"]["urgency"] == "high"
        assert data["alert"]["adjustedSellPrice"] == 1180000

    def test_serialize_alert_with_new_item(self, dispatcher):
        """Test alert serialization with new item recommendation."""
        new_item = NewItemRecommendation(
            itemId=4151,
            item="Abyssal whip",
            buyPrice=2500000,
            sellPrice=2600000,
            quantity=1,
            expectedProfit=80000,
            confidence=0.9,
        )
        alert = Alert(
            id="alert_456",
            tradeId="trade_abc",
            type=AlertType.SWITCH_ITEM,
            reason="Better opportunity found",
            confidence=0.92,
            urgency=AlertUrgency.HIGH,
            timestamp="2024-01-15T12:00:00Z",
            newItem=new_item,
            profitDelta=50000,
        )

        body = dispatcher._serialize_alert("user_123", alert)
        data = json.loads(body)

        assert data["alert"]["type"] == "SWITCH_ITEM"
        assert data["alert"]["newItem"]["itemId"] == 4151
        assert data["alert"]["newItem"]["item"] == "Abyssal whip"
        assert data["alert"]["newItem"]["expectedProfit"] == 80000


class TestAlertFactories:
    """Tests for alert factory functions."""

    def test_create_switch_item_alert(self):
        """Test creating a SWITCH_ITEM alert."""
        new_item = NewItemRecommendation(
            itemId=4151,
            item="Abyssal whip",
            buyPrice=2500000,
            sellPrice=2600000,
            quantity=1,
            expectedProfit=80000,
            confidence=0.9,
        )

        alert = create_switch_item_alert(
            alert_id="alert_1",
            trade_id="trade_1",
            reason="Better opportunity found",
            confidence=0.92,
            urgency=AlertUrgency.HIGH,
            new_item=new_item,
            profit_delta=50000,
        )

        assert alert.type == AlertType.SWITCH_ITEM
        assert alert.newItem == new_item
        assert alert.profitDelta == 50000

    def test_create_sell_now_alert(self):
        """Test creating a SELL_NOW alert."""
        alert = create_sell_now_alert(
            alert_id="alert_2",
            trade_id="trade_2",
            reason="Market dropping, sell immediately",
            confidence=0.88,
            urgency=AlertUrgency.HIGH,
            adjusted_sell_price=1180000,
            profit_delta=-20000,
        )

        assert alert.type == AlertType.SELL_NOW
        assert alert.adjustedSellPrice == 1180000
        assert alert.profitDelta == -20000

    def test_create_adjust_price_alert(self):
        """Test creating an ADJUST_PRICE alert."""
        alert = create_adjust_price_alert(
            alert_id="alert_3",
            trade_id="trade_3",
            reason="Reduce sell price for faster fill",
            confidence=0.75,
            urgency=AlertUrgency.MEDIUM,
            new_sell_price=1220000,
            original_sell_price=1250000,
            profit_delta=-30000,
        )

        assert alert.type == AlertType.ADJUST_PRICE
        assert alert.newSellPrice == 1220000
        assert alert.originalSellPrice == 1250000
        assert alert.profitDelta == -30000


class TestAlertTypes:
    """Tests for AlertType enum."""

    def test_alert_types(self):
        """Test that all alert types are defined."""
        assert AlertType.SWITCH_ITEM.value == "SWITCH_ITEM"
        assert AlertType.SELL_NOW.value == "SELL_NOW"
        assert AlertType.ADJUST_PRICE.value == "ADJUST_PRICE"
        assert AlertType.HOLD.value == "HOLD"


class TestAlertUrgency:
    """Tests for AlertUrgency enum."""

    def test_urgency_levels(self):
        """Test that all urgency levels are defined."""
        assert AlertUrgency.LOW.value == "low"
        assert AlertUrgency.MEDIUM.value == "medium"
        assert AlertUrgency.HIGH.value == "high"
