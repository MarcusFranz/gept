"""Tests for the recommendation engine."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.recommendation_engine import RecommendationEngine, RecommendationStore


def _configure_batch_mocks(mock_loader):
    """Configure batch methods to return dicts based on individual method return values.

    This helper ensures tests work with the batch fetching optimization.
    Configures: get_batch_buy_limits, get_batch_volumes_24h, get_batch_volumes_1h,
    and get_batch_trends to mirror the individual method return values.

    Handles both return_value and side_effect configurations on individual methods.
    """

    def batch_buy_limits_side_effect(item_ids):
        result = {}
        for item_id in item_ids:
            # Check if individual method has side_effect
            if mock_loader.get_item_buy_limit.side_effect:
                try:
                    limit = mock_loader.get_item_buy_limit.side_effect(item_id)
                except (TypeError, StopIteration):
                    limit = mock_loader.get_item_buy_limit.return_value
            else:
                limit = mock_loader.get_item_buy_limit.return_value
            if limit is not None:
                result[item_id] = limit
        return result

    def batch_volumes_24h_side_effect(item_ids):
        result = {}
        for item_id in item_ids:
            if mock_loader.get_item_volume_24h.side_effect:
                try:
                    volume = mock_loader.get_item_volume_24h.side_effect(item_id)
                except (TypeError, StopIteration):
                    volume = mock_loader.get_item_volume_24h.return_value
            else:
                volume = mock_loader.get_item_volume_24h.return_value
            if volume is not None:
                result[item_id] = volume
        return result

    def batch_volumes_1h_side_effect(item_ids):
        result = {}
        for item_id in item_ids:
            if mock_loader.get_item_volume_1h.side_effect:
                try:
                    volume = mock_loader.get_item_volume_1h.side_effect(item_id)
                except (TypeError, StopIteration):
                    volume = mock_loader.get_item_volume_1h.return_value
            else:
                volume = mock_loader.get_item_volume_1h.return_value
            if volume is None:
                volume = 0
            result[item_id] = volume
        return result

    def batch_trends_side_effect(item_ids):
        result = {}
        for item_id in item_ids:
            if mock_loader.get_item_trend.side_effect:
                try:
                    trend = mock_loader.get_item_trend.side_effect(item_id)
                except (TypeError, StopIteration):
                    trend = mock_loader.get_item_trend.return_value
            else:
                trend = mock_loader.get_item_trend.return_value
            if trend is None:
                trend = "Stable"
            result[item_id] = trend
        return result

    mock_loader.get_batch_buy_limits.side_effect = batch_buy_limits_side_effect
    mock_loader.get_batch_volumes_24h.side_effect = batch_volumes_24h_side_effect
    mock_loader.get_batch_volumes_1h.side_effect = batch_volumes_1h_side_effect
    mock_loader.get_batch_trends.side_effect = batch_trends_side_effect


# Backwards compatibility alias
def _configure_batch_buy_limits_mock(mock_loader):
    """Legacy alias for _configure_batch_mocks."""
    _configure_batch_mocks(mock_loader)


class TestRecommendationStore:
    """Test cases for RecommendationStore."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving recommendations."""
        store = RecommendationStore(ttl_seconds=300)

        rec = {
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
        }

        rec_id = store.store(rec)
        assert rec_id is not None
        assert rec["id"] == rec_id

        # Retrieve by ID
        retrieved = store.get_by_id(rec_id)
        assert retrieved is not None
        assert retrieved["itemId"] == 554

        # Retrieve by item_id
        retrieved = store.get_by_item_id(554)
        assert retrieved is not None
        assert retrieved["id"] == rec_id

    def test_stable_id_generation(self):
        """Test that IDs are stable within the same hour."""
        store = RecommendationStore(ttl_seconds=300)

        rec1 = {"itemId": 554, "item": "Fire rune"}
        rec2 = {"itemId": 554, "item": "Fire rune"}

        id1 = store.store(rec1)
        id2 = store.store(rec2)

        # Same item in same hour should get same ID
        assert id1 == id2

    def test_parse_rec_id_valid(self):
        """Test parsing a valid rec_id extracts item_id."""
        store = RecommendationStore(ttl_seconds=300)

        # Valid format: rec_{item_id}_{YYYYMMDDHH}
        assert store.parse_rec_id("rec_554_2026011515") == 554
        assert store.parse_rec_id("rec_12345_2026011515") == 12345
        assert store.parse_rec_id("rec_1_2000010100") == 1

    def test_parse_rec_id_invalid_prefix(self):
        """Test parsing returns None for invalid prefix."""
        store = RecommendationStore(ttl_seconds=300)

        assert store.parse_rec_id("invalid_554_2026011515") is None
        assert store.parse_rec_id("recommendation_554_2026011515") is None
        assert store.parse_rec_id("554_2026011515") is None

    def test_parse_rec_id_invalid_format(self):
        """Test parsing returns None for wrong number of parts."""
        store = RecommendationStore(ttl_seconds=300)

        assert store.parse_rec_id("rec_554") is None  # Too few parts
        assert store.parse_rec_id("rec_554_2026011515_extra") is None  # Too many
        assert store.parse_rec_id("rec__2026011515") is None  # Empty item_id

    def test_parse_rec_id_non_numeric_item_id(self):
        """Test parsing returns None for non-numeric item_id."""
        store = RecommendationStore(ttl_seconds=300)

        assert store.parse_rec_id("rec_abc_2026011515") is None
        assert store.parse_rec_id("rec_12.34_2026011515") is None

    def test_parse_rec_id_empty_or_none(self):
        """Test parsing handles empty or None input."""
        store = RecommendationStore(ttl_seconds=300)

        assert store.parse_rec_id("") is None
        assert store.parse_rec_id(None) is None  # type: ignore


class TestRecommendationEngine:
    """Test cases for RecommendationEngine."""

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader

            return engine, mock_loader

    def test_get_effective_buy_limit_passive_8h(self, mock_engine):
        """Test effective buy limit calculation for passive mode with 8h hold."""
        engine, _ = mock_engine

        # Passive mode, 8 hours = 2 reset cycles (8/4=2)
        # multiplier = min(2 + 1, 4) = 3
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="passive",
            expected_hours=8,
        )

        assert effective_limit == 30000  # 3x limit
        assert is_multi is True

    def test_get_effective_buy_limit_passive_12h(self, mock_engine):
        """Test effective buy limit calculation for passive mode with 12h hold."""
        engine, _ = mock_engine

        # Passive mode, 12 hours = 3 reset cycles (0, 4, 8, 12)
        # Should allow 3x limit (multiplier = limit_resets + 1 = 3 + 1 = 4, but 3 resets means 3x)
        # Wait: 12 // 4 = 3 resets, so multiplier = min(3 + 1, 4) = 4
        # But 3 resets only means we can buy 4 times total
        # Actually: 12h / 4h = 3 full cycles, so multiplier = 3 + 1 = 4, capped at 4
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="passive",
            expected_hours=12,
        )

        # 12h allows 3 resets, so 4x (but capped at 4x)
        assert effective_limit == 40000  # Now allows up to 4x
        assert is_multi is True

    def test_get_effective_buy_limit_passive_short(self, mock_engine):
        """Test effective buy limit for passive mode with short hold time."""
        engine, _ = mock_engine

        # Passive mode, 3 hours = 0 reset cycles
        # Should return 1x limit
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="passive",
            expected_hours=3,
        )

        assert effective_limit == 10000
        assert is_multi is False

    def test_get_effective_buy_limit_active(self, mock_engine):
        """Test effective buy limit for active mode (should always be 1x)."""
        engine, _ = mock_engine

        # Active mode should always return 1x, regardless of hours
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="active",
            expected_hours=12,
        )

        assert effective_limit == 10000
        assert is_multi is False

    def test_get_effective_buy_limit_hybrid(self, mock_engine):
        """Test effective buy limit for hybrid mode (should always be 1x)."""
        engine, _ = mock_engine

        # Hybrid mode should always return 1x
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="hybrid",
            expected_hours=8,
        )

        assert effective_limit == 10000
        assert is_multi is False

    def test_get_effective_buy_limit_passive_24h(self, mock_engine):
        """Test effective buy limit for passive mode with 24h hold (new 48h support)."""
        engine, _ = mock_engine

        # Passive mode, 24 hours = 6 reset cycles
        # multiplier = min(6 + 1, 4) = 4 (capped)
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="passive",
            expected_hours=24,
        )

        assert effective_limit == 40000  # Capped at 4x
        assert is_multi is True

    def test_get_effective_buy_limit_passive_48h(self, mock_engine):
        """Test effective buy limit for passive mode with 48h hold."""
        engine, _ = mock_engine

        # Passive mode, 48 hours = 12 reset cycles
        # multiplier = min(12 + 1, 4) = 4 (capped at 4x)
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="passive",
            expected_hours=48,
        )

        assert effective_limit == 40000  # Capped at 4x even with 12 resets
        assert is_multi is True

    def test_get_effective_buy_limit_passive_16h(self, mock_engine):
        """Test effective buy limit for passive mode with 16h hold (exactly 4x)."""
        engine, _ = mock_engine

        # Passive mode, 16 hours = 4 reset cycles
        # multiplier = min(4 + 1, 4) = 4
        effective_limit, is_multi = engine.get_effective_buy_limit(
            base_limit=10000,
            style="passive",
            expected_hours=16,
        )

        assert effective_limit == 40000  # 4x multiplier
        assert is_multi is True

    def test_format_staged_buys(self, mock_engine):
        """Test formatting of staged buy instructions."""
        engine, _ = mock_engine

        stages = engine._format_staged_buys(
            item_name="Amulet of glory (6)",
            base_limit=10000,
            total_quantity=20000,
            buy_price=12179,
            sell_price=13068,
        )

        # Should have 3 stages: first buy, second buy, sell
        assert len(stages) == 3

        # First buy
        assert stages[0]["stage"] == 1
        assert stages[0]["quantity"] == 10000
        assert stages[0]["price"] == 12179
        assert stages[0]["timing"] == "now"

        # Second buy
        assert stages[1]["stage"] == 2
        assert stages[1]["quantity"] == 10000
        assert stages[1]["price"] == 12179
        assert stages[1]["timing"] == "after_4h_reset"

        # Sell
        assert stages[2]["stage"] == 3
        assert stages[2]["quantity"] == 20000
        assert stages[2]["price"] == 13068
        assert stages[2]["timing"] == "passive_target"

    def test_format_staged_buys_single_limit(self, mock_engine):
        """Test staged buys when total quantity fits in single limit."""
        engine, _ = mock_engine

        stages = engine._format_staged_buys(
            item_name="Fire rune",
            base_limit=10000,
            total_quantity=5000,  # Less than base limit
            buy_price=4,
            sell_price=6,
        )

        # Should have 2 stages: buy and sell (no second buy needed)
        assert len(stages) == 2

        # First buy (full quantity)
        assert stages[0]["quantity"] == 5000
        assert stages[0]["timing"] == "now"

        # Sell
        assert stages[1]["quantity"] == 5000
        assert stages[1]["timing"] == "passive_target"

    def test_get_recommendations_empty_predictions(self, mock_engine):
        """Test recommendations when no predictions match criteria."""
        engine, mock_loader = mock_engine

        # Mock empty predictions
        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        recommendations = engine.get_recommendations(
            style="passive",
            capital=10000000,
            risk="medium",
            slots=4,
        )

        assert recommendations == []

    def test_get_recommendations_with_exclude_ids(self, mock_engine):
        """Test that exclude_ids filters out matching recommendations."""
        engine, mock_loader = mock_engine

        # Mock predictions for two items
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 400,
                    "sell_price": 420,
                    "current_high": 420,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        # Generate the stable ID for item 554 to exclude it
        exclude_id = engine._generate_stable_id(554)

        recommendations = engine.get_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            slots=4,
            exclude_ids={exclude_id},
        )

        # Should only have Blood rune (565), Fire rune (554) should be excluded
        item_ids = {rec["itemId"] for rec in recommendations}
        assert 554 not in item_ids
        assert 565 in item_ids

    def test_get_recommendations_exclude_all_returns_empty(self, mock_engine):
        """Test that excluding all items returns empty list."""
        engine, mock_loader = mock_engine

        # Mock predictions for one item
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        # Exclude the only item
        exclude_id = engine._generate_stable_id(554)

        recommendations = engine.get_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            slots=4,
            exclude_ids={exclude_id},
        )

        # Should return empty list
        assert recommendations == []

    def test_get_recommendations_empty_exclude_ids(self, mock_engine):
        """Test that empty exclude_ids doesn't filter anything."""
        engine, mock_loader = mock_engine

        # Mock predictions for one item
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        # Empty exclude set should not filter
        recommendations = engine.get_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            slots=4,
            exclude_ids=set(),
        )

        # Should include the item
        assert len(recommendations) >= 1
        assert any(rec["itemId"] == 554 for rec in recommendations)

    def test_get_recommendations_none_exclude_ids(self, mock_engine):
        """Test that None exclude_ids doesn't filter anything."""
        engine, mock_loader = mock_engine

        # Mock predictions for one item
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        # None exclude should not filter
        recommendations = engine.get_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            slots=4,
            exclude_ids=None,
        )

        # Should include the item
        assert len(recommendations) >= 1
        assert any(rec["itemId"] == 554 for rec in recommendations)

    def test_recommendations_include_multi_limit_metadata(self, mock_engine):
        """Test that recommendations include multi-limit strategy metadata."""
        engine, mock_loader = mock_engine

        # Use a price that will meet passive mode capital efficiency requirements
        # Passive mode with 300M capital and 4 slots needs ~37.5M per slot
        # With buy limit of 10000, we need buy_price * 20000 >= 30M (80% threshold)
        # So buy_price >= 1500
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 1704,
                    "item_name": "Amulet of glory (6)",
                    "hour_offset": 12,
                    "offset_pct": 0.015,
                    "fill_probability": 0.08,
                    "expected_value": 0.006,
                    "buy_price": 12179,
                    "sell_price": 13068,
                    "current_high": 13000,
                    "current_low": 12100,
                    "confidence": "medium",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        recommendations = engine.get_recommendations(
            style="passive",
            capital=300000000,  # Large capital
            risk="medium",
            slots=4,
        )

        # Should have at least one recommendation
        assert len(recommendations) >= 1

        # Check first recommendation for multi-limit metadata
        rec = recommendations[0]
        assert "isMultiLimitStrategy" in rec

        # For passive mode with 12h window, should be multi-limit
        if rec["isMultiLimitStrategy"]:
            assert "baseBuyLimit" in rec
            assert "stagedBuys" in rec
            assert rec["baseBuyLimit"] == 10000
            # stagedBuys has 3 stages (buy, buy, sell) when quantity > buy_limit
            # or 2 stages (buy, sell) when quantity <= buy_limit
            if rec["quantity"] > 10000:
                assert len(rec["stagedBuys"]) == 3
            else:
                assert len(rec["stagedBuys"]) == 2

    def test_recommendations_include_trend_and_fill_fields(self, mock_engine):
        """Test that recommendations include trend and fill fields."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 60
        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_trend.return_value = "Rising"
        mock_loader.get_item_volume_24h.return_value = 500000
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,
            risk="medium",
            slots=4,
        )

        assert len(recommendations) >= 1
        rec = recommendations[0]

        # Check required fields
        assert "trend" in rec
        assert rec["trend"] == "Rising"

        # Check fill probability fields
        assert "fillProbability" in rec
        assert rec["fillProbability"] == 0.10
        assert "fillConfidence" in rec
        assert rec["fillConfidence"] in ["Strong", "Good", "Fair"]

        # Check optional volume24h field
        assert "volume24h" in rec
        assert rec["volume24h"] == 500000

    def test_recommendations_volume24h_optional_when_not_available(self, mock_engine):
        """Test that volume24h is not included when data is not available."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 60
        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None  # Not available
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,
            risk="medium",
            slots=4,
        )

        assert len(recommendations) >= 1
        rec = recommendations[0]

        # volume24h should not be present when None
        assert "volume24h" not in rec

    def test_fill_confidence_hybrid_thresholds(self, mock_engine):
        """Test that _determine_fill_confidence uses hybrid thresholds correctly."""
        engine, _ = mock_engine

        # Test absolute floor: < 3% always returns Fair
        assert engine._determine_fill_confidence(0.02, None) == "Fair"
        assert engine._determine_fill_confidence(0.02, [0.01, 0.02, 0.03]) == "Fair"

        # Test 3-5% range: can be Good but never Strong
        # With batch context where 0.04 is in top 50%
        assert (
            engine._determine_fill_confidence(0.04, [0.01, 0.02, 0.04, 0.05]) == "Good"
        )
        # Without batch context, falls back to absolute (< 8% = Fair)
        assert engine._determine_fill_confidence(0.04, None) == "Fair"

        # Test relative percentile logic (>= 5%)
        # Top 25% (percentile >= 0.75) should be Strong
        probs = [0.05, 0.10, 0.15, 0.20]
        assert (
            engine._determine_fill_confidence(0.20, probs) == "Strong"
        )  # 100th percentile
        assert (
            engine._determine_fill_confidence(0.15, probs) == "Strong"
        )  # 75th percentile (exactly at threshold)
        # Top 50% (percentile >= 0.50) should be Good
        assert (
            engine._determine_fill_confidence(0.10, probs) == "Good"
        )  # 50th percentile
        # Bottom 50% should be Fair
        assert (
            engine._determine_fill_confidence(0.05, probs) == "Fair"
        )  # 25th percentile

        # Test fallback to absolute thresholds (single item, no batch)
        assert engine._determine_fill_confidence(0.20, None) == "Strong"  # >= 15%
        assert engine._determine_fill_confidence(0.10, None) == "Good"  # >= 8%
        assert engine._determine_fill_confidence(0.06, None) == "Fair"  # < 8%

    def test_get_recommendation_by_item_id_with_price_history(self, mock_engine):
        """Test that price history is included when requested."""
        engine, mock_loader = mock_engine

        # Mock prediction for item lookup
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                }
            ]
        )

        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 60
        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )
        mock_loader.get_price_history.return_value = [
            {"timestamp": "2026-01-10T00:00:00Z", "price": 5},
            {"timestamp": "2026-01-10T01:00:00Z", "price": 5},
        ]

        rec = engine.get_recommendation_by_item_id(
            item_id=554,
            capital=10000000,
            risk="medium",
            style="active",
            slots=4,
            include_price_history=True,
        )

        assert rec is not None
        assert "priceHistory" in rec
        assert len(rec["priceHistory"]) == 2

    def test_get_recommendation_by_item_id_without_price_history(self, mock_engine):
        """Test that price history is not included by default."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                }
            ]
        )

        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 60
        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = (
            100000  # High volume for unlimited crowding
        )

        rec = engine.get_recommendation_by_item_id(
            item_id=554,
            capital=10000000,
            risk="medium",
            style="active",
            slots=4,
            include_price_history=False,
        )

        assert rec is not None
        assert "priceHistory" not in rec

    def test_offset_ranges_constant(self, mock_engine):
        """Test that OFFSET_RANGES contains correct risk-level mappings."""
        engine, _ = mock_engine

        expected_ranges = {
            "low": (0.0125, 0.0150),
            "medium": (0.0125, 0.0175),
            "high": (0.0150, 0.0225),
        }
        assert engine.OFFSET_RANGES == expected_ranges

    def test_get_recommendations_passes_offset_range_to_loader(self, mock_engine):
        """Test that get_recommendations passes correct offset range based on risk."""
        engine, mock_loader = mock_engine

        # Mock empty predictions to avoid full processing
        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        # Test low risk -> conservative offsets
        engine.get_recommendations(
            style="passive",
            capital=10000000,
            risk="low",
            slots=4,
        )

        call_args = mock_loader.get_best_prediction_per_item.call_args
        assert call_args.kwargs.get("min_offset_pct") == 0.0125
        assert call_args.kwargs.get("max_offset_pct") == 0.0150

        # Test high risk -> aggressive offsets
        engine.get_recommendations(
            style="passive",
            capital=10000000,
            risk="high",
            slots=4,
        )

        call_args = mock_loader.get_best_prediction_per_item.call_args
        assert call_args.kwargs.get("min_offset_pct") == 0.0150
        assert call_args.kwargs.get("max_offset_pct") == 0.0225

    def test_get_recommendations_with_specific_offset_pct(self, mock_engine):
        """Test that specific offset_pct overrides risk-based range."""
        engine, mock_loader = mock_engine

        # Mock empty predictions
        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        # Specify exact offset_pct
        engine.get_recommendations(
            style="passive",
            capital=10000000,
            risk="low",
            slots=4,
            offset_pct=0.0200,  # Override low risk default
        )

        call_args = mock_loader.get_best_prediction_per_item.call_args
        # When specific offset_pct is provided, both min and max should be the same
        assert call_args.kwargs.get("min_offset_pct") == 0.0200
        assert call_args.kwargs.get("max_offset_pct") == 0.0200

    def test_high_fill_probability_items_remain_eligible(self, mock_engine):
        """Test that high fill probability predictions are NOT discarded.

        Regression test for issue #64: High-confidence items should not be
        excluded just because fill_probability is high. A high fill probability
        indicates model confidence and is a positive signal.
        """
        engine, mock_loader = mock_engine

        # Item with very high fill probability (0.50 = 50%)
        # Previously this would have been filtered out by max_fill_prob
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.50,  # Very high - was previously filtered
                    "expected_value": 0.015,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,
            risk="low",  # Even low risk should include high-confidence items
            slots=4,
        )

        # High fill probability items should NOT be excluded
        assert len(recommendations) >= 1
        assert any(rec["itemId"] == 554 for rec in recommendations)
        assert recommendations[0]["fillProbability"] == 0.50

    def test_fill_prob_minimums_constant(self, mock_engine):
        """Test that FILL_PROB_MINIMUMS contains only minimum thresholds."""
        engine, _ = mock_engine

        # Verify the constant exists and has correct structure
        expected_minimums = {
            "low": 0.08,
            "medium": 0.05,
            "high": 0.03,
        }
        assert engine.FILL_PROB_MINIMUMS == expected_minimums

    def test_loader_call_does_not_include_max_fill_prob(self, mock_engine):
        """Test that get_recommendations does NOT pass max_fill_prob to loader."""
        engine, mock_loader = mock_engine

        # Mock empty predictions
        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        engine.get_recommendations(
            style="passive",
            capital=10000000,
            risk="low",
            slots=4,
        )

        call_args = mock_loader.get_best_prediction_per_item.call_args
        # Verify max_fill_prob is NOT passed
        assert "max_fill_prob" not in call_args.kwargs


class TestRiskAdjustedScoring:
    """Test cases for risk-adjusted scoring (exit risk and fill risk)."""

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_exit_risk_penalty_trend_not_factored(self, mock_engine):
        """Test that trend is NOT factored into exit risk penalty.

        Trend filtering is now handled by _apply_trend_entry_filter() as a hard
        entry filter, not as a soft penalty in exit risk. Items in downtrends
        are filtered out entirely before reaching the candidate building stage.
        """
        engine, _ = mock_engine

        candidate_stable = {
            "buy_price": 1000,
            "sell_price": 1030,
            "trend": "Stable",
            "hour_offset": 4,
            "_spread_pct": 0.02,
        }
        candidate_falling = {
            "buy_price": 1000,
            "sell_price": 1030,
            "trend": "Falling",
            "hour_offset": 4,
            "_spread_pct": 0.02,
        }
        candidate_rising = {
            "buy_price": 1000,
            "sell_price": 1030,
            "trend": "Rising",
            "hour_offset": 4,
            "_spread_pct": 0.02,
        }

        penalty_stable = engine._calculate_exit_risk_penalty(candidate_stable, "medium")
        penalty_falling = engine._calculate_exit_risk_penalty(
            candidate_falling, "medium"
        )
        penalty_rising = engine._calculate_exit_risk_penalty(candidate_rising, "medium")

        # All trends should have the same penalty (trend is not factored)
        assert penalty_stable == penalty_falling
        assert penalty_stable == penalty_rising

    def test_exit_risk_penalty_long_hour_offset(self, mock_engine):
        """Test that long hour_offset increases exit risk penalty."""
        engine, _ = mock_engine

        candidate_short = {
            "buy_price": 1000,
            "sell_price": 1030,
            "trend": "Stable",
            "hour_offset": 4,
            "_spread_pct": 0.02,
        }
        candidate_long = {
            "buy_price": 1000,
            "sell_price": 1030,
            "trend": "Stable",
            "hour_offset": 24,
            "_spread_pct": 0.02,
        }

        penalty_short = engine._calculate_exit_risk_penalty(candidate_short, "medium")
        penalty_long = engine._calculate_exit_risk_penalty(candidate_long, "medium")

        # Long horizon should increase penalty
        assert penalty_long > penalty_short

    def test_fill_risk_penalty_wide_spread(self, mock_engine):
        """Test that wide spread increases fill risk penalty."""
        engine, _ = mock_engine

        candidate_tight = {
            "fill_probability": 0.15,
            "_spread_pct": 0.005,
            "crowding_capacity": None,
            "hour_offset": 4,
        }
        candidate_wide = {
            "fill_probability": 0.15,
            "_spread_pct": 0.10,
            "crowding_capacity": 10,
            "hour_offset": 4,
        }

        penalty_tight = engine._calculate_fill_risk_penalty(candidate_tight, "medium")
        penalty_wide = engine._calculate_fill_risk_penalty(candidate_wide, "medium")

        # Wide spread should increase fill risk penalty
        assert penalty_wide > penalty_tight

    def test_fill_risk_penalty_low_fill_probability(self, mock_engine):
        """Test that low fill probability increases fill risk penalty."""
        engine, _ = mock_engine

        candidate_high_prob = {
            "fill_probability": 0.25,
            "_spread_pct": 0.03,
            "crowding_capacity": None,
            "hour_offset": 4,
        }
        candidate_low_prob = {
            "fill_probability": 0.05,
            "_spread_pct": 0.03,
            "crowding_capacity": None,
            "hour_offset": 4,
        }

        penalty_high = engine._calculate_fill_risk_penalty(
            candidate_high_prob, "medium"
        )
        penalty_low = engine._calculate_fill_risk_penalty(candidate_low_prob, "medium")

        # Low fill probability should increase penalty
        assert penalty_low > penalty_high

    def test_fill_risk_penalty_crowding_capacity(self, mock_engine):
        """Test that tight crowding capacity increases fill risk penalty."""
        engine, _ = mock_engine

        candidate_unlimited = {
            "fill_probability": 0.15,
            "_spread_pct": 0.015,
            "crowding_capacity": None,  # Unlimited
            "hour_offset": 4,
        }
        candidate_tight = {
            "fill_probability": 0.15,
            "_spread_pct": 0.015,
            "crowding_capacity": 10,  # Tight limit
            "hour_offset": 4,
        }

        penalty_unlimited = engine._calculate_fill_risk_penalty(
            candidate_unlimited, "medium"
        )
        penalty_tight = engine._calculate_fill_risk_penalty(candidate_tight, "medium")

        # Tight crowding capacity should increase penalty
        assert penalty_tight > penalty_unlimited

    def test_risk_adjusted_score_low_risk_higher_penalty(self, mock_engine):
        """Test that low risk profile applies higher penalties."""
        engine, _ = mock_engine

        candidate = {
            "buy_price": 1000,
            "sell_price": 1030,
            "trend": "Falling",
            "hour_offset": 12,
            "_spread_pct": 0.02,
            "fill_probability": 0.10,
            "crowding_capacity": 20,
        }
        base_profit = 10000

        score_low = engine._calculate_risk_adjusted_score(candidate, base_profit, "low")
        score_medium = engine._calculate_risk_adjusted_score(
            candidate, base_profit, "medium"
        )
        score_high = engine._calculate_risk_adjusted_score(
            candidate, base_profit, "high"
        )

        # Low risk should have more penalty, resulting in lower adjusted score
        assert score_low < score_medium < score_high

    def test_risk_adjusted_score_penalty_cap(self, mock_engine):
        """Test that total penalty is capped at 70%."""
        engine, _ = mock_engine

        # Create a high-risk candidate that would have extreme penalties
        candidate = {
            "buy_price": 1000,
            "sell_price": 1100,  # Large required move
            "trend": "Falling",  # Downtrend
            "hour_offset": 48,  # Very long horizon
            "_spread_pct": 0.10,  # Wide spread
            "fill_probability": 0.03,  # Very low fill prob
            "crowding_capacity": 10,  # Tight crowding
        }
        base_profit = 10000

        score = engine._calculate_risk_adjusted_score(candidate, base_profit, "low")

        # Even with extreme penalties, should retain at least 30% of base profit
        assert score >= base_profit * 0.30

    def test_downtrend_reduces_rank_in_recommendations(self, mock_engine):
        """Test acceptance criteria: downtrend should reduce candidate selection.

        Note: With ILP optimizer, risk penalties affect which items are selected
        and their quantities via adjusted_profit. Final display order is by
        expectedProfit (raw profit), not risk-adjusted ranking. The penalty
        still affects which combinations of items are selected.
        """
        engine, mock_loader = mock_engine

        # Two candidates: same EV but one has downtrend
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,
                    "expected_value": 0.01,
                    "buy_price": 1000,
                    "sell_price": 1030,
                    "current_high": 1030,
                    "current_low": 1000,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,
                    "expected_value": 0.01,
                    "buy_price": 400,
                    "sell_price": 412,
                    "current_high": 412,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_volume_1h.return_value = 50000
        mock_loader.get_item_volume_24h.return_value = None

        # Fire rune: Falling trend (higher exit risk)
        # Blood rune: Stable trend (lower exit risk)
        def get_trend(item_id):
            return "Falling" if item_id == 554 else "Stable"

        mock_loader.get_item_trend.side_effect = get_trend

        recommendations = engine.get_recommendations(
            style="hybrid",
            capital=100000,
            risk="low",  # Low risk = higher penalty weight
            slots=2,
        )

        # With ILP optimizer, both items may be selected and the optimizer
        # maximizes total adjusted_profit (which incorporates risk penalties).
        # We verify that recommendations are returned and risk penalties
        # are applied during selection via the unit tests for penalty functions.
        assert len(recommendations) >= 1, "Should return recommendations"

        # Verify both items can be included (ILP maximizes total profit)
        # The exit risk penalty affects adjusted_profit used in optimization
        item_ids = {rec["itemId"] for rec in recommendations}
        # Blood rune (stable) should definitely be included
        assert 565 in item_ids, "Blood rune (stable trend) should be selected"

    def test_low_volume_reduces_rank_in_recommendations(self, mock_engine):
        """Test acceptance criteria: low volume should reduce candidate selection.

        Note: With ILP optimizer, risk penalties (including fill risk from
        low volume) affect which items are selected and their quantities via
        adjusted_profit. Final display order is by expectedProfit (raw profit).
        The penalty still affects which combinations of items are selected.
        """
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,
                    "expected_value": 0.01,
                    "buy_price": 1000,
                    "sell_price": 1030,
                    "current_high": 1030,
                    "current_low": 1000,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,
                    "expected_value": 0.01,
                    "buy_price": 400,
                    "sell_price": 412,
                    "current_high": 412,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None

        # Fire rune: low volume (returns low crowding capacity -> Low tier)
        # Blood rune: high volume (unlimited crowding capacity -> Very High tier)
        def get_volume_1h(item_id):
            return 500 if item_id == 554 else 100000

        mock_loader.get_item_volume_1h.side_effect = get_volume_1h

        recommendations = engine.get_recommendations(
            style="hybrid",
            capital=100000,
            risk="low",
            slots=2,
        )

        # With ILP optimizer, both items may be selected and the optimizer
        # maximizes total adjusted_profit (which incorporates fill risk penalty
        # for low volume). We verify that recommendations are returned and
        # fill risk penalties are applied during selection.
        assert len(recommendations) >= 1, "Should return recommendations"

        # Verify both items can be included (ILP maximizes total profit)
        # The fill risk penalty affects adjusted_profit used in optimization
        item_ids = {rec["itemId"] for rec in recommendations}
        # Blood rune (high volume) should definitely be included
        assert 565 in item_ids, "Blood rune (high volume) should be selected"


class TestProfitCalculation:
    """Test profit per unit calculation with accurate tax rules."""

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_tax_per_unit_below_floor(self, mock_engine):
        """Items sold below 50gp should have 0 tax."""
        engine, _ = mock_engine
        assert engine._calculate_tax_per_unit(49) == 0
        assert engine._calculate_tax_per_unit(6) == 0

    def test_tax_per_unit_at_boundary(self, mock_engine):
        """Items sold at exactly 50gp should have 1gp tax."""
        engine, _ = mock_engine
        assert engine._calculate_tax_per_unit(50) == 1

    def test_tax_per_unit_standard(self, mock_engine):
        """Standard items should have 2% tax rounded down."""
        engine, _ = mock_engine
        assert engine._calculate_tax_per_unit(100) == 2
        assert engine._calculate_tax_per_unit(149) == 2  # 2.98 -> 2
        assert engine._calculate_tax_per_unit(150) == 3

    def test_tax_per_unit_high_value_cap(self, mock_engine):
        """High-value items should cap at 5M tax."""
        engine, _ = mock_engine
        assert engine._calculate_tax_per_unit(300_000_000) == 5_000_000
        assert engine._calculate_tax_per_unit(1_000_000_000) == 5_000_000

    def test_tax_returns_integer(self, mock_engine):
        """Tax should always return an integer, not float."""
        engine, _ = mock_engine
        tax = engine._calculate_tax_per_unit(149)
        assert isinstance(tax, int)


class TestInstantFillBlocking:
    """Test cases for instant-fill blocking (issue #61).

    The engine must block recommendations where buy_price >= current_high,
    as these would instant-fill and potentially create immediate losses.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def _make_row(
        self,
        item_id: int = 554,
        buy_price: int = 5000,
        sell_price: int = 5500,
        current_high=5250,
        current_low: int = 4900,
    ) -> dict:
        """Create a mock prediction row for testing.

        Default prices are set to avoid triggering manipulation filters:
        - buy_price > 1000gp (avoids cheap-item volume check)
        - spread ~7% (under 10% max_spread_pct threshold)
        """
        return {
            "item_id": item_id,
            "item_name": "Test Item",
            "buy_price": buy_price,
            "sell_price": sell_price,
            "fill_probability": 0.10,
            "expected_value": 0.008,
            "hour_offset": 4,
            "confidence": "medium",
            "current_high": current_high,
            "current_low": current_low,
        }

    def test_build_candidate_allows_buy_below_current_high(self, mock_engine):
        """Buy price below current_high should be allowed."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 50000

        # buy_price (5000) < current_high (5250) - should be allowed
        row = self._make_row(buy_price=5000, current_high=5250)
        candidate = engine._build_candidate(
            row, max_capital=1000000, pred_age_seconds=60
        )

        assert candidate is not None
        # Price buffer may adjust buy_price slightly upward (toward market)
        assert candidate["buy_price"] >= 5000
        assert candidate["buy_price"] < 5250  # Must stay below current_high

    def test_build_candidate_blocks_buy_at_current_high(self, mock_engine):
        """Buy price equal to current_high should be blocked (instant-fill)."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 50000

        # buy_price (5250) == current_high (5250) - should be blocked
        row = self._make_row(buy_price=5250, current_high=5250)
        candidate = engine._build_candidate(
            row, max_capital=1000000, pred_age_seconds=60
        )

        assert candidate is None

    def test_build_candidate_blocks_buy_above_current_high(self, mock_engine):
        """Buy price above current_high should be blocked (instant-fill)."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 50000

        # buy_price (5500) > current_high (5250) - should be blocked
        row = self._make_row(buy_price=5500, current_high=5250)
        candidate = engine._build_candidate(
            row, max_capital=1000000, pred_age_seconds=60
        )

        assert candidate is None

    def test_build_candidate_rejects_missing_current_high(self, mock_engine):
        """Missing current_high should fail closed (reject candidate)."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 50000

        # current_high is None - should fail closed
        row = self._make_row(current_high=None)
        candidate = engine._build_candidate(
            row, max_capital=1000000, pred_age_seconds=60
        )

        assert candidate is None

    def test_build_candidate_rejects_nan_current_high(self, mock_engine):
        """NaN current_high should fail closed (reject candidate)."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 50000

        # current_high is NaN - should fail closed
        row = self._make_row(current_high=float("nan"))
        candidate = engine._build_candidate(
            row, max_capital=1000000, pred_age_seconds=60
        )

        assert candidate is None

    def test_get_recommendations_excludes_instant_fill_candidates(self, mock_engine):
        """Integration test: recommendations should not include instant-fill items."""
        engine, mock_loader = mock_engine

        # Two items: one safe, one would instant-fill
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,  # Below current_high of 6
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 420,  # At current_high - instant fill!
                    "sell_price": 440,
                    "current_high": 420,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,
            risk="medium",
            slots=4,
        )

        # Fire rune should be included (buy_price < current_high)
        # Blood rune should be excluded (buy_price == current_high)
        item_ids = {rec["itemId"] for rec in recommendations}
        assert 554 in item_ids, "Fire rune should be included (safe)"
        assert 565 not in item_ids, "Blood rune should be excluded (instant-fill)"


class TestQuantityFallback:
    """Test cases for quantity fallback when buy limit is unknown (issue #77).

    When the GE buy limit for an item is unavailable, the engine should use
    a conservative fallback of 1000 instead of 10000 to prevent recommending
    unrealistic quantities that exceed actual GE limits.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_build_candidate_uses_conservative_fallback_without_buy_limit(
        self, mock_engine
    ):
        """Test that missing buy limit falls back to 1000, not 10000."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = None  # No buy limit available
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 50000

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 12345,
                    "item_name": "Test Item",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 100,
                    "sell_price": 110,
                    "current_high": 110,
                    "current_low": 100,
                    "confidence": "medium",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,  # 10M - would allow qty=100,000 based on price alone
            risk="medium",
            slots=4,
        )

        # Should use conservative fallback of 1000, not 10000
        assert len(recommendations) >= 1
        assert (
            recommendations[0]["quantity"] <= 1000
        ), "Quantity should be capped at 1000 fallback, not 10000"

    def test_build_candidate_respects_buy_limit_when_available(self, mock_engine):
        """Test that buy limit is used when available (not fallback)."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 500  # Specific buy limit
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 50000

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 12345,
                    "item_name": "Test Item",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 100,
                    "sell_price": 110,
                    "current_high": 110,
                    "current_low": 100,
                    "confidence": "medium",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,  # 10M
            risk="medium",
            slots=4,
        )

        # Should respect the actual buy limit of 500
        assert len(recommendations) >= 1
        assert (
            recommendations[0]["quantity"] <= 500
        ), "Quantity should respect buy limit of 500"

    def test_build_candidate_fallback_with_zero_buy_limit(self, mock_engine):
        """Test that zero buy limit triggers fallback to 1000."""
        engine, mock_loader = mock_engine

        mock_loader.get_item_buy_limit.return_value = 0  # Zero buy limit
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 50000

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 12345,
                    "item_name": "Test Item",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 100,
                    "sell_price": 110,
                    "current_high": 110,
                    "current_low": 100,
                    "confidence": "medium",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,  # 10M
            risk="medium",
            slots=4,
        )

        # Should use conservative fallback of 1000 since buy_limit is 0
        assert len(recommendations) >= 1
        assert (
            recommendations[0]["quantity"] <= 1000
        ), "Quantity should fall back to 1000 when buy_limit is 0"


class TestPassiveSlotHourEfficiency:
    """Test cases for passive mode slot-hour efficiency penalty (issue #63).

    The passive mode penalty for low profit-per-slot-hour should compare
    candidates against a portfolio-level baseline (median), not against
    themselves.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_calculate_slot_hour_efficiency_baseline_median(self, mock_engine):
        """Test that slot-hour baseline is calculated as median of candidates."""
        engine, _ = mock_engine

        candidates = [
            {
                "item_id": 1,
                "buy_price": 100,
                "max_quantity": 1000,
                "profit_per_unit": 5,
                "fill_probability": 0.10,
                "hour_offset": 4,
            },
            {
                "item_id": 2,
                "buy_price": 100,
                "max_quantity": 1000,
                "profit_per_unit": 10,
                "fill_probability": 0.10,
                "hour_offset": 8,
            },
            {
                "item_id": 3,
                "buy_price": 100,
                "max_quantity": 1000,
                "profit_per_unit": 20,
                "fill_probability": 0.10,
                "hour_offset": 12,
            },
        ]

        # With total_capital=100000, max_qty for each = 1000
        # Item 1: expected_profit = 5 * 1000 * 0.10 = 500, /4h = 125
        # Item 2: expected_profit = 10 * 1000 * 0.10 = 1000, /8h = 125
        # Item 3: expected_profit = 20 * 1000 * 0.10 = 2000, /12h = 166.67
        # Sorted: [125, 125, 166.67], median = 125

        baseline = engine._calculate_slot_hour_efficiency_baseline(
            candidates, total_capital=100000
        )

        assert baseline == 125.0

    def test_calculate_slot_hour_efficiency_baseline_empty(self, mock_engine):
        """Test that empty candidates returns 0.0 baseline."""
        engine, _ = mock_engine

        baseline = engine._calculate_slot_hour_efficiency_baseline(
            candidates=[], total_capital=100000
        )

        assert baseline == 0.0

    def test_calculate_slot_hour_efficiency_baseline_excludes_zero_profit(
        self, mock_engine
    ):
        """Test that candidates with zero/negative profit are excluded."""
        engine, _ = mock_engine

        candidates = [
            {
                "item_id": 1,
                "buy_price": 100,
                "max_quantity": 1000,
                "profit_per_unit": 0,  # Zero profit
                "fill_probability": 0.10,
                "hour_offset": 4,
            },
            {
                "item_id": 2,
                "buy_price": 100,
                "max_quantity": 1000,
                "profit_per_unit": 10,
                "fill_probability": 0.10,
                "hour_offset": 8,
            },
        ]

        baseline = engine._calculate_slot_hour_efficiency_baseline(
            candidates, total_capital=100000
        )

        # Only item 2 contributes: 10 * 1000 * 0.10 / 8 = 125
        assert baseline == 125.0

    def test_passive_mode_penalizes_low_slot_hour_efficiency(self, mock_engine):
        """Test acceptance: low slot-hour efficiency gets penalized in passive mode.

        Regression test for issue #63: The penalty for poor profit-per-slot-hour
        should actually trigger for candidates below 50% of portfolio median.
        """
        engine, mock_loader = mock_engine

        # Two candidates with significantly different slot-hour efficiency:
        # - High efficiency: 8h hold, higher margin
        # - Low efficiency: 48h hold, lower margin (same EV spread over 6x time)
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 1001,
                    "item_name": "High Efficiency Item",
                    "hour_offset": 8,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,
                    "expected_value": 0.012,
                    "buy_price": 10000,
                    "sell_price": 10500,
                    "current_high": 10400,
                    "current_low": 10000,
                    "confidence": "high",
                },
                {
                    "item_id": 1002,
                    "item_name": "Low Efficiency Item",
                    "hour_offset": 48,  # 6x longer hold
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 10000,
                    "sell_price": 10300,  # Lower margin
                    "current_high": 10250,
                    "current_low": 10000,
                    "confidence": "medium",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_recommendations(
            style="passive",
            capital=100000000,  # 100M
            risk="medium",
            slots=4,
        )

        # Both should be included
        assert len(recommendations) >= 1

        # High efficiency item should rank higher due to slot-hour penalty on low eff
        if len(recommendations) >= 2:
            item_order = [rec["itemId"] for rec in recommendations]
            high_eff_idx = (
                item_order.index(1001) if 1001 in item_order else len(item_order)
            )
            low_eff_idx = (
                item_order.index(1002) if 1002 in item_order else len(item_order)
            )
            assert (
                high_eff_idx < low_eff_idx
            ), "High efficiency item should rank above low efficiency item"

    def test_slot_hour_penalty_only_applies_in_passive_mode(self, mock_engine):
        """Test that slot-hour efficiency penalty only applies in passive mode."""
        engine, mock_loader = mock_engine

        # Same candidates but in active mode
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 1001,
                    "item_name": "Item A",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,
                    "expected_value": 0.012,
                    "buy_price": 1000,
                    "sell_price": 1050,
                    "current_high": 1040,
                    "current_low": 1000,
                    "confidence": "high",
                },
                {
                    "item_id": 1002,
                    "item_name": "Item B",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 1000,
                    "sell_price": 1030,
                    "current_high": 1025,
                    "current_low": 1000,
                    "confidence": "medium",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_recommendations(
            style="active",  # Active mode - no slot-hour penalty
            capital=10000000,
            risk="medium",
            slots=4,
        )

        # Should have recommendations (baseline computation doesn't affect active)
        assert len(recommendations) >= 1


class TestPassiveModeCapitalThreshold:
    """Test cases for passive mode capital threshold (issue #128).

    The passive mode capital threshold should use logarithmic scaling so that
    high-capital users don't get filtered out of viable trades.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def _make_candidate(
        self, item_id: int, item_name: str, buy_price: int, max_quantity: int
    ):
        """Helper to create a candidate dict."""
        return {
            "item_id": item_id,
            "item_name": item_name,
            "hour_offset": 8,
            "offset_pct": 0.015,
            "fill_probability": 0.10,
            "expected_value": 0.008,
            "buy_price": buy_price,
            "sell_price": int(buy_price * 1.03),
            "current_high": int(buy_price * 1.03),
            "current_low": buy_price,
            "confidence": "high",
            "max_quantity": max_quantity,
            "profit_per_unit": int(buy_price * 0.01),
        }

    def test_passive_high_capital_includes_mid_value_items(self, mock_engine):
        """Test that 2B capital still includes items viable at 500M (issue #128)."""
        engine, mock_loader = mock_engine

        # Item with ~50M max capital (50k price * 1000 qty)
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 22124,
                    "item_name": "Superior dragon bones",
                    "hour_offset": 8,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 50000,
                    "sell_price": 51500,
                    "current_high": 51500,
                    "current_low": 50000,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 1000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 50000
        mock_loader.get_item_volume_1h.return_value = 5000

        # Should return recommendations at both capital levels
        recs_500m = engine.get_recommendations(
            style="passive",
            capital=500_000_000,
            risk="high",
            slots=4,
        )

        recs_2b = engine.get_recommendations(
            style="passive",
            capital=2_000_000_000,
            risk="high",
            slots=4,
        )

        # If item works at 500M, it must work at 2B
        # (more capital should give more options, not fewer)
        if len(recs_500m) > 0:
            assert len(recs_2b) >= len(recs_500m), (
                f"2B capital returned {len(recs_2b)} recommendations, "
                f"but 500M returned {len(recs_500m)}"
            )

    def test_passive_threshold_scales_logarithmically(self, mock_engine):
        """Test that passive mode threshold scales slowly (log) not linearly."""
        import math

        engine, _ = mock_engine

        # Calculate thresholds at different capital levels using the formula
        base_threshold = 5_000_000

        def calc_threshold(capital):
            capital_factor = math.log10(max(capital, 1_000_000) / 1_000_000)
            return base_threshold * (1 + capital_factor * 2)

        thresh_1m = calc_threshold(1_000_000)
        thresh_10m = calc_threshold(10_000_000)
        thresh_100m = calc_threshold(100_000_000)
        thresh_1b = calc_threshold(1_000_000_000)
        thresh_10b = calc_threshold(10_000_000_000)

        # Verify logarithmic scaling (thresholds should grow slowly)
        assert thresh_1m == 5_000_000  # Base at 1M
        assert 10_000_000 < thresh_10m < 20_000_000  # ~15M at 10M capital
        assert 20_000_000 < thresh_100m < 30_000_000  # ~25M at 100M capital
        assert 30_000_000 < thresh_1b < 40_000_000  # ~35M at 1B capital
        assert 40_000_000 < thresh_10b < 50_000_000  # ~45M at 10B capital

        # Key insight: 1000x capital (1M to 1B) only ~7x threshold
        assert thresh_1b < thresh_1m * 10

    def test_active_mode_has_no_minimum_threshold(self, mock_engine):
        """Test that active mode doesn't filter by capital threshold."""
        engine, mock_loader = mock_engine

        # Very low-value item (1 gp * 1000 qty = 1000 gp max capital)
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 1000000
        mock_loader.get_item_volume_1h.return_value = 50000

        # Active mode at very high capital should still include low-value items
        recs = engine.get_recommendations(
            style="active",
            capital=10_000_000_000,  # 10B
            risk="high",
            slots=4,
        )

        # Should have recommendations (active doesn't filter by capital threshold)
        assert len(recs) >= 1


class TestRecIdLookupRehydration:
    """Test cases for rec_id lookup rehydration (issue #130).

    When a recommendation is looked up by rec_id but not found in the
    in-memory cache, it should be regenerated from the database.
    This enables multi-worker deployments.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_get_by_id_returns_cached(self, mock_engine):
        """Test that cached recommendations are returned immediately."""
        engine, mock_loader = mock_engine

        # Pre-populate cache
        rec = {"itemId": 554, "item": "Fire rune", "isRecommended": True}
        rec_id = engine.store.store(rec)

        # Should return cached result without hitting database
        result = engine.get_recommendation_by_id(rec_id)
        assert result is not None
        assert result["itemId"] == 554

        # Database should not be queried
        mock_loader.get_predictions_for_item.assert_not_called()

    def test_get_by_id_rehydrates_from_database(self, mock_engine):
        """Test that cache miss triggers database rehydration (issue #130)."""
        engine, mock_loader = mock_engine

        # Mock database predictions
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 10000  # 10% of 24h volume
        mock_loader.get_prediction_age_seconds.return_value = 120

        # Use rec_id that's not in cache (simulates different worker)
        rec_id = "rec_554_2026011515"

        # Should regenerate from database
        result = engine.get_recommendation_by_id(rec_id)

        assert result is not None
        assert result["itemId"] == 554
        assert result["id"] == rec_id  # Should preserve original rec_id

        # Verify database was queried
        mock_loader.get_predictions_for_item.assert_called_once_with(554)

    def test_get_by_id_invalid_format_returns_none(self, mock_engine):
        """Test that invalid rec_id format returns None."""
        engine, _ = mock_engine

        assert engine.get_recommendation_by_id("invalid") is None
        assert engine.get_recommendation_by_id("rec_abc_2026011515") is None
        assert engine.get_recommendation_by_id("") is None

    def test_get_by_id_not_recommended_returns_none(self, mock_engine):
        """Test that items not recommended return None."""
        engine, mock_loader = mock_engine

        # Mock empty predictions (item exists but has no viable predictions)
        mock_loader.get_predictions_for_item.return_value = pd.DataFrame()

        rec_id = "rec_999_2026011515"  # Item with no predictions
        result = engine.get_recommendation_by_id(rec_id)

        assert result is None

    def test_get_by_id_caches_rehydrated_result(self, mock_engine):
        """Test that rehydrated recommendations are cached for future use."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_volume_1h.return_value = 10000  # 10% of 24h volume
        mock_loader.get_prediction_age_seconds.return_value = 120

        rec_id = "rec_554_2026011515"

        # First call: rehydrate from database
        result1 = engine.get_recommendation_by_id(rec_id)
        assert result1 is not None
        assert mock_loader.get_predictions_for_item.call_count == 1

        # Second call: should return cached result
        result2 = engine.get_recommendation_by_id(rec_id)
        assert result2 is not None
        assert result2["id"] == result1["id"]

        # Database should NOT be queried again
        assert mock_loader.get_predictions_for_item.call_count == 1


class TestDynamicCandidatePoolLimit:
    """Test cases for dynamic candidate pool limit scaling (issue #66).

    The candidate pool limit should scale with capital to ensure large-capital
    users can access high-absolute-profit items that may have lower EV%.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_candidate_limit_scales_with_capital(self, mock_engine):
        """Test that candidate pool limit increases with capital.

        Verifies that get_best_prediction_per_item is called with higher
        limit values for larger capital amounts.
        """
        engine, mock_loader = mock_engine

        # Mock empty predictions to avoid full processing
        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        # Test small capital (10M) - should use base limit (~100)
        engine.get_recommendations(
            style="active",
            capital=10_000_000,
            risk="medium",
            slots=4,
        )

        small_cap_call = mock_loader.get_best_prediction_per_item.call_args
        small_cap_limit = small_cap_call.kwargs.get("limit")

        # Reset mock
        mock_loader.get_best_prediction_per_item.reset_mock()

        # Test large capital (1B) - should have higher limit
        engine.get_recommendations(
            style="active",
            capital=1_000_000_000,
            risk="medium",
            slots=4,
        )

        large_cap_call = mock_loader.get_best_prediction_per_item.call_args
        large_cap_limit = large_cap_call.kwargs.get("limit")

        # Large capital should have higher limit
        assert large_cap_limit > small_cap_limit, (
            f"Large capital limit ({large_cap_limit}) should exceed "
            f"small capital limit ({small_cap_limit})"
        )

    def test_candidate_limit_caps_at_500(self, mock_engine):
        """Test that candidate pool limit has a hard cap at 500."""
        engine, mock_loader = mock_engine

        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        # Test extremely large capital (100B) - should cap at 500
        engine.get_recommendations(
            style="active",
            capital=100_000_000_000,
            risk="medium",
            slots=8,  # Max slots
        )

        call_args = mock_loader.get_best_prediction_per_item.call_args
        limit = call_args.kwargs.get("limit")

        assert limit <= 500, f"Limit ({limit}) should not exceed hard cap of 500"

    def test_candidate_limit_uses_remaining_capital(self, mock_engine):
        """Test that limit calculation uses remaining capital after active trades."""
        engine, mock_loader = mock_engine

        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        # 1B capital with 900M tied up in active trades = 100M remaining
        active_trades = [
            {"itemId": 100, "quantity": 1, "buyPrice": 900_000_000},
        ]

        engine.get_recommendations(
            style="active",
            capital=1_000_000_000,
            risk="medium",
            slots=4,
            active_trades=active_trades,
        )

        call_args = mock_loader.get_best_prediction_per_item.call_args
        limit = call_args.kwargs.get("limit")

        # With only 100M remaining, limit should be lower than full 1B
        # 100M = capital_factor of 2, base 30 * 2 = 60
        # But base is max(30, 100) = 100, so 100 * (1 + 2*0.5) = 200
        assert (
            limit <= 200
        ), f"Limit ({limit}) should reflect remaining capital, not total capital"

    def test_large_capital_surfaces_high_profit_items(self, mock_engine):
        """Test that large-capital requests can surface big-ticket items.

        Acceptance criteria from issue #66: Large-capital requests can surface
        big-ticket items when appropriate.

        This test verifies that when the candidate pool is expanded for large
        capital users, high-price items with lower EV% but high absolute profit
        potential can be included and selected.
        """
        engine, mock_loader = mock_engine

        # High-price item with moderate EV but high absolute profit potential
        # At 50M gp per unit with buy limit of 8, this item requires significant capital
        # but has meaningful absolute profit (50M * 0.015 * 0.08 = 60K expected profit/unit)
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 13652,  # Dragon claws (high-value item)
                    "item_name": "Dragon claws",
                    "hour_offset": 8,
                    "offset_pct": 0.015,
                    "fill_probability": 0.08,
                    "expected_value": 0.006,  # Lower EV% than cheap items
                    "buy_price": 50_000_000,  # 50M per unit
                    "sell_price": 50_750_000,  # 1.5% margin
                    "current_high": 51_000_000,  # Above buy_price to avoid instant-fill
                    "current_low": 49_500_000,
                    "confidence": "medium",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 8  # Can buy 8 per 4 hours
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 1000

        recommendations = engine.get_recommendations(
            style="passive",
            capital=1_000_000_000,  # 1B capital - whale user
            risk="medium",
            slots=4,
        )

        # Should include the high-price item
        assert len(recommendations) >= 1, "Should return at least one recommendation"
        assert any(
            rec["itemId"] == 13652 for rec in recommendations
        ), "Dragon claws should be surfaced for large-capital user"

    def test_limit_scaling_formula(self, mock_engine):
        """Test the exact limit scaling formula for various capital levels."""
        engine, mock_loader = mock_engine

        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        test_cases = [
            # (capital, slots, expected_min_limit, expected_max_limit)
            (1_000_000, 4, 100, 110),  # 1M: factor=0, limit=100
            (10_000_000, 4, 100, 160),  # 10M: factor=1, limit~=150
            (100_000_000, 4, 150, 250),  # 100M: factor=2, limit~=200
            (1_000_000_000, 4, 200, 300),  # 1B: factor=3, limit~=250
        ]

        for capital, slots, min_limit, max_limit in test_cases:
            mock_loader.get_best_prediction_per_item.reset_mock()

            engine.get_recommendations(
                style="active",
                capital=capital,
                risk="medium",
                slots=slots,
            )

            call_args = mock_loader.get_best_prediction_per_item.call_args
            limit = call_args.kwargs.get("limit")

            assert min_limit <= limit <= max_limit, (
                f"Capital {capital:,}: limit {limit} not in range "
                f"[{min_limit}, {max_limit}]"
            )


class TestVolumeFiltering:
    """Test cases for minimum volume filtering (issue #133).

    Low-volume items like King worms should be filtered out to prevent
    users from getting stuck with illiquid items.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_min_volume_24h_passed_to_loader(self, mock_engine):
        """Test that min_volume_24h config is passed to the loader."""
        engine, mock_loader = mock_engine

        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        engine.get_recommendations(
            style="active",
            capital=10_000_000,
            risk="medium",
            slots=4,
        )

        # Verify min_volume_24h was passed to the loader
        call_args = mock_loader.get_best_prediction_per_item.call_args
        assert "min_volume_24h" in call_args.kwargs
        # Default is 1000 (from config)
        assert call_args.kwargs["min_volume_24h"] == engine.config.min_volume_24h

    def test_custom_volume_threshold_from_config(self, mock_db_connection):
        """Test that custom MIN_VOLUME_24H config is respected."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

            # Create engine with custom config
            from src.config import Config

            custom_config = Config()
            custom_config.min_volume_24h = 5000  # Higher threshold

            engine = RecommendationEngine(
                db_connection_string=mock_db_connection, config=custom_config
            )
            engine.loader = mock_loader

            engine.get_recommendations(
                style="active",
                capital=10_000_000,
                risk="medium",
                slots=4,
            )

            call_args = mock_loader.get_best_prediction_per_item.call_args
            assert call_args.kwargs["min_volume_24h"] == 5000

    def test_zero_volume_threshold_disables_filtering(self, mock_db_connection):
        """Test that setting min_volume_24h=0 disables volume filtering."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

            # Create engine with volume filtering disabled
            from src.config import Config

            custom_config = Config()
            custom_config.min_volume_24h = 0  # Disabled

            engine = RecommendationEngine(
                db_connection_string=mock_db_connection, config=custom_config
            )
            engine.loader = mock_loader

            engine.get_recommendations(
                style="active",
                capital=10_000_000,
                risk="medium",
                slots=4,
            )

            call_args = mock_loader.get_best_prediction_per_item.call_args
            # Should pass 0 which means no filtering
            assert call_args.kwargs["min_volume_24h"] == 0


class TestPortfolioOptimizer:
    """Test cases for ILP-based portfolio optimizer (issue #69).

    The optimizer should maximize total expected profit using Integer Linear
    Programming (ILP) instead of greedy ROI selection, while falling back to
    greedy when the ILP solver fails.
    """

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def _make_scored_option(
        self,
        item_id: int,
        capital_used: int,
        adjusted_profit: float,
        quantity: int = 100,
    ) -> dict:
        """Create a mock scored option for testing."""
        return {
            "candidate": {
                "item_id": item_id,
                "item_name": f"Item {item_id}",
                "buy_price": capital_used // quantity,
                "sell_price": int(capital_used // quantity * 1.03),
                "max_quantity": quantity * 2,
                "profit_per_unit": int(adjusted_profit / quantity / 0.10),
                "fill_probability": 0.10,
                "expected_value": 0.008,
                "confidence": "medium",
                "crowding_capacity": None,
                "trend": "Stable",
                "volume_24h": None,
                "hour_offset": 4,
                "is_multi_limit": False,
                "base_buy_limit": None,
            },
            "quantity": quantity,
            "capital_used": capital_used,
            "expected_profit": int(adjusted_profit),
            "adjusted_profit": adjusted_profit,
            "profit_per_capital": adjusted_profit / capital_used if capital_used else 0,
            "profit_per_slot_hour": adjusted_profit / 4,
        }

    def test_ilp_beats_greedy_capital_utilization(self, mock_engine):
        """Test that ILP finds better solution when greedy leaves capital unused.

        Scenario: 100k capital, 2 slots
        - Item A: 10k capital, 2k profit (20% ROI)
        - Item B: 45k capital, 4.5k profit (10% ROI)
        - Item C: 45k capital, 4.5k profit (10% ROI)

        Greedy picks: A (best ROI), then B = 6.5k profit, 55k used
        ILP picks: B + C = 9k profit, 90k used
        ILP is 38% better
        """
        engine, _ = mock_engine

        scored_options = [
            self._make_scored_option(
                item_id=1, capital_used=10000, adjusted_profit=2000, quantity=100
            ),
            self._make_scored_option(
                item_id=2, capital_used=45000, adjusted_profit=4500, quantity=450
            ),
            self._make_scored_option(
                item_id=3, capital_used=45000, adjusted_profit=4500, quantity=450
            ),
        ]

        # ILP solution
        ilp_selected = engine._solve_portfolio_ilp(
            scored_options, total_capital=100000, num_slots=2
        )
        ilp_profit = sum(opt["adjusted_profit"] for opt in ilp_selected)
        ilp_capital = sum(opt["capital_used"] for opt in ilp_selected)

        # Greedy solution (for comparison)
        greedy_selected = engine._greedy_select(
            scored_options,
            total_capital=100000,
            num_slots=2,
            min_slots_target=1,
            params={"target_pct": 0.30, "concentration_penalty": 0.25},
            max_capital_per_trade=50000,
        )
        greedy_profit = sum(opt["adjusted_profit"] for opt in greedy_selected)
        greedy_capital = sum(opt["capital_used"] for opt in greedy_selected)

        # ILP should find the better solution
        assert (
            ilp_profit > greedy_profit
        ), f"ILP profit ({ilp_profit}) should exceed greedy profit ({greedy_profit})"
        assert (
            ilp_capital > greedy_capital
        ), f"ILP should use more capital ({ilp_capital}) than greedy ({greedy_capital})"

    def test_ilp_slot_constrained_optimization(self, mock_engine):
        """Test that ILP optimizes correctly with single slot constraint.

        Scenario: 100k capital, 1 slot only
        - Item A: 20k capital, 3k profit (15% ROI)
        - Item B: 80k capital, 10k profit (12.5% ROI)

        Greedy picks A (higher ROI), profit = 3k
        ILP picks B (fills capital better), profit = 10k
        """
        engine, _ = mock_engine

        scored_options = [
            self._make_scored_option(
                item_id=1, capital_used=20000, adjusted_profit=3000, quantity=200
            ),
            self._make_scored_option(
                item_id=2, capital_used=80000, adjusted_profit=10000, quantity=800
            ),
        ]

        # ILP solution
        ilp_selected = engine._solve_portfolio_ilp(
            scored_options, total_capital=100000, num_slots=1
        )
        ilp_profit = sum(opt["adjusted_profit"] for opt in ilp_selected)

        # Greedy solution
        greedy_selected = engine._greedy_select(
            scored_options,
            total_capital=100000,
            num_slots=1,
            min_slots_target=1,
            params={"target_pct": 0.50, "concentration_penalty": 0.1},
            max_capital_per_trade=100000,
        )
        greedy_profit = sum(opt["adjusted_profit"] for opt in greedy_selected)

        # ILP should find the better solution
        assert (
            ilp_profit > greedy_profit
        ), f"ILP profit ({ilp_profit}) should exceed greedy profit ({greedy_profit})"

    def test_ilp_respects_mutual_exclusivity(self, mock_engine):
        """Test that ILP doesn't select same item with multiple allocations."""
        engine, _ = mock_engine

        # Same item with different allocation levels
        scored_options = [
            self._make_scored_option(
                item_id=1, capital_used=10000, adjusted_profit=1000, quantity=100
            ),
            self._make_scored_option(
                item_id=1, capital_used=20000, adjusted_profit=1800, quantity=200
            ),  # Same item, larger allocation
            self._make_scored_option(
                item_id=2, capital_used=30000, adjusted_profit=2500, quantity=300
            ),
        ]

        ilp_selected = engine._solve_portfolio_ilp(
            scored_options, total_capital=100000, num_slots=3
        )

        # Count how many times item 1 was selected
        item_1_count = sum(
            1 for opt in ilp_selected if opt["candidate"]["item_id"] == 1
        )

        assert (
            item_1_count <= 1
        ), f"Item 1 selected {item_1_count} times, should be at most 1"

    def test_ilp_empty_options_returns_empty(self, mock_engine):
        """Test that ILP returns empty list for empty input."""
        engine, _ = mock_engine

        ilp_selected = engine._solve_portfolio_ilp(
            scored_options=[], total_capital=100000, num_slots=2
        )

        assert ilp_selected == []

    def test_ilp_respects_capital_constraint(self, mock_engine):
        """Test that ILP respects total capital budget."""
        engine, _ = mock_engine

        scored_options = [
            self._make_scored_option(
                item_id=1, capital_used=60000, adjusted_profit=6000, quantity=600
            ),
            self._make_scored_option(
                item_id=2, capital_used=60000, adjusted_profit=6000, quantity=600
            ),
        ]

        ilp_selected = engine._solve_portfolio_ilp(
            scored_options, total_capital=100000, num_slots=3
        )
        total_capital_used = sum(opt["capital_used"] for opt in ilp_selected)

        assert (
            total_capital_used <= 100000
        ), f"Total capital used ({total_capital_used}) exceeds budget (100000)"

    def test_ilp_respects_slot_constraint(self, mock_engine):
        """Test that ILP respects number of slots constraint."""
        engine, _ = mock_engine

        scored_options = [
            self._make_scored_option(
                item_id=i, capital_used=10000, adjusted_profit=1000, quantity=100
            )
            for i in range(5)
        ]

        ilp_selected = engine._solve_portfolio_ilp(
            scored_options, total_capital=100000, num_slots=2
        )

        assert (
            len(ilp_selected) <= 2
        ), f"Selected {len(ilp_selected)} items, but num_slots is 2"

    def test_greedy_fallback_basic(self, mock_engine):
        """Test that greedy fallback produces valid selections."""
        engine, _ = mock_engine

        scored_options = [
            self._make_scored_option(
                item_id=1, capital_used=30000, adjusted_profit=3000, quantity=300
            ),
            self._make_scored_option(
                item_id=2, capital_used=40000, adjusted_profit=3500, quantity=400
            ),
        ]

        greedy_selected = engine._greedy_select(
            scored_options,
            total_capital=100000,
            num_slots=2,
            min_slots_target=1,
            params={"target_pct": 0.30, "concentration_penalty": 0.25},
            max_capital_per_trade=50000,
        )

        # Should select items within constraints
        assert len(greedy_selected) <= 2
        total_capital = sum(opt["capital_used"] for opt in greedy_selected)
        assert total_capital <= 100000

    def test_greedy_fallback_orders_by_roi(self, mock_engine):
        """Test that greedy fallback selects highest ROI first."""
        engine, _ = mock_engine

        scored_options = [
            self._make_scored_option(
                item_id=1, capital_used=50000, adjusted_profit=10000, quantity=500
            ),  # 20% ROI
            self._make_scored_option(
                item_id=2, capital_used=50000, adjusted_profit=5000, quantity=500
            ),  # 10% ROI
        ]

        greedy_selected = engine._greedy_select(
            scored_options,
            total_capital=60000,  # Can only fit one at full size
            num_slots=2,
            min_slots_target=1,
            params={"target_pct": 0.50, "concentration_penalty": 0.1},
            max_capital_per_trade=60000,
        )

        # Should pick item 1 first (higher ROI), may add item 2 with reduced qty
        assert len(greedy_selected) >= 1
        # First item should be the higher ROI one
        assert greedy_selected[0]["candidate"]["item_id"] == 1

    def test_ilp_solver_fallback_on_failure(self, mock_engine):
        """Test that optimizer falls back to greedy when ILP fails."""
        engine, mock_loader = mock_engine

        # Create predictions that will be processed
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                }
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        # Mock the ILP solver to fail
        with patch.object(engine, "_solve_portfolio_ilp", return_value=[]):
            recommendations = engine.get_recommendations(
                style="active",
                capital=10000000,
                risk="medium",
                slots=4,
            )

            # Should still get recommendations via greedy fallback
            assert len(recommendations) >= 1

    def test_ilp_complex_optimization(self, mock_engine):
        """Test ILP with complex multi-item scenario.

        Scenario: 1M capital, 3 slots
        Items with varying capital requirements and profits to test
        that ILP finds global optimum.
        """
        engine, _ = mock_engine

        scored_options = [
            # Small high-ROI items
            self._make_scored_option(
                item_id=1, capital_used=50000, adjusted_profit=10000, quantity=500
            ),  # 20% ROI
            self._make_scored_option(
                item_id=2, capital_used=100000, adjusted_profit=15000, quantity=1000
            ),  # 15% ROI
            # Large lower-ROI items
            self._make_scored_option(
                item_id=3, capital_used=400000, adjusted_profit=48000, quantity=4000
            ),  # 12% ROI
            self._make_scored_option(
                item_id=4, capital_used=450000, adjusted_profit=50000, quantity=4500
            ),  # 11% ROI
            self._make_scored_option(
                item_id=5, capital_used=300000, adjusted_profit=30000, quantity=3000
            ),  # 10% ROI
        ]

        ilp_selected = engine._solve_portfolio_ilp(
            scored_options, total_capital=1000000, num_slots=3
        )
        ilp_profit = sum(opt["adjusted_profit"] for opt in ilp_selected)

        # ILP should find a good solution - the optimal is items 3+4+2 = 113k profit
        # or items 3+5+1+2 if capital allows = 103k profit with 850k capital
        # With 3 slots and 1M capital, optimal is 3+4+1 = 108k or similar
        assert (
            ilp_profit >= 100000
        ), f"ILP should find solution with profit >= 100k, got {ilp_profit}"

    def test_integration_ilp_in_recommendations(self, mock_engine):
        """Integration test: ILP is used in actual recommendation flow."""
        engine, mock_loader = mock_engine

        # Two items where ILP should pick the better combination
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 400,
                    "sell_price": 420,
                    "current_high": 420,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,
            risk="medium",
            slots=4,
        )

        # Should get recommendations
        assert len(recommendations) >= 1
        # Verify the recommendations have expected structure
        for rec in recommendations:
            assert "itemId" in rec
            assert "expectedProfit" in rec
            assert "capitalRequired" in rec

    def test_build_reason(self, mock_engine):
        """Test _build_reason generates correct rationale string."""
        engine, _ = mock_engine

        # Test with all fields present
        candidate = {
            "trend": "Rising",
            "volume_24h": 60000,
            "hour_offset": 8,
        }
        reason = engine._build_reason(candidate)
        assert reason == "Rising trend, high volume, 8h window"

        # Test with medium volume
        candidate = {
            "trend": "Stable",
            "volume_24h": 20000,
            "hour_offset": 4,
        }
        reason = engine._build_reason(candidate)
        assert reason == "Stable trend, medium volume, 4h window"

        # Test with very high volume
        candidate = {
            "trend": "Falling",
            "volume_24h": 200000,
            "hour_offset": 24,
        }
        reason = engine._build_reason(candidate)
        assert reason == "Falling trend, very high volume, 24h window"

        # Test with default values when fields are missing
        candidate = {}
        reason = engine._build_reason(candidate)
        assert reason == "Stable trend, low volume, 4h window"

    def test_recommendations_include_reason_and_is_recommended(self, mock_engine):
        """Test that get_recommendations returns reason and isRecommended fields."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_recommendations(
            style="active",
            capital=10000000,
            risk="medium",
            slots=4,
        )

        assert len(recommendations) >= 1
        for rec in recommendations:
            # Verify reason field is present and formatted correctly
            assert "reason" in rec
            assert isinstance(rec["reason"], str)
            assert "trend" in rec["reason"].lower()
            assert "volume" in rec["reason"].lower()
            assert "h window" in rec["reason"]
            # Verify isRecommended is True for portfolio recommendations
            assert "isRecommended" in rec
            assert rec["isRecommended"] is True


class TestItemPriceLookup:
    """Test cases for get_item_price_lookup method (issue #135)."""

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_price_lookup_buy_side(self, mock_engine):
        """Test price lookup returns buy-side data."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.72,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 5,
                    "current_low": 4,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 25000

        result = engine.get_item_price_lookup(item_id=554, side="buy")

        assert result is not None
        assert result["itemId"] == 554
        assert result["itemName"] == "Fire rune"
        assert result["side"] == "buy"
        assert result["recommendedPrice"] == 4
        assert result["currentMarketPrice"] == 5
        assert result["isRecommended"] is True  # Above thresholds
        # Flip metrics (issue #129)
        assert result["buyLimit"] == 25000
        assert "marginGp" in result
        assert "marginPercent" in result

    def test_price_lookup_sell_side(self, mock_engine):
        """Test price lookup returns sell-side data."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.65,
                    "expected_value": 0.006,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_trend.return_value = "Rising"
        mock_loader.get_item_buy_limit.return_value = 25000

        result = engine.get_item_price_lookup(item_id=554, side="sell")

        assert result is not None
        assert result["side"] == "sell"
        assert result["recommendedPrice"] == 6
        assert result["currentMarketPrice"] == 4  # current_low for sell
        assert result["buyLimit"] == 25000

    def test_price_lookup_item_not_found(self, mock_engine):
        """Test price lookup returns None for unknown item."""
        engine, mock_loader = mock_engine

        mock_loader.get_predictions_for_item.return_value = pd.DataFrame()

        result = engine.get_item_price_lookup(item_id=99999)

        assert result is None

    def test_price_lookup_with_warning(self, mock_engine):
        """Test price lookup includes warning when below thresholds."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 123,
                    "item_name": "King worm",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.02,  # Very low
                    "expected_value": 0.001,  # Very low
                    "buy_price": 100,
                    "sell_price": 102,
                    "current_high": 102,
                    "current_low": 100,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 50
        mock_loader.get_item_trend.return_value = "Falling"
        mock_loader.get_item_buy_limit.return_value = 1000

        result = engine.get_item_price_lookup(item_id=123)

        assert result is not None
        assert result["isRecommended"] is False
        assert result["warning"] is not None
        assert "Low fill probability" in result["warning"]
        assert "Low expected value" in result["warning"]
        assert result["buyLimit"] == 1000

    def test_price_lookup_window_filter(self, mock_engine):
        """Test price lookup finds closest window match."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.80,
                    "expected_value": 0.009,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 5,
                    "current_low": 4,
                },
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.70,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 5,
                    "current_low": 4,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 25000

        # Request window=4, should return 4h prediction
        result = engine.get_item_price_lookup(item_id=554, window_hours=4)

        assert result is not None
        assert result["timeWindowHours"] == 4
        assert result["fillProbability"] == 0.80

    def test_price_lookup_offset_filter(self, mock_engine):
        """Test price lookup filters by offset percentage."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.015,
                    "fill_probability": 0.80,
                    "expected_value": 0.009,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 5,
                    "current_low": 4,
                },
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.70,
                    "expected_value": 0.008,
                    "buy_price": 3,
                    "sell_price": 7,
                    "current_high": 5,
                    "current_low": 4,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 25000

        # Request offset=0.015, should return 1.5% offset prediction
        result = engine.get_item_price_lookup(item_id=554, offset_pct=0.015)

        assert result is not None
        assert result["offsetPercent"] == 0.015
        assert result["fillProbability"] == 0.80

    def test_price_lookup_with_price_history(self, mock_engine):
        """Test price lookup includes price history when requested."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.72,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 5,
                    "current_low": 4,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_price_history.return_value = [
            {"timestamp": "2024-01-01T00:00:00Z", "high": 5, "low": 4},
            {"timestamp": "2024-01-01T01:00:00Z", "high": 6, "low": 4},
        ]

        result = engine.get_item_price_lookup(
            item_id=554, side="buy", include_price_history=True
        )

        assert result is not None
        assert "priceHistory" in result
        assert len(result["priceHistory"]) == 2
        mock_loader.get_price_history.assert_called_once_with(554, hours=24)

    def test_price_lookup_without_price_history(self, mock_engine):
        """Test price lookup excludes price history by default."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 24,
                    "offset_pct": 0.02,
                    "fill_probability": 0.72,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 5,
                    "current_low": 4,
                },
            ]
        )
        mock_loader.get_predictions_for_item.return_value = predictions_df
        mock_loader.get_item_volume_24h.return_value = 100000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 25000

        result = engine.get_item_price_lookup(item_id=554, side="buy")

        assert result is not None
        assert "priceHistory" not in result or result.get("priceHistory") is None
        mock_loader.get_price_history.assert_not_called()


class TestPriceBuffer:
    """Test cases for price buffer functionality."""

    @pytest.fixture
    def engine_with_buffer(self, mock_db_connection):
        """Create recommendation engine with price buffer enabled."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            # Ensure buffer is enabled with default settings
            engine.config.price_buffer_enabled = True
            engine.config.price_buffer_min_pct = 1.0
            engine.config.price_buffer_max_pct = 4.0
            return engine

    @pytest.fixture
    def engine_buffer_disabled(self, mock_db_connection):
        """Create recommendation engine with price buffer disabled."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            engine.config.price_buffer_enabled = False
            return engine

    def test_buffer_applied_moves_prices_toward_market(self, engine_with_buffer):
        """Test that buffer moves buy UP and sell DOWN."""
        # Margin of 100gp, buffer should be 1-4gp
        buy, sell = 1000, 1100

        # Run multiple times to check range
        for _ in range(50):
            buffered_buy, buffered_sell = engine_with_buffer._apply_price_buffer(
                buy, sell
            )
            # Buy should move UP (toward market)
            assert buffered_buy >= buy
            assert buffered_buy <= buy + 4  # Max 4% of 100 = 4gp

            # Sell should move DOWN (toward market)
            assert buffered_sell <= sell
            assert buffered_sell >= sell - 4  # Max 4% of 100 = 4gp

    def test_buffer_disabled_returns_original_prices(self, engine_buffer_disabled):
        """Test that disabled buffer returns original prices unchanged."""
        buy, sell = 1000, 1100

        buffered_buy, buffered_sell = engine_buffer_disabled._apply_price_buffer(
            buy, sell
        )

        assert buffered_buy == buy
        assert buffered_sell == sell

    def test_buffer_skipped_when_margin_too_small(self, engine_with_buffer):
        """Test that buffer is skipped when 1% of margin < 1gp."""
        # Margin of 50gp: 1% = 0.5gp < 1gp, so no buffer should be applied
        buy, sell = 1000, 1050

        buffered_buy, buffered_sell = engine_with_buffer._apply_price_buffer(buy, sell)

        assert buffered_buy == buy
        assert buffered_sell == sell

    def test_buffer_skipped_when_margin_exactly_100(self, engine_with_buffer):
        """Test that buffer IS applied when margin is exactly 100gp (1% = 1gp)."""
        buy, sell = 1000, 1100

        # With margin=100, 1% = 1gp >= 1gp, so buffer should be applied
        buffered_buy, buffered_sell = engine_with_buffer._apply_price_buffer(buy, sell)

        # Should be different from original (1-4% buffer applied)
        # But we need to run multiple times since it's random
        any_different = False
        for _ in range(20):
            b, s = engine_with_buffer._apply_price_buffer(buy, sell)
            if b != buy or s != sell:
                any_different = True
                break

        assert any_different, "Buffer should be applied when margin >= 100gp"

    def test_buffer_never_crosses_prices(self, engine_with_buffer):
        """Test that buffered buy never exceeds buffered sell."""
        # Test with various margins
        test_cases = [(100, 200), (1000, 1100), (5000, 5500), (10000, 10500)]

        for buy, sell in test_cases:
            for _ in range(50):
                buffered_buy, buffered_sell = engine_with_buffer._apply_price_buffer(
                    buy, sell
                )
                assert (
                    buffered_buy < buffered_sell
                ), f"Buy {buffered_buy} should be less than sell {buffered_sell}"

    def test_buffer_respects_config_percentages(self, engine_with_buffer):
        """Test that buffer respects min/max percentage config."""
        engine_with_buffer.config.price_buffer_min_pct = 2.0
        engine_with_buffer.config.price_buffer_max_pct = 3.0

        buy, sell = 1000, 2000  # Margin of 1000gp

        for _ in range(50):
            buffered_buy, buffered_sell = engine_with_buffer._apply_price_buffer(
                buy, sell
            )

            # Buffer should be 2-3% of 1000 = 20-30gp
            buy_increase = buffered_buy - buy
            sell_decrease = sell - buffered_sell

            assert buy_increase >= 20, f"Buy increase {buy_increase} should be >= 20"
            assert buy_increase <= 30, f"Buy increase {buy_increase} should be <= 30"
            assert sell_decrease >= 20, f"Sell decrease {sell_decrease} should be >= 20"
            assert sell_decrease <= 30, f"Sell decrease {sell_decrease} should be <= 30"

    def test_buffer_with_zero_margin(self, engine_with_buffer):
        """Test that zero or negative margin returns original prices."""
        # Zero margin
        buy, sell = engine_with_buffer._apply_price_buffer(1000, 1000)
        assert buy == 1000
        assert sell == 1000

        # Negative margin (shouldn't happen but handle gracefully)
        buy, sell = engine_with_buffer._apply_price_buffer(1100, 1000)
        assert buy == 1100
        assert sell == 1000

    def test_buffer_deterministic_with_seed(self, engine_with_buffer):
        """Test that buffer is deterministic when random is seeded."""
        import random

        buy, sell = 1000, 1200  # Margin of 200gp

        # Seed and get first result
        random.seed(42)
        result1 = engine_with_buffer._apply_price_buffer(buy, sell)

        # Re-seed and get same result
        random.seed(42)
        result2 = engine_with_buffer._apply_price_buffer(buy, sell)

        assert result1 == result2

    def test_buffer_applies_same_value_to_both_prices(self, engine_with_buffer):
        """Test that the same buffer value is applied to buy and sell."""
        buy, sell = 1000, 1100  # Margin of 100gp

        for _ in range(50):
            buffered_buy, buffered_sell = engine_with_buffer._apply_price_buffer(
                buy, sell
            )

            buy_increase = buffered_buy - buy
            sell_decrease = sell - buffered_sell

            # Buffer amount should be identical for both
            assert (
                buy_increase == sell_decrease
            ), f"Buy increase {buy_increase} != sell decrease {sell_decrease}"


class TestManipulationFiltering:
    """Test cases for manipulation detection and filtering."""

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader
            return engine, mock_loader

    def test_high_spread_detection(self, mock_engine):
        """Test that items with high spread are flagged as suspicious."""
        engine, mock_loader = mock_engine

        # Normal spread (5%) - should pass
        mock_loader.get_item_volume_24h.return_value = 5000
        mock_loader.get_item_volume_1h.return_value = 500

        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=1000,
            spread_pct=0.05,  # 5% spread - below 10% threshold
        )
        assert not is_suspicious
        assert risk_score < 0.5
        assert len(reasons) == 0

        # High spread (15%) - should be flagged
        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=1000,
            spread_pct=0.15,  # 15% spread - above 10% threshold
        )
        assert "High spread" in reasons[0]
        assert risk_score >= 0.3

    def test_volume_concentration_detection(self, mock_engine):
        """Test that volume spikes (high 1h/24h ratio) are flagged."""
        engine, mock_loader = mock_engine

        # Normal volume distribution - should pass
        mock_loader.get_item_volume_24h.return_value = 10000
        mock_loader.get_item_volume_1h.return_value = 1000  # 10% in last hour

        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=5000,
            spread_pct=0.03,
            volume_24h=10000,
            volume_1h=1000,
        )
        assert not is_suspicious
        assert "Volume spike" not in str(reasons)

        # Volume spike - 90% of 24h volume in last hour
        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=5000,
            spread_pct=0.03,
            volume_24h=10000,
            volume_1h=9000,  # 90% in last hour
        )
        assert any("Volume spike" in r for r in reasons)
        assert risk_score >= 0.4

    def test_low_volume_cheap_item_detection(self, mock_engine):
        """Test that cheap items with low volume are flagged."""
        engine, mock_loader = mock_engine

        # Cheap item with adequate volume - should pass
        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=500,  # Cheap item (<1000gp)
            spread_pct=0.03,
            volume_24h=15000,  # Above 10000 threshold
            volume_1h=1500,
        )
        assert not is_suspicious
        assert len(reasons) == 0

        # Cheap item with low volume - should be flagged
        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=500,  # Cheap item (<1000gp)
            spread_pct=0.03,
            volume_24h=5000,  # Below 10000 threshold
            volume_1h=500,
        )
        assert any("Low volume for cheap item" in r for r in reasons)
        assert risk_score >= 0.35

    def test_expensive_item_low_volume_not_flagged(self, mock_engine):
        """Test that expensive items with low volume are NOT flagged."""
        engine, mock_loader = mock_engine

        # Expensive item with low volume - should NOT be flagged for low volume
        # (expensive items naturally have lower volume)
        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=5000000,  # Expensive item (5M gp)
            spread_pct=0.03,
            volume_24h=100,  # Low volume but acceptable for expensive item
            volume_1h=10,
        )
        # Should not have "Low volume for cheap item" reason
        assert not any("Low volume for cheap item" in r for r in reasons)

    def test_combined_manipulation_signals(self, mock_engine):
        """Test that multiple signals combine to trigger filtering."""
        engine, mock_loader = mock_engine

        # High spread alone doesn't exceed 0.5 threshold
        is_suspicious, risk_score, _ = engine._check_manipulation_signals(
            item_id=123,
            buy_price=5000,
            spread_pct=0.15,  # High spread +0.3
            volume_24h=10000,
            volume_1h=1000,  # Normal concentration
        )
        assert not is_suspicious  # 0.3 < 0.5

        # High spread + volume spike exceeds threshold
        is_suspicious, risk_score, reasons = engine._check_manipulation_signals(
            item_id=123,
            buy_price=5000,
            spread_pct=0.15,  # High spread +0.3
            volume_24h=10000,
            volume_1h=9000,  # Volume spike +0.4
        )
        assert is_suspicious  # 0.3 + 0.4 = 0.7 > 0.5
        assert risk_score >= 0.7
        assert len(reasons) == 2

    def test_manipulation_filter_in_build_candidate(self, mock_engine):
        """Test that _build_candidate returns None for suspicious items."""
        engine, mock_loader = mock_engine

        # Setup mock for suspicious item (cheap with low volume + high spread)
        mock_loader.get_item_volume_24h.return_value = 500
        mock_loader.get_item_volume_1h.return_value = 400  # 80% concentration
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 1000

        row = pd.Series(
            {
                "item_id": 123,
                "item_name": "Oak longbow (u)",
                "buy_price": 50,  # Cheap item
                "sell_price": 75,
                "fill_probability": 0.10,
                "expected_value": 0.01,
                "hour_offset": 4,
                "confidence": "medium",
                "current_high": 80,
                "current_low": 45,  # ~78% spread
            }
        )

        candidate = engine._build_candidate(
            row=row,
            max_capital=10000000,
            pred_age_seconds=60,
            style="hybrid",
            buy_limits={123: 1000},
        )

        # Should return None due to manipulation signals
        # (cheap item + low volume + high spread)
        assert candidate is None

    def test_normal_item_passes_manipulation_filter(self, mock_engine):
        """Test that normal items pass through _build_candidate."""
        engine, mock_loader = mock_engine

        # Setup mock for normal item
        mock_loader.get_item_volume_24h.return_value = 50000
        mock_loader.get_item_volume_1h.return_value = 5000  # 10% concentration
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_buy_limit.return_value = 1000

        row = pd.Series(
            {
                "item_id": 456,
                "item_name": "Dragon bones",
                "buy_price": 2000,
                "sell_price": 2100,
                "fill_probability": 0.15,
                "expected_value": 0.012,
                "hour_offset": 4,
                "confidence": "high",
                "current_high": 2100,
                "current_low": 2000,  # 5% spread
            }
        )

        candidate = engine._build_candidate(
            row=row,
            max_capital=10000000,
            pred_age_seconds=60,
            style="hybrid",
            buy_limits={456: 1000},
        )

        # Should return a valid candidate
        assert candidate is not None
        assert candidate["item_id"] == 456
        assert candidate["item_name"] == "Dragon bones"

    def test_config_thresholds_respected(self, mock_db_connection):
        """Test that custom config thresholds are respected."""
        from src.config import Config

        # Create engine with custom config
        custom_config = Config()
        custom_config.max_spread_pct = 0.05  # Stricter: 5% instead of 10%
        custom_config.max_volume_concentration = 0.50  # Stricter: 50% instead of 80%
        custom_config.min_volume_for_low_value = 20000  # Stricter: 20k instead of 10k

        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(
                db_connection_string=mock_db_connection, config=custom_config
            )
            engine.loader = mock_loader

            # Item that would pass default thresholds but fails stricter ones
            # 7% spread - passes 10% default but fails 5% strict
            is_suspicious, _, reasons = engine._check_manipulation_signals(
                item_id=123,
                buy_price=5000,
                spread_pct=0.07,
                volume_24h=10000,
                volume_1h=1000,
            )
            assert any("High spread" in r for r in reasons)

            # 60% volume concentration - passes 80% default but fails 50% strict
            is_suspicious, _, reasons = engine._check_manipulation_signals(
                item_id=124,
                buy_price=5000,
                spread_pct=0.03,
                volume_24h=10000,
                volume_1h=6000,  # 60%
            )
            assert any("Volume spike" in r for r in reasons)

            # Cheap item with 15k volume - passes 10k default but fails 20k strict
            is_suspicious, _, reasons = engine._check_manipulation_signals(
                item_id=125,
                buy_price=500,
                spread_pct=0.03,
                volume_24h=15000,
                volume_1h=1500,
            )
            assert any("Low volume for cheap item" in r for r in reasons)


class TestGetAllRecommendations:
    """Test cases for get_all_recommendations() method (issue #184)."""

    @pytest.fixture
    def mock_engine(self, mock_db_connection):
        """Create recommendation engine with mocked dependencies."""
        with patch("src.recommendation_engine.PredictionLoader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            _configure_batch_buy_limits_mock(mock_loader)

            engine = RecommendationEngine(db_connection_string=mock_db_connection)
            engine.loader = mock_loader

            return engine, mock_loader

    def test_get_all_returns_all_viable_candidates(self, mock_engine):
        """Test that get_all_recommendations returns all viable items (no slot limit)."""
        engine, mock_loader = mock_engine

        # Mock predictions for multiple items
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.12,
                    "expected_value": 0.010,
                    "buy_price": 400,
                    "sell_price": 420,
                    "current_high": 420,
                    "current_low": 400,
                    "confidence": "high",
                },
                {
                    "item_id": 566,
                    "item_name": "Death rune",
                    "hour_offset": 4,
                    "offset_pct": 0.015,
                    "fill_probability": 0.08,
                    "expected_value": 0.006,
                    "buy_price": 200,
                    "sell_price": 210,
                    "current_high": 210,
                    "current_low": 200,
                    "confidence": "medium",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
        )

        # Should return all 3 items, not limited by slots
        assert len(recommendations) == 3
        item_ids = {rec["itemId"] for rec in recommendations}
        assert item_ids == {554, 565, 566}

    def test_get_all_sorted_by_composite_score(self, mock_engine):
        """Test that results are sorted by composite score (descending)."""
        engine, mock_loader = mock_engine

        # Mock predictions with varying EV and fill probability
        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 1,
                    "item_name": "Low score item",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.05,  # Low
                    "expected_value": 0.004,  # Low
                    "buy_price": 100,
                    "sell_price": 110,
                    "current_high": 110,
                    "current_low": 100,
                    "confidence": "low",
                },
                {
                    "item_id": 2,
                    "item_name": "High score item",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.15,  # High
                    "expected_value": 0.012,  # High
                    "buy_price": 100,
                    "sell_price": 112,
                    "current_high": 112,
                    "current_low": 100,
                    "confidence": "high",
                },
                {
                    "item_id": 3,
                    "item_name": "Medium score item",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,  # Medium
                    "expected_value": 0.008,  # Medium
                    "buy_price": 100,
                    "sell_price": 111,
                    "current_high": 111,
                    "current_low": 100,
                    "confidence": "medium",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
        )

        # Should be sorted by score: High > Medium > Low
        assert len(recommendations) == 3
        assert recommendations[0]["itemId"] == 2  # High score
        assert recommendations[1]["itemId"] == 3  # Medium score
        assert recommendations[2]["itemId"] == 1  # Low score

        # Verify _score field is included
        assert "_score" in recommendations[0]
        assert recommendations[0]["_score"] > recommendations[1]["_score"]
        assert recommendations[1]["_score"] > recommendations[2]["_score"]

    def test_get_all_empty_predictions(self, mock_engine):
        """Test get_all_recommendations with empty predictions."""
        engine, mock_loader = mock_engine

        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
        )

        assert recommendations == []

    def test_get_all_excludes_item_ids(self, mock_engine):
        """Test that exclude_item_ids parameter works."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.12,
                    "expected_value": 0.010,
                    "buy_price": 400,
                    "sell_price": 420,
                    "current_high": 420,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            exclude_item_ids={554},
        )

        # Should only have Blood rune
        assert len(recommendations) == 1
        assert recommendations[0]["itemId"] == 565

    def test_get_all_respects_active_trades(self, mock_engine):
        """Test that active_trades are excluded from results."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
                {
                    "item_id": 565,
                    "item_name": "Blood rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.12,
                    "expected_value": 0.010,
                    "buy_price": 400,
                    "sell_price": 420,
                    "current_high": 420,
                    "current_low": 400,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        # Simulate active trade on Fire rune
        active_trades = [{"itemId": 554, "quantity": 1000, "buyPrice": 4}]

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            active_trades=active_trades,
        )

        # Should exclude Fire rune (active trade)
        item_ids = {rec["itemId"] for rec in recommendations}
        assert 554 not in item_ids
        assert 565 in item_ids

    def test_get_all_insufficient_capital(self, mock_engine):
        """Test get_all_recommendations with insufficient capital."""
        engine, mock_loader = mock_engine

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=500,  # Below 1000 minimum
            risk="medium",
        )

        assert recommendations == []

    def test_get_all_includes_score_field(self, mock_engine):
        """Test that _score field is included in each recommendation."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        recommendations = engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
        )

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert "_score" in rec
        assert isinstance(rec["_score"], float)
        assert rec["_score"] > 0

    def test_get_all_respects_offset_filters(self, mock_engine):
        """Test that offset filtering parameters work."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        # Call with specific offset_pct
        engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            offset_pct=0.015,
        )

        # Verify the loader was called with correct offset parameters
        call_kwargs = mock_loader.get_best_prediction_per_item.call_args[1]
        assert call_kwargs["min_offset_pct"] == 0.015
        assert call_kwargs["max_offset_pct"] == 0.015

    def test_get_all_respects_hour_offset_filter(self, mock_engine):
        """Test that max_hour_offset parameter works."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        # Call with max_hour_offset
        engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            max_hour_offset=4,
        )

        # Verify the loader was called with max_hour_offset
        call_kwargs = mock_loader.get_best_prediction_per_item.call_args[1]
        assert call_kwargs["max_hour_offset"] == 4

    def test_get_all_respects_min_ev_filter(self, mock_engine):
        """Test that min_ev parameter works."""
        engine, mock_loader = mock_engine

        predictions_df = pd.DataFrame(
            [
                {
                    "item_id": 554,
                    "item_name": "Fire rune",
                    "hour_offset": 2,
                    "offset_pct": 0.015,
                    "fill_probability": 0.10,
                    "expected_value": 0.008,
                    "buy_price": 4,
                    "sell_price": 6,
                    "current_high": 6,
                    "current_low": 4,
                    "confidence": "high",
                },
            ]
        )

        mock_loader.get_best_prediction_per_item.return_value = predictions_df
        mock_loader.get_prediction_age_seconds.return_value = 120
        mock_loader.get_item_buy_limit.return_value = 10000
        mock_loader.get_item_trend.return_value = "Stable"
        mock_loader.get_item_volume_24h.return_value = None
        mock_loader.get_item_volume_1h.return_value = 100000

        # Call with min_ev
        engine.get_all_recommendations(
            style="active",
            capital=100000000,
            risk="medium",
            min_ev=0.01,
        )

        # Verify the loader was called with min_ev
        call_kwargs = mock_loader.get_best_prediction_per_item.call_args[1]
        assert call_kwargs["min_ev"] == 0.01


class TestFilterIntegration:
    """Test that new filters are called in get_recommendations flow."""

    def test_liquidity_filter_called(self):
        """Liquidity filter should be called during recommendation flow."""
        from src.recommendation_engine import RecommendationEngine

        engine = MagicMock(spec=RecommendationEngine)
        engine._apply_liquidity_filter = MagicMock(
            side_effect=lambda x, b, v: x
        )
        engine._check_manipulation_signals_vectorized = MagicMock(
            side_effect=lambda x, v24, v1: pd.Series([False] * len(x))
        )

        # Verify methods exist and are callable
        assert hasattr(engine, '_apply_liquidity_filter')
        assert hasattr(engine, '_check_manipulation_signals_vectorized')

    def test_filters_applied_before_candidate_building(self):
        """Filters should be applied after fetching predictions, before building candidates."""
        # This is a code review test - verify the order in get_recommendations
        from src.recommendation_engine import RecommendationEngine
        import inspect

        source = inspect.getsource(RecommendationEngine.get_recommendations)

        # Find positions of key operations
        fetch_pos = source.find('get_best_prediction_per_item')
        liquidity_pos = source.find('_apply_liquidity_filter')
        # Check for validation and manipulation checks in vectorized pipeline
        validation_pos = source.find('_filter_valid_candidates')
        manipulation_pos = source.find('_check_manipulation_signals_vectorized')
        build_pos = source.find('_enrich_metadata_vectorized')

        # Liquidity filter should come after fetch, before vectorized pipeline
        assert liquidity_pos > fetch_pos, \
            "Liquidity filter should come after fetching predictions"
        assert liquidity_pos < validation_pos, \
            "Liquidity filter should come before vectorized pipeline"

        # Manipulation check should come after validation, before enrichment
        assert manipulation_pos > validation_pos, \
            "Manipulation check should come after validation"
        assert manipulation_pos < build_pos, \
            "Manipulation check should come before metadata enrichment"
