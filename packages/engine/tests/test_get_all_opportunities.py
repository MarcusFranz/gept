"""Tests for get_all_opportunities() method in RecommendationEngine.

These tests establish a safety net before refactoring iterrows() to vectorized operations.
"""

from unittest.mock import MagicMock

import pandas as pd

from src.recommendation_engine import RecommendationEngine


def _configure_batch_mocks(mock_loader):
    """Configure batch methods to return dicts based on individual method return values."""

    def batch_buy_limits_side_effect(item_ids):
        result = {}
        for item_id in item_ids:
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
    mock_loader.get_batch_trends.side_effect = batch_trends_side_effect


class TestGetAllOpportunities:
    """Test suite for get_all_opportunities() method."""

    def test_returns_empty_list_when_no_predictions(self):
        """Verify empty list returned when no predictions available."""
        mock_loader = MagicMock()
        mock_loader.get_best_prediction_per_item.return_value = pd.DataFrame()

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader

        opportunities = engine.get_all_opportunities()

        assert opportunities == []
        assert isinstance(opportunities, list)

    def test_returns_valid_opportunity_structure(self):
        """Verify opportunity dicts have required fields with correct types."""
        mock_loader = MagicMock()

        # Mock predictions with one valid opportunity
        predictions_df = pd.DataFrame([{
            "item_id": 554,
            "item_name": "Fire rune",
            "buy_price": 4,
            "sell_price": 10,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        # Mock batch data (ensure buy_limit/volume ratio < 0.05 to pass liquidity filter)
        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter0  # High volume to pass filter
        mock_loader.get_item_trend.return_value = "Rising"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        opp = opportunities[0]

        # Verify required fields exist
        required_fields = [
            "item_id", "item_name", "icon_url", "buy_price", "sell_price",
            "quantity", "capital_required", "expected_profit", "expected_hours",
            "confidence", "fill_probability", "expected_value", "volume_24h",
            "trend", "category", "hour_offset", "volume_tier"
        ]
        for field in required_fields:
            assert field in opp, f"Missing required field: {field}"

        # Verify types
        assert isinstance(opp["item_id"], int)
        assert isinstance(opp["buy_price"], int)
        assert isinstance(opp["sell_price"], int)
        assert isinstance(opp["quantity"], int)
        assert isinstance(opp["capital_required"], int)
        assert isinstance(opp["expected_profit"], int)
        assert isinstance(opp["fill_probability"], float)

    def test_filters_negative_profit_opportunities(self):
        """Verify opportunities with negative profit are excluded."""
        mock_loader = MagicMock()

        # Mock predictions: one profitable, one unprofitable
        predictions_df = pd.DataFrame([
            {
                "item_id": 554,
                "item_name": "Fire rune",
                "buy_price": 10,  # Buy high
                "sell_price": 5,  # Sell low = negative profit
                "fill_probability": 0.25,
                "expected_value": 0.015,
                "hour_offset": 12,
                "confidence": "high",
            },
            {
                "item_id": 555,
                "item_name": "Water rune",
                "buy_price": 4,
                "sell_price": 10,  # Profitable
                "fill_probability": 0.25,
                "expected_value": 0.015,
                "hour_offset": 12,
                "confidence": "high",
            }
        ])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        # Only the profitable one should be returned
        assert len(opportunities) == 1
        assert opportunities[0]["item_id"] == 555

    def test_respects_buy_limit_for_quantity(self):
        """Verify max quantity is capped by buy limit."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 554,
            "item_name": "Fire rune",
            "buy_price": 100,  # Low price means capital allows many
            "sell_price": 200,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        # Buy limit is the constraint (not capital)
        buy_limit = 5000
        mock_loader.get_item_buy_limit.return_value = buy_limit
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        # Quantity should be capped by buy limit (not default 1B capital)
        assert opportunities[0]["quantity"] == buy_limit

    def test_respects_capital_for_quantity(self):
        """Verify max quantity is capped by available capital when buy limit is higher."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 2,
            "item_name": "Twisted bow",  # Expensive item
            "buy_price": 500_000_000,  # 500M per item
            "sell_price": 550_000_000,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        # Buy limit is high (not the constraint)
        # Ensure liquidity filter passes: buy_limit/volume < 0.05
        mock_loader.get_item_buy_limit.return_value = 10
        mock_loader.get_item_volume_24h.return_value = 1000  # 10/1000 = 0.01 < 0.05
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        # With 1B capital and 500M price, max quantity = 2
        # (even though buy limit is 10)
        assert opportunities[0]["quantity"] == 2

    def test_filters_zero_quantity_opportunities(self):
        """Verify opportunities with zero quantity are excluded."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([
            {
                "item_id": 1,
                "item_name": "Zero quantity item",
                "buy_price": 2_000_000_000,  # Price > default capital
                "sell_price": 2_100_000_000,
                "fill_probability": 0.25,
                "expected_value": 0.015,
                "hour_offset": 12,
                "confidence": "high",
            },
            {
                "item_id": 2,
                "item_name": "Valid item",
                "buy_price": 100,
                "sell_price": 200,
                "fill_probability": 0.25,
                "expected_value": 0.015,
                "hour_offset": 12,
                "confidence": "high",
            }
        ])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        mock_loader.get_item_buy_limit.return_value = 10
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        # Only item 2 should be returned (item 1 has qty=0)
        assert len(opportunities) == 1
        assert opportunities[0]["item_id"] == 2

    def test_calculates_ge_tax_correctly(self):
        """Verify GE tax (1%) is applied to profit calculation."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 554,
            "item_name": "Fire rune",
            "buy_price": 1000,
            "sell_price": 2000,
            "fill_probability": 1.0,  # 100% fill
            "expected_value": 0.5,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        mock_loader.get_item_buy_limit.return_value = 100
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        opp = opportunities[0]

        # Profit = (sell - buy - tax) * quantity * fill_prob
        # Tax = 2% of sell = 0.02 * 2000 = 40
        # Profit per unit = 2000 - 1000 - 40 = 960
        # Expected profit = 960 * 100 * 1.0 = 96,000
        assert opp["expected_profit"] == 96000

    def test_returns_empty_after_liquidity_filter(self):
        """Verify empty list when all items filtered by liquidity check."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 554,
            "item_name": "Fire rune",
            "buy_price": 100,
            "sell_price": 200,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        # Set up liquidity filter to reject (buy_limit >> volume)
        mock_loader.get_item_buy_limit.return_value = 100000
        mock_loader.get_item_volume_24h.return_value = 10  # Very low volume
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        # Should be filtered by liquidity check
        assert opportunities == []

    def test_includes_volume_tier_metadata(self):
        """Verify volume_tier is correctly computed and included."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 554,
            "item_name": "Fire rune",
            "buy_price": 100,
            "sell_price": 200,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Rising"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        assert "volume_tier" in opportunities[0]
        # Volume tier should be a string or None
        assert isinstance(opportunities[0]["volume_tier"], (str, type(None)))

    def test_handles_missing_item_name_gracefully(self):
        """Verify fallback item name when item_name is missing."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 999,
            # No item_name field
            "buy_price": 100,
            "sell_price": 200,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        # Should use fallback name
        assert opportunities[0]["item_name"] == "Item 999"

    def test_icon_url_format(self):
        """Verify icon URL is correctly formatted."""
        mock_loader = MagicMock()

        predictions_df = pd.DataFrame([{
            "item_id": 554,
            "item_name": "Fire rune",
            "buy_price": 100,
            "sell_price": 200,
            "fill_probability": 0.25,
            "expected_value": 0.015,
            "hour_offset": 12,
            "confidence": "high",
        }])
        mock_loader.get_best_prediction_per_item.return_value = predictions_df

        mock_loader.get_item_buy_limit.return_value = 25000
        mock_loader.get_item_volume_24h.return_value = 1000000  # High volume to pass liquidity filter
        mock_loader.get_item_trend.return_value = "Stable"
        _configure_batch_mocks(mock_loader)

        mock_db_connection = "postgresql://test:test@localhost/test"
        engine = RecommendationEngine(db_connection_string=mock_db_connection)
        engine.loader = mock_loader
        opportunities = engine.get_all_opportunities()

        assert len(opportunities) == 1
        expected_url = "https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id=554"
        assert opportunities[0]["icon_url"] == expected_url
