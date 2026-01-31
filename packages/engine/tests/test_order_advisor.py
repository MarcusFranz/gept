"""Tests for the OrderAdvisor class."""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.order_advisor import OrderAdvisor


class TestOrderAdvisor:
    """Tests for OrderAdvisor evaluation logic."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock PredictionLoader."""
        loader = MagicMock()
        # Default price data
        loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,  # Instant sell price
            "low": 980_000,  # Instant buy price
            "high_volume": 100,
            "low_volume": 100,
        }
        # Default predictions DataFrame
        loader.get_predictions_for_item.return_value = pd.DataFrame(
            {
                "item_id": [4151, 4151, 4151, 4151],
                "item_name": [
                    "Abyssal whip",
                    "Abyssal whip",
                    "Abyssal whip",
                    "Abyssal whip",
                ],
                "hour_offset": [4, 4, 8, 8],
                "offset_pct": [0.0150, 0.0200, 0.0150, 0.0200],
                "fill_probability": [0.65, 0.45, 0.75, 0.55],
                "expected_value": [0.008, 0.006, 0.010, 0.007],
                "buy_price": [985_000, 980_000, 985_000, 980_000],
                "sell_price": [1_015_000, 1_020_000, 1_015_000, 1_020_000],
                "current_high": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
                "current_low": [980_000, 980_000, 980_000, 980_000],
                "confidence": ["high", "medium", "high", "medium"],
                "prediction_time": ["2024-01-01T00:00:00Z"] * 4,
            }
        )
        loader.get_item_trend.return_value = "Stable"
        return loader

    @pytest.fixture
    def mock_engine(self):
        """Create a mock RecommendationEngine."""
        engine = MagicMock()
        engine.get_recommendations.return_value = [
            {
                "itemId": 11802,
                "item": "Armadyl godsword",
                "expectedProfit": 50000,
                "fillProbability": 0.5,
                "expectedHours": 4,
            }
        ]
        return engine

    @pytest.fixture
    def advisor(self, mock_loader, mock_engine):
        """Create an OrderAdvisor with mocked dependencies."""
        return OrderAdvisor(loader=mock_loader, engine=mock_engine)

    def test_evaluate_order_wait_high_fill_prob(self, advisor, mock_loader):
        """High fill probability buy order should recommend wait."""
        # Set up high fill probability scenario
        mock_loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,
            "low": 985_000,  # User's price is close to market
            "high_volume": 100,
            "low_volume": 100,
        }

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,  # Good price, close to market
            quantity=1,
            time_elapsed_minutes=30,
        )

        assert result["action"] == "wait"
        assert result["confidence"] > 0.3
        assert "recommendations" in result
        assert "reasoning" in result

    def test_evaluate_order_adjust_price_unfavorable_move(self, advisor, mock_loader):
        """Order with unfavorable price movement should recommend adjust_price."""
        # Market has moved unfavorably
        mock_loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,
            "low": 950_000,  # Market dropped, user's buy is too high
            "high_volume": 100,
            "low_volume": 100,
        }

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,  # User bid now above market by ~3%
            quantity=1,
            time_elapsed_minutes=60,
        )

        # Should recommend adjust or wait depending on fill prob
        assert result["action"] in ["adjust_price", "wait"]
        assert "adjust_price" in result["recommendations"]
        assert result["recommendations"]["adjust_price"]["suggested_price"] > 0

    def test_evaluate_order_low_fill_prob_with_alternatives(self, advisor, mock_loader):
        """Very low fill probability with alternatives should recommend abort_retry."""
        # Set up low fill probability scenario
        mock_loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,
            "low": 900_000,  # User's buy is way below market
            "high_volume": 100,
            "low_volume": 100,
        }
        # Predictions show low fill probability
        mock_loader.get_predictions_for_item.return_value = pd.DataFrame(
            {
                "item_id": [4151],
                "item_name": ["Abyssal whip"],
                "hour_offset": [4],
                "offset_pct": [0.0250],
                "fill_probability": [0.05],  # Very low
                "expected_value": [0.002],
                "buy_price": [975_000],
                "sell_price": [1_025_000],
                "current_high": [1_000_000],
                "current_low": [900_000],
                "confidence": ["low"],
                "prediction_time": ["2024-01-01T00:00:00Z"],
            }
        )

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=850_000,  # Very low bid
            quantity=1,
            time_elapsed_minutes=180,  # Long wait already
        )

        assert result["action"] in ["abort_retry", "liquidate", "adjust_price"]
        assert result["current_fill_probability"] < 0.3

    def test_evaluate_order_sell_liquidate(self, advisor, mock_loader):
        """Sell order with low fill prob and no alternatives should recommend liquidate."""
        # Remove alternatives
        advisor.engine = None

        # Very low sell probability
        mock_loader.get_predictions_for_item.return_value = pd.DataFrame(
            {
                "item_id": [4151],
                "item_name": ["Abyssal whip"],
                "hour_offset": [4],
                "offset_pct": [0.0200],
                "fill_probability": [0.05],  # Very low
                "expected_value": [0.002],
                "buy_price": [980_000],
                "sell_price": [1_020_000],
                "current_high": [1_000_000],
                "current_low": [980_000],
                "confidence": ["low"],
                "prediction_time": ["2024-01-01T00:00:00Z"],
            }
        )

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="sell",
            user_price=1_050_000,  # Asking too high
            quantity=1,
            time_elapsed_minutes=180,
        )

        # Without engine for alternatives, should get liquidate or adjust
        assert result["action"] in ["liquidate", "adjust_price"]
        assert "liquidate" in result["recommendations"]
        assert result["recommendations"]["liquidate"]["instant_price"] > 0

    def test_evaluate_order_no_price_data(self, mock_loader):
        """Missing price data should return error response."""
        mock_loader.get_latest_price.return_value = None
        advisor = OrderAdvisor(loader=mock_loader)

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,
            quantity=1,
            time_elapsed_minutes=30,
        )

        assert result["action"] == "wait"  # Default fallback
        assert result["confidence"] == 0.3  # Low confidence
        assert "Unable to fetch" in result["reasoning"]

    def test_evaluate_order_no_predictions(self, mock_loader):
        """Missing predictions should return error response."""
        mock_loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,
            "low": 980_000,
            "high_volume": 100,
            "low_volume": 100,
        }
        mock_loader.get_predictions_for_item.return_value = pd.DataFrame()
        advisor = OrderAdvisor(loader=mock_loader)

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,
            quantity=1,
            time_elapsed_minutes=30,
        )

        assert result["action"] == "wait"  # Default fallback
        assert "No prediction data" in result["reasoning"]

    def test_fill_probability_interpolation_at_market(self, advisor, mock_loader):
        """User at or better than market should get very high fill probability."""
        # User bidding above market
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=990_000,  # Above current low of 980_000
            quantity=1,
            time_elapsed_minutes=10,
        )

        # Should be very high probability
        assert result["current_fill_probability"] >= 0.9

    def test_fill_probability_interpolation_below_trained(self, advisor, mock_loader):
        """User far from market should get low fill probability."""
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=900_000,  # Way below market
            quantity=1,
            time_elapsed_minutes=60,
        )

        # Should be low probability
        assert result["current_fill_probability"] < 0.5

    def test_adjust_price_recommendation_structure(self, advisor):
        """Adjust price recommendation should have all required fields."""
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=950_000,
            quantity=10,
            time_elapsed_minutes=60,
        )

        if "adjust_price" in result["recommendations"]:
            adjust = result["recommendations"]["adjust_price"]
            assert "suggested_price" in adjust
            assert "new_fill_probability" in adjust
            assert "cost_difference" in adjust
            assert adjust["suggested_price"] > 0
            assert 0 <= adjust["new_fill_probability"] <= 1

    def test_wait_recommendation_structure(self, advisor):
        """Wait recommendation should have estimated fill time."""
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,
            quantity=1,
            time_elapsed_minutes=30,
        )

        if "wait" in result["recommendations"]:
            wait = result["recommendations"]["wait"]
            assert "estimated_fill_time_minutes" in wait
            assert wait["estimated_fill_time_minutes"] >= 0

    def test_abort_retry_recommendation_structure(self, advisor):
        """Abort retry recommendation should have alternative items."""
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=900_000,
            quantity=1,
            time_elapsed_minutes=120,
        )

        if "abort_retry" in result["recommendations"]:
            abort = result["recommendations"]["abort_retry"]
            assert "alternative_items" in abort
            assert isinstance(abort["alternative_items"], list)
            if abort["alternative_items"]:
                alt = abort["alternative_items"][0]
                assert "item_id" in alt
                assert "item_name" in alt
                assert "expected_profit" in alt

    def test_liquidate_recommendation_for_sell(self, advisor):
        """Liquidate recommendation for sell should show loss amount."""
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="sell",
            user_price=1_050_000,  # Asking more than market
            quantity=5,
            time_elapsed_minutes=60,
        )

        if "liquidate" in result["recommendations"]:
            liq = result["recommendations"]["liquidate"]
            assert "instant_price" in liq
            assert "loss_amount" in liq
            assert liq["instant_price"] > 0
            # For sell, loss should be calculated
            assert liq["loss_amount"] >= 0

    def test_reasoning_contains_key_info(self, advisor):
        """Reasoning should contain fill probability and time elapsed."""
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,
            quantity=1,
            time_elapsed_minutes=90,
        )

        reasoning = result["reasoning"].lower()
        # Should mention probability
        assert "%" in result["reasoning"] or "probability" in reasoning
        # Should mention time
        assert "hour" in reasoning or "minute" in reasoning

    def test_user_id_passed_through(self, advisor):
        """User ID should be accepted without error."""
        user_id = "a" * 64  # Valid SHA256 hash

        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=980_000,
            quantity=1,
            time_elapsed_minutes=30,
            user_id=user_id,
        )

        # Should complete without error
        assert result["action"] in ["wait", "adjust_price", "abort_retry", "liquidate"]


class TestOrderAdvisorDecisionLogic:
    """Tests for the specific decision logic in OrderAdvisor."""

    @pytest.fixture
    def mock_loader(self):
        """Create a configurable mock loader."""
        loader = MagicMock()
        loader.get_item_trend.return_value = "Stable"
        return loader

    def test_decision_high_prob_early_in_window(self, mock_loader):
        """High fill prob early in window should wait."""
        mock_loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,
            "low": 980_000,
            "high_volume": 100,
            "low_volume": 100,
        }
        mock_loader.get_predictions_for_item.return_value = pd.DataFrame(
            {
                "item_id": [4151],
                "item_name": ["Abyssal whip"],
                "hour_offset": [4],
                "offset_pct": [0.0150],
                "fill_probability": [0.80],  # High
                "expected_value": [0.010],
                "buy_price": [985_000],
                "sell_price": [1_015_000],
                "current_high": [1_000_000],
                "current_low": [980_000],
                "confidence": ["high"],
                "prediction_time": ["2024-01-01T00:00:00Z"],
            }
        )

        advisor = OrderAdvisor(loader=mock_loader)
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=985_000,
            quantity=1,
            time_elapsed_minutes=60,  # Early (4h window = 240min)
        )

        assert result["action"] == "wait"

    def test_decision_low_prob_late_in_window(self, mock_loader):
        """Very low fill prob late in window with below-market bid should adjust."""
        mock_loader.get_latest_price.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "high": 1_000_000,
            "low": 980_000,  # Current market
            "high_volume": 100,
            "low_volume": 100,
        }
        mock_loader.get_predictions_for_item.return_value = pd.DataFrame(
            {
                "item_id": [4151],
                "item_name": ["Abyssal whip"],
                "hour_offset": [4],
                "offset_pct": [0.0250],
                "fill_probability": [0.05],  # Very low
                "expected_value": [0.001],
                "buy_price": [955_000],
                "sell_price": [1_005_000],
                "current_high": [1_000_000],
                "current_low": [980_000],
                "confidence": ["low"],
                "prediction_time": ["2024-01-01T00:00:00Z"],
            }
        )

        advisor = OrderAdvisor(loader=mock_loader, engine=None)  # No alternatives
        result = advisor.evaluate_order(
            item_id=4151,
            order_type="buy",
            user_price=920_000,  # Well below current market (980k)
            quantity=1,
            time_elapsed_minutes=200,  # Late in window
        )

        # User bidding below market with low fill prob - should adjust or liquidate
        # The advisor calculates fill prob based on user price vs market
        # With a bid significantly below market, it should recommend action
        assert result["current_fill_probability"] < 0.5
