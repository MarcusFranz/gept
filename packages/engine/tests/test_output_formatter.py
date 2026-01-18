"""Tests for the output formatter."""

from src.output_formatter import OutputFormatter


class TestOutputFormatter:
    """Test cases for OutputFormatter."""

    def test_init_defaults(self):
        """Test default initialization."""
        formatter = OutputFormatter()

        assert formatter.ev_threshold == 0.001
        assert formatter.confidence_high_auc == 0.75
        assert formatter.confidence_medium_auc == 0.60
        assert formatter.data_stale_seconds == 600

    def test_calculate_tax_sub_50gp(self):
        """Test tax calculation for items below 50gp (tax-exempt)."""
        formatter = OutputFormatter()

        # Iron arrows at 6gp - should be tax-exempt
        tax = formatter.calculate_tax(6, 10000)
        assert tax == 0

        # Fire rune at 5gp - should be tax-exempt
        tax = formatter.calculate_tax(5, 1000)
        assert tax == 0

        # Exactly at threshold - should be taxed
        tax = formatter.calculate_tax(50, 100)
        assert tax == 100  # 50 * 100 * 0.02 = 100

    def test_calculate_tax_above_threshold(self):
        """Test tax calculation for items at or above 50gp."""
        formatter = OutputFormatter()

        # 100gp item, quantity 1000
        # Tax = 100 * 1000 * 0.02 = 2000
        tax = formatter.calculate_tax(100, 1000)
        assert tax == 2000

        # 1000gp item, quantity 100
        # Tax = 1000 * 100 * 0.02 = 2000
        tax = formatter.calculate_tax(1000, 100)
        assert tax == 2000

    def test_calculate_tax_edge_cases(self):
        """Test tax calculation edge cases."""
        formatter = OutputFormatter()

        # 49gp (just below threshold)
        tax = formatter.calculate_tax(49, 10000)
        assert tax == 0

        # 51gp (just above threshold)
        tax = formatter.calculate_tax(51, 10000)
        assert tax == 10200  # 51 * 10000 * 0.02

    def test_calculate_expected_value(self):
        """Test expected value calculation."""
        formatter = OutputFormatter()

        # EV = P(fill) * (2*offset - 0.02)
        # With 10% fill rate and 2% offset: 0.1 * (0.04 - 0.02) = 0.002
        ev = formatter.calculate_expected_value(0.1, 0.02)
        assert abs(ev - 0.002) < 0.0001

        # With 50% fill rate and 1.5% offset: 0.5 * (0.03 - 0.02) = 0.005
        ev = formatter.calculate_expected_value(0.5, 0.015)
        assert abs(ev - 0.005) < 0.0001

        # With 0% fill rate
        ev = formatter.calculate_expected_value(0, 0.02)
        assert ev == 0

    def test_calculate_suggested_prices(self):
        """Test suggested price calculation."""
        formatter = OutputFormatter()

        # With 2% offset
        buy, sell = formatter.calculate_suggested_prices(100, 110, 0.02)
        assert buy == 98  # 100 * 0.98
        assert sell == 112  # 110 * 1.02

        # With 1.5% offset
        buy, sell = formatter.calculate_suggested_prices(1000, 1100, 0.015)
        assert buy == 985  # 1000 * 0.985
        assert sell == 1116  # 1100 * 1.015

    def test_calculate_suggested_prices_minimum(self):
        """Test minimum price handling."""
        formatter = OutputFormatter()

        # Very low prices
        buy, sell = formatter.calculate_suggested_prices(1, 2, 0.5)
        assert buy >= 1
        assert sell > buy

    def test_calculate_suggested_prices_50gp_boundary_optimization(self):
        """Test 50gp boundary optimization (tax rounding exploit)."""
        formatter = OutputFormatter()

        # Test exact multiples of 50 get reduced by 1
        buy, sell = formatter.calculate_suggested_prices(48, 50, 0.0)
        assert sell == 49  # 50 rounded down to 49

        buy, sell = formatter.calculate_suggested_prices(98, 100, 0.0)
        assert sell == 99  # 100 rounded down to 99

        buy, sell = formatter.calculate_suggested_prices(148, 150, 0.0)
        assert sell == 149  # 150 rounded down to 149

        buy, sell = formatter.calculate_suggested_prices(9998, 10000, 0.0)
        assert sell == 9999  # 10,000 rounded down to 9,999

    def test_calculate_suggested_prices_non_boundary_unchanged(self):
        """Test that non-multiples of 50 are not affected by optimization."""
        formatter = OutputFormatter()

        # Test non-multiples of 50 remain unchanged
        buy, sell = formatter.calculate_suggested_prices(48, 49, 0.0)
        assert sell == 49  # Already 49, no change

        buy, sell = formatter.calculate_suggested_prices(97, 99, 0.0)
        assert sell == 99  # Already 99, no change

        buy, sell = formatter.calculate_suggested_prices(100, 101, 0.0)
        assert sell == 101  # 101 is not a multiple of 50

        buy, sell = formatter.calculate_suggested_prices(150, 175, 0.0)
        assert sell == 175  # 175 is not a multiple of 50

    def test_calculate_suggested_prices_boundary_with_offset(self):
        """Test 50gp boundary optimization with various offset percentages."""
        formatter = OutputFormatter()

        # When offset calculation lands exactly on 50
        buy, sell = formatter.calculate_suggested_prices(49, 49, 0.02041)
        # 49 * 1.02041 ≈ 50 (rounds to 50), then optimized to 49
        assert sell == 49

        # When offset calculation lands exactly on 100
        buy, sell = formatter.calculate_suggested_prices(98, 98, 0.02041)
        # 98 * 1.02041 ≈ 100, then optimized to 99
        assert sell == 99

        # When offset calculation lands exactly on 150
        buy, sell = formatter.calculate_suggested_prices(147, 147, 0.02041)
        # 147 * 1.02041 ≈ 150, then optimized to 149
        assert sell == 149

    def test_calculate_suggested_prices_edge_case_price_1(self):
        """Test that price 1 is not reduced (edge case)."""
        formatter = OutputFormatter()

        # Edge case: suggested_sell = 1 should not be reduced
        # This shouldn't normally happen, but test the guard
        buy, sell = formatter.calculate_suggested_prices(1, 1, 0.0)
        assert sell >= 1  # Must stay >= 1

    def test_calculate_suggested_prices_boundary_maintains_order(self):
        """Test that boundary optimization doesn't violate buy < sell."""
        formatter = OutputFormatter()

        # Ensure buy price is always less than sell price after optimization
        buy, sell = formatter.calculate_suggested_prices(49, 50, 0.0)
        assert buy < sell
        # Optimization skipped because (50 - 1) == 49 == buy, would violate invariant
        assert sell == 50  # NOT optimized, maintains buy < sell

        # Edge case where buy might equal sell after optimization
        buy, sell = formatter.calculate_suggested_prices(50, 50, 0.0)
        assert buy < sell  # Still enforced
        # buy=50, sell forced to 51 by line 119, optimization skipped (50 not equal to 51)
        assert sell == 51

    def test_assign_confidence_high(self):
        """Test high confidence assignment."""
        formatter = OutputFormatter()

        confidence = formatter.assign_confidence(auc=0.85, data_age_seconds=60, tier=1)
        assert confidence == "high"

    def test_assign_confidence_medium(self):
        """Test medium confidence assignment."""
        formatter = OutputFormatter()

        # Medium AUC
        confidence = formatter.assign_confidence(auc=0.65, data_age_seconds=60, tier=1)
        assert confidence == "medium"

        # Tier 2
        confidence = formatter.assign_confidence(auc=0.80, data_age_seconds=60, tier=2)
        assert confidence == "medium"

    def test_assign_confidence_low(self):
        """Test low confidence assignment."""
        formatter = OutputFormatter()

        # Low AUC
        confidence = formatter.assign_confidence(auc=0.55, data_age_seconds=60, tier=1)
        assert confidence == "low"

        # Tier 3
        confidence = formatter.assign_confidence(auc=0.80, data_age_seconds=60, tier=3)
        assert confidence == "low"

    def test_assign_confidence_stale(self):
        """Test stale confidence assignment."""
        formatter = OutputFormatter()

        confidence = formatter.assign_confidence(auc=0.90, data_age_seconds=700, tier=1)
        assert confidence == "stale"

    def test_assign_confidence_unreliable(self):
        """Test unreliable confidence assignment."""
        formatter = OutputFormatter()

        # Very low AUC
        confidence = formatter.assign_confidence(auc=0.50, data_age_seconds=60, tier=1)
        assert confidence == "unreliable"

        # No AUC
        confidence = formatter.assign_confidence(auc=None, data_age_seconds=60, tier=1)
        assert confidence == "unreliable"

    def test_select_recommendation(self):
        """Test recommendation selection."""
        formatter = OutputFormatter()

        predictions = [
            {"expected_value": 0.002, "hour_window": 1},
            {"expected_value": 0.005, "hour_window": 4},
            {"expected_value": 0.003, "hour_window": 2},
        ]

        best = formatter.select_recommendation(predictions)

        assert best is not None
        assert best["expected_value"] == 0.005
        assert best["hour_window"] == 4

    def test_select_recommendation_empty(self):
        """Test recommendation selection with empty list."""
        formatter = OutputFormatter()
        best = formatter.select_recommendation([])
        assert best is None

    def test_select_recommendation_no_positive_ev(self):
        """Test recommendation selection with no positive EV."""
        formatter = OutputFormatter()

        predictions = [
            {"expected_value": -0.001, "hour_window": 1},
            {"expected_value": 0, "hour_window": 4},
        ]

        best = formatter.select_recommendation(predictions)
        assert best is None

    def test_determine_action_trade(self):
        """Test trade action determination."""
        formatter = OutputFormatter()

        best = {"expected_value": 0.005}
        action = formatter.determine_action(best, "high")
        assert action == "trade"

    def test_determine_action_hold(self):
        """Test hold action determination."""
        formatter = OutputFormatter()

        # No prediction
        action = formatter.determine_action(None, "high")
        assert action == "hold"

        # Low EV
        best = {"expected_value": 0.0005}
        action = formatter.determine_action(best, "high")
        assert action == "hold"

    def test_determine_action_avoid(self):
        """Test avoid action determination."""
        formatter = OutputFormatter()

        best = {"expected_value": 0.005}
        action = formatter.determine_action(best, "stale")
        assert action == "avoid"

        action = formatter.determine_action(best, "unreliable")
        assert action == "avoid"

    def test_format_prediction(self):
        """Test full prediction formatting."""
        formatter = OutputFormatter()

        result = formatter.format_prediction(
            item_id=554,
            item_name="Fire rune",
            tier=1,
            current_prices={
                "high": 5,
                "low": 4,
                "high_volume": 1000,
                "low_volume": 800,
            },
            predictions=[
                {"hour_window": 1, "offset_pct": 0.02, "fill_probability": 0.15},
                {"hour_window": 4, "offset_pct": 0.02, "fill_probability": 0.25},
            ],
            model_metadata={"avg_auc": 0.85},
            data_age_seconds=30,
        )

        assert result["item_id"] == 554
        assert result["item_name"] == "Fire rune"
        assert result["tier"] == 1
        assert "current_market" in result
        assert "predictions" in result
        assert "recommendation" in result
        assert "model_metadata" in result

        # Check predictions are formatted
        assert len(result["predictions"]) == 2
        assert "expected_value" in result["predictions"][0]
        assert "confidence" in result["predictions"][0]

    def test_format_for_discord_bot(self):
        """Test Discord bot format conversion."""
        formatter = OutputFormatter()

        prediction = {
            "item_id": 554,
            "item_name": "Fire rune",
            "current_market": {
                "high": 5,
                "low": 4,
                "volume_1h": 50000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 3,
                "suggested_sell": 6,
                "fill_probability": 0.15,
                "confidence": "high",
            },
        }

        result = formatter.format_for_discord_bot(prediction, capital=1000000)

        assert result is not None
        assert result["itemId"] == 554
        assert result["item"] == "Fire rune"
        assert result["buyPrice"] == 3
        assert result["sellPrice"] == 6
        assert result["confidence"] == "high"
        assert result["capitalRequired"] == result["buyPrice"] * result["quantity"]

    def test_format_for_discord_bot_insufficient_capital(self):
        """Test Discord format with insufficient capital."""
        formatter = OutputFormatter()

        prediction = {
            "item_id": 554,
            "item_name": "Fire rune",
            "current_market": {"volume_1h": 50000},
            "recommendation": {
                "action": "trade",
                "suggested_buy": 1000000,  # Very expensive
                "suggested_sell": 1100000,
                "fill_probability": 0.15,
                "confidence": "high",
            },
        }

        # Only 1000 GP available
        result = formatter.format_for_discord_bot(prediction, capital=1000)
        assert result is None

    def test_format_for_discord_bot_non_trade(self):
        """Test Discord format for non-trade recommendations."""
        formatter = OutputFormatter()

        prediction = {
            "item_id": 554,
            "item_name": "Fire rune",
            "current_market": {"volume_1h": 50000},
            "recommendation": {
                "action": "hold",
                "suggested_buy": None,
                "suggested_sell": None,
            },
        }

        result = formatter.format_for_discord_bot(prediction, capital=1000000)
        assert result is None

    def test_format_for_discord_bot_sub_50gp_no_tax(self):
        """Test Discord format profit calculation for sub-50gp items (no tax)."""
        formatter = OutputFormatter()

        # Iron arrows: buy @ 5gp, sell @ 6gp
        # With 100% fill probability for testing
        prediction = {
            "item_id": 884,
            "item_name": "Iron arrow",
            "current_market": {
                "high": 6,
                "low": 5,
                "volume_1h": 50000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 5,
                "suggested_sell": 6,
                "fill_probability": 1.0,  # 100% for testing
                "confidence": "high",
            },
        }

        result = formatter.format_for_discord_bot(prediction, capital=100000)

        assert result is not None
        assert result["buyPrice"] == 5
        assert result["sellPrice"] == 6
        # With 1,000 fallback quantity max (no buy_limit provided)
        assert result["quantity"] == 1000
        # Expected profit = (6 - 5) * 1000 * 1.0 = 1,000gp (no tax)
        assert result["expectedProfit"] == 1000

    def test_format_for_discord_bot_above_50gp_with_tax(self):
        """Test Discord format profit calculation for items above 50gp (with tax)."""
        formatter = OutputFormatter()

        # Item: buy @ 100gp, sell @ 110gp
        # Tax = 110 * 0.02 = 2.2gp per item
        # Profit per item = 110 - 100 - 2.2 = 7.8gp
        prediction = {
            "item_id": 555,
            "item_name": "Test item",
            "current_market": {
                "high": 110,
                "low": 100,
                "volume_1h": 50000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 100,
                "suggested_sell": 110,
                "fill_probability": 1.0,  # 100% for testing
                "confidence": "high",
            },
        }

        result = formatter.format_for_discord_bot(prediction, capital=1000000)

        assert result is not None
        # With capital of 1M and buy price 100, can afford 10,000 items
        # but capped at 1,000 fallback (no buy_limit provided)
        assert result["quantity"] == 1000
        # Revenue: 110 * 1000 = 110,000
        # Cost: 100 * 1000 = 100,000
        # Tax: 110 * 1000 * 0.02 = 2,200
        # Profit: 110,000 - 100,000 - 2,200 = 7,800
        assert result["expectedProfit"] == 7800

    def test_format_for_discord_bot_with_buy_limit(self):
        """Test Discord format respects buy_limit parameter."""
        formatter = OutputFormatter()

        prediction = {
            "item_id": 2503,
            "item_name": "Black d'hide body",
            "current_market": {
                "high": 8500,
                "low": 8000,
                "volume_1h": 5000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 8000,
                "suggested_sell": 8500,
                "fill_probability": 0.15,
                "confidence": "medium",
            },
        }

        # With 10M capital, could afford 1,250 at 8000gp each
        # But buy_limit is 70, so should cap at 70
        result = formatter.format_for_discord_bot(
            prediction, capital=10000000, buy_limit=70
        )

        assert result is not None
        assert result["quantity"] == 70  # Capped at GE buy limit
        assert result["capitalRequired"] == 8000 * 70

    def test_format_for_discord_bot_buy_limit_70_black_dhide(self):
        """Test black d'hide bodies scenario - buy limit of 70.

        This is the exact scenario from issue #59 where recommendations
        were suggesting ~7,000 quantity when the buy limit is only 70.
        """
        formatter = OutputFormatter()

        prediction = {
            "item_id": 2503,
            "item_name": "Black d'hide body",
            "current_market": {
                "high": 8500,
                "low": 8000,
                "volume_1h": 2000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 8000,
                "suggested_sell": 8500,
                "fill_probability": 0.12,
                "confidence": "medium",
            },
        }

        # With 100M capital, could afford 12,500 at 8000gp each
        # Before fix: would cap at 10,000 (or 1,000 with new fallback)
        # After fix: should cap at 70 (the actual GE buy limit)
        result = formatter.format_for_discord_bot(
            prediction, capital=100000000, buy_limit=70
        )

        assert result is not None
        assert result["quantity"] == 70  # Must respect GE buy limit
        assert result["capitalRequired"] == 8000 * 70  # 560,000gp

    def test_format_for_discord_bot_fallback_cap_1000(self):
        """Test that fallback cap is 1000 when no buy_limit provided."""
        formatter = OutputFormatter()

        prediction = {
            "item_id": 999,
            "item_name": "Unknown item",
            "current_market": {
                "high": 100,
                "low": 90,
                "volume_1h": 10000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 90,
                "suggested_sell": 100,
                "fill_probability": 0.20,
                "confidence": "medium",
            },
        }

        # With 1M capital at 90gp, could afford 11,111 items
        # Without buy_limit, should cap at conservative 1,000 fallback
        result = formatter.format_for_discord_bot(
            prediction, capital=1000000, buy_limit=None
        )

        assert result is not None
        assert result["quantity"] == 1000  # Fallback cap

    def test_format_for_discord_bot_capital_limits_before_buy_limit(self):
        """Test that capital-based limit is applied before buy_limit."""
        formatter = OutputFormatter()

        prediction = {
            "item_id": 554,
            "item_name": "Fire rune",
            "current_market": {
                "high": 6,
                "low": 5,
                "volume_1h": 100000,
            },
            "recommendation": {
                "action": "trade",
                "suggested_buy": 5,
                "suggested_sell": 6,
                "fill_probability": 0.25,
                "confidence": "high",
            },
        }

        # With only 250gp capital at 5gp each, can only afford 50 items
        # Even though buy_limit is 25,000, capital should limit us to 50
        result = formatter.format_for_discord_bot(
            prediction, capital=250, buy_limit=25000
        )

        assert result is not None
        assert result["quantity"] == 50  # Capital-limited, not buy_limit-limited
