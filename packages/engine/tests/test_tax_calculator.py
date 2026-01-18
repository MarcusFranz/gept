"""Comprehensive test suite for GE tax calculations.

Tests all tax rules from the May 2025 update:
- Rate: 2% (updated from 1%)
- Rounding: Always rounds DOWN
- Floor: Items below 50gp have 0 tax
- Cap: Maximum 5,000,000gp per item
- Boundary behavior at 50gp threshold
"""

from src.tax_calculator import (
    TAX_CAP_GP,
    TAX_FLOOR_GP,
    TAX_RATE,
    calculate_flip_profit,
    calculate_net_proceeds,
    calculate_tax,
    effective_tax_rate,
    get_tax_info,
)


class TestTaxConstants:
    """Test tax constants are set correctly."""

    def test_tax_rate(self):
        """Tax rate should be 2% as of May 2025."""
        assert TAX_RATE == 0.02

    def test_tax_floor(self):
        """Tax floor should be 50gp."""
        assert TAX_FLOOR_GP == 50

    def test_tax_cap(self):
        """Tax cap should be 5,000,000gp per item."""
        assert TAX_CAP_GP == 5_000_000


class TestCalculateTax:
    """Test calculate_tax function with all edge cases."""

    # Sub-50gp: no tax
    def test_sub_floor_no_tax(self):
        """Items sold below 50gp should have 0 tax."""
        assert calculate_tax(sell_price=6, qty=10000) == 0
        assert calculate_tax(sell_price=49, qty=1) == 0
        assert calculate_tax(sell_price=1, qty=1) == 0
        assert calculate_tax(sell_price=25, qty=100) == 0

    # Boundary: 50gp pays 1gp tax
    def test_boundary_50gp(self):
        """Selling at exactly 50gp should pay 1gp tax."""
        assert calculate_tax(sell_price=50, qty=1) == 1

    def test_boundary_49_vs_50(self):
        """Boundary behavior: 49gp and 50gp should net seller same amount.

        At 49gp: 49 - 0 = 49 net
        At 50gp: 50 - 1 = 49 net
        """
        net_49 = calculate_net_proceeds(49, 1)
        net_50 = calculate_net_proceeds(50, 1)
        assert net_49 == net_50 == 49

    # Standard: 2% rounded down
    def test_standard_2_percent(self):
        """Standard tax calculation should be 2% rounded down."""
        assert calculate_tax(sell_price=100, qty=1) == 2  # 100 * 0.02 = 2
        assert calculate_tax(sell_price=149, qty=1) == 2  # 2.98 -> 2
        assert calculate_tax(sell_price=150, qty=1) == 3  # 3.0 -> 3
        assert calculate_tax(sell_price=200, qty=1) == 4  # 4.0 -> 4
        assert calculate_tax(sell_price=1000, qty=1) == 20  # 20.0 -> 20

    def test_rounding_down(self):
        """Tax should always round DOWN to nearest whole number."""
        # 2.98 rounds down to 2
        assert calculate_tax(sell_price=149, qty=1) == 2

        # 19.98 rounds down to 19
        assert calculate_tax(sell_price=999, qty=1) == 19

        # 99.98 rounds down to 99
        assert calculate_tax(sell_price=4999, qty=1) == 99

    # High value: cap at 5M per item
    def test_high_value_cap(self):
        """Tax should cap at 5,000,000gp per item."""
        assert calculate_tax(sell_price=300_000_000, qty=1) == 5_000_000
        assert calculate_tax(sell_price=1_000_000_000, qty=1) == 5_000_000

        # Max before cap: 250,000,000 * 0.02 = 5,000,000
        assert calculate_tax(sell_price=250_000_000, qty=1) == 5_000_000

        # Just over cap threshold
        assert calculate_tax(sell_price=250_000_001, qty=1) == 5_000_000

    def test_cap_threshold(self):
        """Verify exact price where cap kicks in.

        Cap applies at: price * 0.02 >= 5,000,000
        price >= 250,000,000
        """
        # Just below cap threshold
        price_below = 249_999_999
        tax_below = int(price_below * TAX_RATE)
        assert calculate_tax(price_below, 1) == tax_below
        assert tax_below < TAX_CAP_GP

        # At cap threshold
        price_at = 250_000_000
        assert calculate_tax(price_at, 1) == TAX_CAP_GP

        # Above cap threshold
        price_above = 250_000_001
        assert calculate_tax(price_above, 1) == TAX_CAP_GP

    # Bulk: tax calculated per item, not on total
    def test_bulk_per_item_calculation(self):
        """Tax should be calculated per item, then multiplied."""
        # 1000gp item = 20gp tax each
        # 100 items = 2000gp total tax
        assert calculate_tax(sell_price=1000, qty=100) == 2000

        # Verify per-item calculation
        single_tax = calculate_tax(sell_price=1000, qty=1)
        bulk_tax = calculate_tax(sell_price=1000, qty=100)
        assert bulk_tax == single_tax * 100

    def test_bulk_with_cap(self):
        """Cap applies per item, not per transaction."""
        # Each item hits 5M cap, selling 10 items
        assert calculate_tax(sell_price=300_000_000, qty=10) == 50_000_000

        # 10 items * 5M cap each = 50M total
        single_capped = calculate_tax(sell_price=300_000_000, qty=1)
        bulk_capped = calculate_tax(sell_price=300_000_000, qty=10)
        assert bulk_capped == single_capped * 10 == 50_000_000

    def test_bulk_below_floor(self):
        """Bulk sales below floor should still have 0 tax."""
        assert calculate_tax(sell_price=10, qty=1000) == 0


class TestNetProceeds:
    """Test net proceeds calculation."""

    def test_net_proceeds_standard(self):
        """Net proceeds should be gross - tax."""
        # 100gp item: 100 - 2 = 98
        assert calculate_net_proceeds(100, 1) == 98

        # 1000gp item * 10: 10000 - 200 = 9800
        assert calculate_net_proceeds(1000, 10) == 9800

    def test_net_proceeds_below_floor(self):
        """Net proceeds should equal gross when below floor."""
        assert calculate_net_proceeds(49, 1) == 49
        assert calculate_net_proceeds(25, 100) == 2500

    def test_net_proceeds_at_boundary(self):
        """Net proceeds at 50gp boundary."""
        # 50gp - 1gp tax = 49gp net
        assert calculate_net_proceeds(50, 1) == 49

    def test_net_proceeds_with_cap(self):
        """Net proceeds with capped tax."""
        # 300M item: 300M - 5M = 295M
        assert calculate_net_proceeds(300_000_000, 1) == 295_000_000


class TestFlipProfit:
    """Test flip profit calculation."""

    def test_simple_flip(self):
        """Basic flip profit calculation."""
        # Buy at 98, sell at 102: 102 - 2 tax - 98 cost = 2 profit
        assert calculate_flip_profit(98, 102, 1) == 2

        # Buy at 100, sell at 105: (105 - 2 tax) - 100 = 3 profit
        assert calculate_flip_profit(100, 105, 1) == 3

    def test_bulk_flip(self):
        """Bulk flip profit calculation."""
        # Buy 10 at 100 (1000 cost), sell 10 at 105 (1050 - 20 tax = 1030 proceeds)
        # Profit: 1030 - 1000 = 30
        assert calculate_flip_profit(100, 105, 10) == 30

    def test_losing_flip(self):
        """Flip can result in negative profit."""
        # Buy at 100, sell at 101: (101 - 2 tax = 99) - 100 = -1 loss
        assert calculate_flip_profit(100, 101, 1) == -1

    def test_flip_below_floor(self):
        """Flip below tax floor has no tax."""
        # Buy at 40, sell at 45: 45 - 40 = 5 profit (no tax)
        assert calculate_flip_profit(40, 45, 1) == 5

    def test_flip_across_boundary(self):
        """Flip crossing the 50gp boundary."""
        # Buy at 48, sell at 52: (52 - 1 tax = 51) - 48 = 3 profit
        assert calculate_flip_profit(48, 52, 1) == 3


class TestEffectiveTaxRate:
    """Test effective tax rate calculation."""

    def test_effective_rate_exact(self):
        """Effective rate when tax divides evenly."""
        # 100gp: 2gp tax = 2.0% effective
        assert effective_tax_rate(100) == 0.02

        # 1000gp: 20gp tax = 2.0% effective
        assert effective_tax_rate(1000) == 0.02

    def test_effective_rate_with_rounding(self):
        """Effective rate is lower when rounding down occurs."""
        # 149gp: 2gp tax = 1.34% effective (not 2%)
        rate_149 = effective_tax_rate(149)
        assert rate_149 < 0.02
        assert abs(rate_149 - 0.0134) < 0.0001

        # 151gp: 3gp tax = 1.99% effective
        rate_151 = effective_tax_rate(151)
        assert rate_151 < 0.02
        assert abs(rate_151 - 0.0199) < 0.0001

    def test_effective_rate_below_floor(self):
        """Effective rate is 0% below floor."""
        assert effective_tax_rate(49) == 0.0
        assert effective_tax_rate(1) == 0.0

    def test_effective_rate_at_cap(self):
        """Effective rate decreases as price increases past cap."""
        # At cap: 250M gp: 5M tax = 2.0%
        rate_at_cap = effective_tax_rate(250_000_000)
        assert abs(rate_at_cap - 0.02) < 0.0001

        # Above cap: 500M gp: 5M tax = 1.0%
        rate_above_cap = effective_tax_rate(500_000_000)
        assert abs(rate_above_cap - 0.01) < 0.0001

        # Well above cap: 1B gp: 5M tax = 0.5%
        rate_high = effective_tax_rate(1_000_000_000)
        assert abs(rate_high - 0.005) < 0.0001


class TestTaxInfo:
    """Test tax info metadata."""

    def test_get_tax_info(self):
        """Tax info should return current configuration."""
        info = get_tax_info()

        assert info["rate"] == 0.02
        assert info["floor_gp"] == 50
        assert info["cap_gp"] == 5_000_000
        assert info["effective_date"] == "2025-05-29"
        assert "2%" in info["notes"]


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_price(self):
        """Zero price should have zero tax."""
        assert calculate_tax(0, 1) == 0
        assert calculate_tax(0, 100) == 0

    def test_zero_quantity(self):
        """Zero quantity should have zero tax."""
        assert calculate_tax(1000, 0) == 0

    def test_negative_values_protection(self):
        """Function should handle negative inputs gracefully."""
        # Negative price is below floor
        assert calculate_tax(-100, 1) == 0

    def test_large_quantity(self):
        """Very large quantities should work correctly."""
        # 100gp * 1M qty = 100M gross, 2M tax
        assert calculate_tax(100, 1_000_000) == 2_000_000

    def test_exact_multiples(self):
        """Test prices that are exact multiples of 50."""
        assert calculate_tax(50, 1) == 1  # 50 * 0.02 = 1
        assert calculate_tax(100, 1) == 2  # 100 * 0.02 = 2
        assert calculate_tax(150, 1) == 3  # 150 * 0.02 = 3
        assert calculate_tax(200, 1) == 4  # 200 * 0.02 = 4


class TestRealWorldExamples:
    """Test with real OSRS item prices."""

    def test_cheap_item_fire_rune(self):
        """Fire rune at ~5gp (below floor)."""
        assert calculate_tax(5, 1) == 0
        assert calculate_tax(5, 10000) == 0

    def test_mid_item_bones(self):
        """Dragon bones at ~2000gp."""
        # 2000 * 0.02 = 40gp tax
        assert calculate_tax(2000, 1) == 40
        assert calculate_tax(2000, 100) == 4000

    def test_expensive_item_twisted_bow(self):
        """Twisted bow at ~1.1B gp (hits cap)."""
        # Would be 22M tax, but capped at 5M
        assert calculate_tax(1_100_000_000, 1) == 5_000_000

    def test_high_volume_flip(self):
        """High volume flip on medium-priced item."""
        # Cannon balls: buy 5000 at 180gp, sell at 185gp
        # Cost: 900,000
        # Proceeds: 185 * 5000 = 925,000 - (3 * 5000) = 910,000
        # Profit: 10,000
        profit = calculate_flip_profit(180, 185, 5000)
        assert profit == 10_000
