"""
Tests for Target Variable Engineering
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np

from target_engine import (
    TargetEngine,
    TargetConfig,
    TargetValidator,
    compute_expected_value,
    compute_ge_tax,
    compute_expected_value_gp,
    compute_expected_value_pct,
)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 2000

    # Simulate price movements
    base_price = 1000
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'avg_high_price': prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        'avg_low_price': prices * (1 - np.abs(np.random.randn(n) * 0.005)),
    })

    return df


class TestTargetEngine:
    """Tests for TargetEngine class."""

    def test_init_default_config(self):
        """Test default configuration."""
        engine = TargetEngine()
        assert len(engine.config.offsets) > 0
        assert len(engine.config.windows_hours) > 0
        assert engine.config.tax_rate == 0.02

    def test_init_custom_config(self):
        """Test custom configuration."""
        config = TargetConfig(
            offsets=[0.01, 0.02],
            windows_hours=[12, 24],
            tax_rate=0.01
        )
        engine = TargetEngine(config=config)
        assert engine.config.offsets == [0.01, 0.02]
        assert engine.config.tax_rate == 0.01

    def test_compute_targets_shape(self, sample_price_data):
        """Test that targets are added correctly."""
        engine = TargetEngine()
        result = engine.compute_targets(sample_price_data)

        # Should have more columns
        assert len(result.columns) > len(sample_price_data.columns)

        # Should have same number of rows
        assert len(result) == len(sample_price_data)

    def test_fill_targets_are_binary(self, sample_price_data):
        """Test that fill targets are 0/1."""
        engine = TargetEngine()
        result = engine.compute_targets(sample_price_data)

        for col in result.columns:
            if 'fills' in col or 'roundtrip' in col:
                # Drop NaN (end of series)
                values = result[col].dropna().unique()
                assert set(values).issubset({0, 1}), f"{col} has non-binary values"

    def test_roundtrip_requires_both_fills(self, sample_price_data):
        """Test that roundtrip is AND of buy and sell fills."""
        config = TargetConfig(offsets=[0.02], windows_hours=[24])
        engine = TargetEngine(config=config)
        result = engine.compute_targets(sample_price_data)

        buy_col = 'buy_fills_2pct_24h'
        sell_col = 'sell_fills_2pct_24h'
        roundtrip_col = 'roundtrip_2pct_24h'

        # Where both exist
        mask = result[[buy_col, sell_col, roundtrip_col]].notna().all(axis=1)

        # Roundtrip should be 1 only when both buy and sell are 1
        expected = (result.loc[mask, buy_col] & result.loc[mask, sell_col]).astype(int)
        actual = result.loc[mask, roundtrip_col].astype(int)

        assert (expected == actual).all()

    def test_longer_windows_have_higher_fill_rates(self, sample_price_data):
        """Test monotonicity: longer windows should fill more often."""
        config = TargetConfig(offsets=[0.02], windows_hours=[12, 24, 48])
        engine = TargetEngine(config=config)
        result = engine.compute_targets(sample_price_data)

        rates = engine.analyze_target_rates(result)

        rate_12h = rates.get('roundtrip_2pct_12h', 0)
        rate_24h = rates.get('roundtrip_2pct_24h', 0)
        rate_48h = rates.get('roundtrip_2pct_48h', 0)

        # Allow some tolerance for randomness
        assert rate_24h >= rate_12h * 0.8, f"24h rate {rate_24h} < 12h rate {rate_12h}"
        assert rate_48h >= rate_24h * 0.8, f"48h rate {rate_48h} < 24h rate {rate_24h}"

    def test_larger_offsets_have_lower_fill_rates(self, sample_price_data):
        """Test monotonicity: larger offsets should fill less often."""
        config = TargetConfig(offsets=[0.01, 0.02, 0.03], windows_hours=[24])
        engine = TargetEngine(config=config)
        result = engine.compute_targets(sample_price_data)

        rates = engine.analyze_target_rates(result)

        rate_1pct = rates.get('roundtrip_1pct_24h', 1)
        rate_2pct = rates.get('roundtrip_2pct_24h', 1)
        rate_3pct = rates.get('roundtrip_3pct_24h', 1)

        # Allow some tolerance
        assert rate_2pct <= rate_1pct * 1.1, f"2% rate {rate_2pct} > 1% rate {rate_1pct}"
        assert rate_3pct <= rate_2pct * 1.1, f"3% rate {rate_3pct} > 2% rate {rate_2pct}"


class TestExpectedValue:
    """Tests for expected value calculations."""

    def test_ev_positive_for_profitable_offset(self):
        """Test EV is positive when offset exceeds tax."""
        # 2% offset with 2% tax = 2% net profit
        # 50% fill rate should give 1% EV
        ev = compute_expected_value(0.5, 0.02, 0.02)
        assert 0.009 < ev < 0.011  # ~1%

    def test_ev_zero_for_break_even(self):
        """Test EV is zero when offset equals half tax."""
        # 1% offset with 2% tax = 0% net profit
        ev = compute_expected_value(0.5, 0.01, 0.02)
        assert abs(ev) < 0.001

    def test_ev_scales_with_probability(self):
        """Test EV scales linearly with fill probability."""
        ev_25 = compute_expected_value(0.25, 0.02, 0.02)
        ev_50 = compute_expected_value(0.50, 0.02, 0.02)

        assert abs(ev_50 - 2 * ev_25) < 0.001

    def test_ev_with_prices_uses_accurate_tax(self):
        """Test EV with prices uses GP-based tax calculation."""
        # 1000 GP item with 2% offset
        # Buy at 980, sell at 1020
        # Tax = ceil(1020 * 0.02) = 21 GP
        # Net profit = 40 - 21 = 19 GP = 1.94% of 980
        ev = compute_expected_value(
            1.0, 0.02, current_high=1000, current_low=1000
        )
        # 19 / 980 = 0.0194
        assert 0.019 < ev < 0.020


class TestGETax:
    """Tests for OSRS GE tax calculation."""

    def test_tax_minimum_1gp(self):
        """Items under 50 GP should still have 1 GP minimum tax."""
        assert compute_ge_tax(10) == 1  # 10 * 0.02 = 0.2, rounds to 1 (min)
        assert compute_ge_tax(49) == 1  # 49 * 0.02 = 0.98, rounds to 1 (min)

    def test_tax_rounds_up(self):
        """Tax should round up to nearest GP."""
        assert compute_ge_tax(51) == 2   # 51 * 0.02 = 1.02 -> ceil = 2
        assert compute_ge_tax(100) == 2  # 100 * 0.02 = 2.0 -> ceil = 2
        assert compute_ge_tax(101) == 3  # 101 * 0.02 = 2.02 -> ceil = 3

    def test_tax_normal_items(self):
        """Regular items should have 2% tax rounded up."""
        assert compute_ge_tax(1000) == 20     # 1000 * 0.02 = 20
        assert compute_ge_tax(10000) == 200   # 10000 * 0.02 = 200
        assert compute_ge_tax(123456) == 2470  # 123456 * 0.02 = 2469.12 -> 2470

    def test_tax_maximum_5m(self):
        """Items >= 250M GP should hit 5M cap."""
        assert compute_ge_tax(250_000_000) == 5_000_000  # Exactly at cap
        assert compute_ge_tax(500_000_000) == 5_000_000  # Above cap
        assert compute_ge_tax(1_000_000_000) == 5_000_000  # Well above cap

    def test_tax_just_under_cap(self):
        """Items just under 250M should not hit cap."""
        assert compute_ge_tax(249_999_999) == 5_000_000  # 4,999,999.98 -> 5M
        assert compute_ge_tax(200_000_000) == 4_000_000  # 200M * 2% = 4M

    def test_tax_zero_price(self):
        """Zero or negative price should return 0 tax."""
        assert compute_ge_tax(0) == 0
        assert compute_ge_tax(-100) == 0

    def test_tax_nan_price(self):
        """NaN price should return 0 tax without raising ValueError."""
        assert compute_ge_tax(float('nan')) == 0


class TestNaNHandling:
    """Tests for NaN value handling in price calculations (Issue #114)."""

    def test_ge_tax_nan(self):
        """compute_ge_tax should return 0 for NaN price."""
        assert compute_ge_tax(float('nan')) == 0

    def test_ev_gp_nan_buy_price(self):
        """compute_expected_value_gp should return 0 for NaN buy price."""
        assert compute_expected_value_gp(0.5, float('nan'), 1000) == 0

    def test_ev_gp_nan_sell_price(self):
        """compute_expected_value_gp should return 0 for NaN sell price."""
        assert compute_expected_value_gp(0.5, 1000, float('nan')) == 0

    def test_ev_gp_both_nan(self):
        """compute_expected_value_gp should return 0 when both prices are NaN."""
        assert compute_expected_value_gp(0.5, float('nan'), float('nan')) == 0

    def test_ev_pct_nan_buy_price(self):
        """compute_expected_value_pct should return 0 for NaN buy price."""
        assert compute_expected_value_pct(0.5, float('nan'), 1000) == 0

    def test_ev_pct_nan_sell_price(self):
        """compute_expected_value_pct should return 0 for NaN sell price."""
        # NaN sell price propagates through compute_expected_value_gp
        assert compute_expected_value_pct(0.5, 1000, float('nan')) == 0

    def test_ev_pct_both_nan(self):
        """compute_expected_value_pct should return 0 when both prices are NaN."""
        assert compute_expected_value_pct(0.5, float('nan'), float('nan')) == 0


class TestPriceAwareEV:
    """Tests for price-aware expected value calculation."""

    def test_ev_gp_basic(self):
        """Test basic GP-based EV calculation."""
        # Buy at 980, sell at 1020
        # Gross profit: 40 GP
        # Tax: ceil(1020 * 0.02) = 21 GP
        # Net profit: 19 GP
        # With 50% fill prob: EV = 9.5 GP
        ev = compute_expected_value_gp(0.5, 980, 1020)
        assert abs(ev - 9.5) < 0.01

    def test_ev_pct_basic(self):
        """Test percentage EV calculation."""
        ev_pct = compute_expected_value_pct(0.5, 980, 1020)
        # 9.5 / 980 = 0.00969 ~ 0.97%
        assert abs(ev_pct - 0.00969) < 0.001

    def test_ev_low_price_item(self):
        """Test EV for low-price items where min tax matters."""
        # Item at ~50 GP, 2% offset
        # Buy at 49, sell at 51
        # Gross: 2 GP, Tax: 2 GP (ceil(51*0.02)=1.02 -> 2)
        # Net: 0 GP - break even!
        ev = compute_expected_value_gp(1.0, 49, 51)
        assert ev == 0

    def test_ev_high_value_item(self):
        """Test EV for high-value items where 5M cap applies."""
        # Item at 300M GP, 2% offset
        # Buy at 294M, sell at 306M
        # Gross: 12M GP, Tax: 5M (capped)
        # Net: 7M GP
        ev = compute_expected_value_gp(1.0, 294_000_000, 306_000_000)
        assert ev == 7_000_000

    def test_ev_scales_with_probability(self):
        """Test that EV scales linearly with fill probability."""
        ev_25 = compute_expected_value_gp(0.25, 980, 1020)
        ev_50 = compute_expected_value_gp(0.50, 980, 1020)
        ev_100 = compute_expected_value_gp(1.00, 980, 1020)

        assert abs(ev_50 - 2 * ev_25) < 0.01
        assert abs(ev_100 - 4 * ev_25) < 0.01

    def test_ev_zero_on_invalid_prices(self):
        """Test graceful handling of invalid prices."""
        assert compute_expected_value_gp(0.5, 0, 1000) == 0
        assert compute_expected_value_gp(0.5, 1000, 0) == 0
        assert compute_expected_value_pct(0.5, 0, 1000) == 0


class TestTargetValidator:
    """Tests for target validation."""

    def test_validates_monotonicity(self, sample_price_data):
        """Test that validator checks monotonicity."""
        engine = TargetEngine()
        result = engine.compute_targets(sample_price_data)

        validator = TargetValidator()
        validation = validator.validate_targets(result, engine)

        # Should have target rates
        assert len(validation['target_rates']) > 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
