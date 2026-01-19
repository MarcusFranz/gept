"""Tests for price stability and momentum field computation."""

import sys
import os

# Set dummy DB environment variables before importing modules that need them
# These are only used to satisfy the module import; no actual DB connection is made in tests
os.environ.setdefault('DB_PASS', 'test')
os.environ.setdefault('DB_HOST', 'localhost')
os.environ.setdefault('DB_PORT', '5432')
os.environ.setdefault('DB_NAME', 'test')
os.environ.setdefault('DB_USER', 'test')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from batch_predictor_multitarget import compute_stability_fields


@pytest.fixture
def sample_price_history():
    """Create 14 days of hourly price data."""
    hours = 14 * 24  # 14 days
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]

    # Stable item: prices hover around 1000 with small noise
    np.random.seed(42)
    base_price = 1000
    noise = np.random.normal(0, 10, hours)  # ~1% noise
    highs = base_price + noise + 5
    lows = base_price + noise - 5

    return pd.DataFrame({
        'timestamp': timestamps,
        'avg_high_price': highs,
        'avg_low_price': lows,
    }).set_index('timestamp').sort_index()


@pytest.fixture
def falling_price_history():
    """Create price data with clear downtrend."""
    hours = 14 * 24
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours, 0, -1)]

    # Item falling 5% over last 4 hours
    base_prices = np.linspace(1000, 950, hours)  # gradual decline
    # Make last 4 hours steeper
    base_prices[-4:] = np.linspace(970, 920, 4)

    return pd.DataFrame({
        'timestamp': timestamps,
        'avg_high_price': base_prices + 5,
        'avg_low_price': base_prices - 5,
    }).set_index('timestamp')


class TestComputeStabilityFields:
    """Test stability field computation."""

    def test_returns_all_required_fields(self, sample_price_history):
        """Should return dict with all 6 required fields."""
        result = compute_stability_fields(sample_price_history)

        required_fields = [
            'median_14d', 'price_vs_median_ratio',
            'return_1h', 'return_4h', 'return_24h', 'volatility_24h'
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_median_14d_is_reasonable(self, sample_price_history):
        """Median should be close to the mean for stable prices."""
        result = compute_stability_fields(sample_price_history)

        # For our sample data centered around 1000
        assert 950 < result['median_14d'] < 1050

    def test_price_vs_median_ratio_near_one_for_stable(self, sample_price_history):
        """Ratio should be ~1.0 for stable prices."""
        result = compute_stability_fields(sample_price_history)

        assert 0.95 < result['price_vs_median_ratio'] < 1.05

    def test_return_4h_negative_for_falling(self, falling_price_history):
        """4h return should be negative for falling prices."""
        result = compute_stability_fields(falling_price_history)

        assert result['return_4h'] < -0.02  # At least -2%

    def test_return_1h_negative_for_falling(self, falling_price_history):
        """1h return should be negative for active decline."""
        result = compute_stability_fields(falling_price_history)

        assert result['return_1h'] < 0

    def test_volatility_24h_positive(self, sample_price_history):
        """Volatility should always be positive."""
        result = compute_stability_fields(sample_price_history)

        assert result['volatility_24h'] > 0

    def test_handles_missing_data_gracefully(self):
        """Should return None values if insufficient history."""
        # Only 2 hours of data
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(2, 0, -1)]
        short_history = pd.DataFrame({
            'timestamp': timestamps,
            'avg_high_price': [100, 101],
            'avg_low_price': [99, 100],
        }).set_index('timestamp')

        result = compute_stability_fields(short_history)

        # Should still return dict, but some fields may be None
        assert isinstance(result, dict)
        # median_14d requires 14 days, so should be None
        assert result['median_14d'] is None


class TestStabilityFieldsEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe_returns_none_fields(self):
        """Empty input should return dict with all None values."""
        empty_df = pd.DataFrame(columns=['avg_high_price', 'avg_low_price'])
        result = compute_stability_fields(empty_df)

        assert all(v is None for v in result.values())

    def test_nan_prices_handled(self, sample_price_history):
        """NaN prices should be handled without crashing."""
        # Inject some NaNs
        sample_price_history.iloc[10:15, 0] = np.nan

        result = compute_stability_fields(sample_price_history)

        # Should still compute (using available data)
        assert result['median_14d'] is not None


class TestInferenceIntegration:
    """Test stability fields are included in predictions output."""

    def test_stability_fields_added_to_predictions(self, sample_price_history):
        """Stability fields should be computable and addable to predictions."""
        # Compute stability fields (as would be done in run_inference_cycle)
        stability = compute_stability_fields(sample_price_history)

        # Create a sample prediction dict (as would be generated by predict_item)
        prediction = {
            'item_id': 1,
            'item_name': 'Test Item',
            'hour_offset': 4,
            'offset_pct': 0.02,
            'fill_probability': 0.5,
            'expected_value': 0.01,
            'buy_price': 100,
            'sell_price': 104,
            'current_high': 102,
            'current_low': 98,
            'confidence': 'medium',
        }

        # Add stability fields to prediction (as done in run_inference_cycle)
        prediction['median_14d'] = stability.get('median_14d')
        prediction['price_vs_median_ratio'] = stability.get('price_vs_median_ratio')
        prediction['return_1h'] = stability.get('return_1h')
        prediction['return_4h'] = stability.get('return_4h')
        prediction['return_24h'] = stability.get('return_24h')
        prediction['volatility_24h'] = stability.get('volatility_24h')

        # Verify expected columns exist
        expected_new_cols = [
            'median_14d', 'price_vs_median_ratio',
            'return_1h', 'return_4h', 'return_24h', 'volatility_24h'
        ]
        for col in expected_new_cols:
            assert col in prediction, f"Missing field: {col}"

        # Verify values are properly computed (not all None)
        assert prediction['median_14d'] is not None
        assert prediction['price_vs_median_ratio'] is not None
        assert prediction['volatility_24h'] is not None

    def test_stability_cache_pattern(self, sample_price_history, falling_price_history):
        """Test the stability cache pattern used in run_inference_cycle."""
        # Simulate price_data dict as used in run_inference_cycle
        price_data = {
            1: sample_price_history.reset_index(),
            2: falling_price_history.reset_index()
        }

        # Build stability cache (as done in run_inference_cycle)
        stability_cache = {}
        for item_id, df in price_data.items():
            # Need to set index back for compute_stability_fields
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            stability_cache[item_id] = compute_stability_fields(df)

        # Verify cache structure
        assert 1 in stability_cache
        assert 2 in stability_cache

        # Stable item should have ratio close to 1
        assert 0.95 < stability_cache[1]['price_vs_median_ratio'] < 1.05

        # Falling item should have negative returns
        assert stability_cache[2]['return_4h'] < 0
