"""
Tests for Feature Engineering Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_engine import FeatureEngine, Granularity, FeatureConfig, FeatureValidator


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n = 1000

    # Simulate realistic price movements
    base_price = 100
    returns = np.random.randn(n) * 0.002
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'avg_high_price': prices * (1 + np.abs(np.random.randn(n) * 0.003)),
        'avg_low_price': prices * (1 - np.abs(np.random.randn(n) * 0.003)),
        'high_price_volume': np.random.randint(100, 1000, n),
        'low_price_volume': np.random.randint(100, 1000, n),
    })

    # Ensure low < high
    df['avg_low_price'] = df[['avg_high_price', 'avg_low_price']].min(axis=1)

    return df


class TestFeatureEngine:
    """Tests for FeatureEngine class."""

    def test_init_5min(self):
        """Test initialization with 5-minute granularity."""
        engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        assert engine.granularity == Granularity.FIVE_MIN
        # 24 hours = 288 periods at 5min
        assert engine.ma_windows[24] == 288

    def test_init_1min(self):
        """Test initialization with 1-minute granularity."""
        engine = FeatureEngine(granularity=Granularity.ONE_MIN)
        assert engine.granularity == Granularity.ONE_MIN
        # 24 hours = 1440 periods at 1min
        assert engine.ma_windows[24] == 1440

    def test_compute_features_output_shape(self, sample_price_data):
        """Test that feature computation adds columns."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)

        # Should have more columns than input
        assert len(result.columns) > len(sample_price_data.columns)

        # Should have same number of rows
        assert len(result) == len(sample_price_data)

    def test_compute_features_no_nan_explosion(self, sample_price_data):
        """Test that features don't have excessive NaN values."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)

        # After warmup, should have minimal NaN
        result_trimmed = result.iloc[300:]

        for col in engine.get_feature_columns():
            if col in result_trimmed.columns:
                nan_ratio = result_trimmed[col].isna().mean()
                assert nan_ratio < 0.1, f"Column {col} has {nan_ratio:.1%} NaN values"

    def test_spread_features(self, sample_price_data):
        """Test spread-related features are computed correctly."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)

        # Spread should always be positive
        assert (result['spread'] >= 0).all()

        # Spread percentage should be reasonable (< 100%)
        assert (result['spread_pct'].dropna() < 1).all()

    def test_ma_features(self, sample_price_data):
        """Test moving average features."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)

        # MA ratio should be close to 1 on average
        ma_ratio_cols = [c for c in result.columns if 'ma_ratio' in c]
        for col in ma_ratio_cols:
            mean_ratio = result[col].dropna().mean()
            assert 0.9 < mean_ratio < 1.1, f"{col} mean ratio {mean_ratio} too far from 1"

    def test_time_features(self, sample_price_data):
        """Test time-based features."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)

        # Hour should be 0-23
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23

        # Day of week should be 0-6
        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6

        # Cyclical encoding should be -1 to 1
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1

    def test_volume_features(self, sample_price_data):
        """Test volume features."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)

        # Volume should be non-negative
        assert (result['volume'] >= 0).all()

        # Log volume should be finite
        assert np.isfinite(result['log_volume']).all()

    def test_get_feature_columns(self):
        """Test that feature column names are returned correctly."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        # Should have many features
        assert len(cols) > 20

        # Should not include intermediate columns
        assert 'mid' not in cols
        assert 'high' not in cols
        assert 'low' not in cols

    def test_custom_config(self, sample_price_data):
        """Test with custom feature configuration."""
        config = FeatureConfig(
            ma_hours=[1, 4],
            return_hours=[1],
            volatility_hours=[4],
            range_hours=[4]
        )
        engine = FeatureEngine(config=config)
        result = engine.compute_features(sample_price_data)

        # Should have fewer features with limited config
        assert len(result.columns) < 100


class TestFeatureValidator:
    """Tests for FeatureValidator class."""

    def test_validate_good_features(self, sample_price_data):
        """Test validation passes for good features."""
        engine = FeatureEngine()
        result = engine.compute_features(sample_price_data)
        result = result.iloc[300:]  # Skip warmup

        validator = FeatureValidator()
        validation = validator.validate(result, engine.get_feature_columns())

        # Should pass with minimal issues
        assert len(validation['issues']) < 5

    def test_validate_detects_constant_columns(self):
        """Test that constant columns are flagged."""
        df = pd.DataFrame({
            'constant_col': [1.0] * 100,
            'normal_col': np.random.randn(100)
        })

        validator = FeatureValidator()
        validation = validator.validate(df, ['constant_col', 'normal_col'])

        # Should flag constant column
        constant_issues = [i for i in validation['issues'] if 'constant' in i.lower() or 'variance' in i.lower()]
        assert len(constant_issues) > 0


class TestFeatureColumnConsistency:
    """Tests to ensure feature column selection remains consistent across pipeline."""

    def test_critical_features_present(self):
        """Test that MA and day-of-week features are present in feature columns."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        # MA features (critical for price prediction)
        ma_features = [c for c in cols if c.startswith('mid_ma_')]
        assert len(ma_features) >= 6, f"Expected 6+ MA features, got {len(ma_features)}: {ma_features}"

        # Day-of-week cyclical encoding (not raw dayofweek)
        assert 'dow_sin' in cols, "Missing dow_sin cyclical feature"
        assert 'dow_cos' in cols, "Missing dow_cos cyclical feature"

    def test_range_features_present(self):
        """Test that range features (critical for fill probability) are present."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        range_features = [c for c in cols if c.startswith('range_') and 'position' not in c]
        assert len(range_features) >= 6, f"Expected 6+ range features, got {len(range_features)}: {range_features}"

        range_position_features = [c for c in cols if c.startswith('range_position_')]
        assert len(range_position_features) >= 6, f"Expected 6+ range_position features, got {len(range_position_features)}"

        dist_features = [c for c in cols if c.startswith('dist_from_')]
        assert len(dist_features) >= 6, f"Expected 6+ dist features, got {len(dist_features)}: {dist_features}"

    def test_parkinson_volatility_present(self):
        """Test Parkinson volatility features (better than standard deviation)."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        parkinson_features = [c for c in cols if c.startswith('parkinson_')]
        assert len(parkinson_features) >= 3, f"Expected 3+ Parkinson features, got {len(parkinson_features)}: {parkinson_features}"

    def test_time_flags_present(self):
        """Test binary time flags are present."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        assert 'is_weekend' in cols, "Missing is_weekend flag"
        assert 'is_peak_hours' in cols, "Missing is_peak_hours flag"

    def test_feature_count_minimum(self):
        """Test minimum expected feature count to catch silent drops."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        # With default config, should have ~80+ features
        assert len(cols) >= 75, f"Feature count too low: {len(cols)}. Expected 75+."

    def test_no_intermediate_columns_in_features(self):
        """Test that intermediate computation columns are excluded."""
        engine = FeatureEngine()
        cols = engine.get_feature_columns()

        intermediate = ['mid', 'high', 'low', 'spread', 'volume', 'hour', 'day_of_week']
        for col in intermediate:
            assert col not in cols, f"Intermediate column '{col}' should not be in feature columns"


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
