"""Tests for the feature engine."""

import numpy as np
import pandas as pd

from src.feature_engine import FeatureEngine


class TestFeatureEngine:
    """Test cases for FeatureEngine."""

    def test_init_1m_granularity(self):
        """Test initialization with 1-minute granularity."""
        engine = FeatureEngine(granularity="1m")
        assert engine.granularity == "1m"
        assert engine.window_multiplier == 5

    def test_init_5m_granularity(self):
        """Test initialization with 5-minute granularity."""
        engine = FeatureEngine(granularity="5m")
        assert engine.granularity == "5m"
        assert engine.window_multiplier == 1

    def test_compute_features_output_shape(self, sample_price_data):
        """Test that computed features have correct shape."""
        engine = FeatureEngine(granularity="1m")
        features = engine.compute_features(sample_price_data)

        assert len(features.columns) == len(engine.get_feature_names())
        assert len(features) == len(sample_price_data)

    def test_compute_features_no_nan_on_valid_data(self, sample_price_data):
        """Test that features don't contain NaN on valid data."""
        engine = FeatureEngine(granularity="1m")
        features = engine.compute_features(sample_price_data)

        # Last row should have no NaN (after forward filling)
        assert not features.iloc[-1].isna().any()

    def test_get_latest_features(self, sample_price_data):
        """Test getting the latest feature vector."""
        engine = FeatureEngine(granularity="1m")
        features = engine.get_latest_features(sample_price_data)

        assert features is not None
        assert len(features) == len(engine.get_feature_names())
        assert not np.isnan(features).any()

    def test_get_latest_features_empty_data(self):
        """Test handling of empty data."""
        engine = FeatureEngine(granularity="1m")
        features = engine.get_latest_features(pd.DataFrame())

        assert features is None

    def test_get_feature_names(self):
        """Test feature names list."""
        engine = FeatureEngine()
        names = engine.get_feature_names()

        assert isinstance(names, list)
        assert len(names) == 26
        assert "mid" in names
        assert "spread_pct" in names
        assert "hour" in names

    def test_validate_features_valid(self, sample_features):
        """Test validation of valid features."""
        engine = FeatureEngine()
        assert engine.validate_features(sample_features)

    def test_validate_features_with_nan(self, sample_features):
        """Test validation fails with NaN."""
        engine = FeatureEngine()
        features = sample_features.copy()
        features[0] = np.nan
        assert not engine.validate_features(features)

    def test_validate_features_wrong_length(self, sample_features):
        """Test validation fails with wrong length."""
        engine = FeatureEngine()
        features = sample_features[:10]
        assert not engine.validate_features(features)

    def test_get_minimum_history_required(self):
        """Test minimum history calculation."""
        engine = FeatureEngine(granularity="1m")
        min_history = engine.get_minimum_history_required()

        # 1440 * 5 = 7200 for 1-min data
        assert min_history == 7200

    def test_price_basics_calculation(self, sample_price_data):
        """Test basic price feature calculations."""
        engine = FeatureEngine(granularity="1m")
        features = engine.compute_features(sample_price_data)

        # Check mid price calculation
        expected_mid = (sample_price_data["high"] + sample_price_data["low"]) / 2
        assert np.allclose(features["mid"], expected_mid)

        # Check spread calculation
        expected_spread = sample_price_data["high"] - sample_price_data["low"]
        assert np.allclose(features["spread"], expected_spread)

    def test_time_features(self, sample_price_data):
        """Test time-based feature extraction."""
        engine = FeatureEngine(granularity="1m")
        features = engine.compute_features(sample_price_data)

        # Hour should be 0-23
        assert features["hour"].min() >= 0
        assert features["hour"].max() <= 23

        # Day of week should be 0-6
        assert features["day_of_week"].min() >= 0
        assert features["day_of_week"].max() <= 6

        # is_weekend should be 0 or 1
        assert set(features["is_weekend"].unique()).issubset({0, 1})
