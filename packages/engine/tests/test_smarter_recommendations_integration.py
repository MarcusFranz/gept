"""Integration tests for smarter recommendation engine filters."""

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestSmarterRecommendationsIntegration:
    """End-to-end tests for the new filtering logic."""

    @pytest.fixture
    def mock_predictions_df(self):
        """Create test prediction data with mix of good and bad predictions."""
        return pd.DataFrame({
            'item_id': [1, 2, 3, 4, 5],
            'item_name': [
                'Good Item',           # Normal, should pass
                'Manipulated Junk',    # High ratio, low volume, cheap
                'Falling Knife',       # Good stability, bad momentum
                'Torva platebody',     # Low volume but expensive, ok
                'Recovering Item',     # Was falling, now stable
            ],
            'current_high': [5_000_000, 500_000, 3_000_000, 500_000_000, 2_000_000],
            'current_low': [4_900_000, 480_000, 2_900_000, 495_000_000, 1_950_000],
            'buy_price': [4_900_000, 480_000, 2_900_000, 495_000_000, 1_950_000],
            'sell_price': [5_100_000, 520_000, 3_100_000, 510_000_000, 2_050_000],
            'hour_offset': [4, 4, 4, 12, 4],
            'offset_pct': [0.02, 0.02, 0.02, 0.015, 0.02],
            'fill_probability': [0.5, 0.6, 0.55, 0.4, 0.5],
            'expected_value': [0.01, 0.015, 0.012, 0.008, 0.01],
            'confidence': ['medium'] * 5,
            # Stability fields
            'median_14d': [4_800_000, 400_000, 3_000_000, 480_000_000, 2_000_000],
            'price_vs_median_ratio': [1.04, 1.25, 1.00, 1.04, 1.00],
            'volume_24h': [15000, 300, 8000, 150, 5000],
            # Momentum fields
            'return_1h': [0.005, 0.01, -0.02, 0.002, 0.001],
            'return_4h': [0.01, 0.02, -0.04, 0.005, -0.01],
            'return_24h': [0.02, 0.05, -0.06, 0.01, -0.03],
            'volatility_24h': [0.01, 0.02, 0.015, 0.008, 0.012],
        })

    @pytest.fixture
    def mock_loader(self, mock_predictions_df):
        """Create mock prediction loader with test data."""
        loader = MagicMock()

        # Return the test predictions
        loader.get_best_prediction_per_item.return_value = mock_predictions_df.copy()

        # Mock batch methods
        loader.get_batch_buy_limits.return_value = {
            1: 100, 2: 1000, 3: 50, 4: 2, 5: 100
        }
        loader.get_batch_volumes_24h.return_value = {
            1: 15000, 2: 300, 3: 8000, 4: 150, 5: 5000
        }
        loader.get_batch_volumes_1h.return_value = {
            1: 500, 2: 10, 3: 300, 4: 5, 5: 200
        }
        loader.get_batch_trends.return_value = {
            1: 'Stable', 2: 'Rising', 3: 'Falling', 4: 'Stable', 5: 'Stable'
        }
        loader.get_prediction_age_seconds.return_value = 60.0

        return loader

    @pytest.fixture
    def engine(self, mock_loader):
        """Create RecommendationEngine with mock loader."""
        from src.config import Config
        from src.recommendation_engine import RecommendationEngine

        # Create engine without invoking __init__ to bypass DB connection
        engine = RecommendationEngine.__new__(RecommendationEngine)

        # Set up required attributes
        engine.config = Config()
        engine.config.price_buffer_enabled = False  # Disable randomness in tests
        engine.loader = mock_loader
        engine.store = MagicMock()
        engine.crowding_tracker = MagicMock()
        engine.crowding_tracker.filter_crowded_items.side_effect = lambda x: x

        # ML ranker shadow mode attributes (disabled for tests)
        engine.enable_ml_shadow = False
        engine.ml_ranker = None
        engine.ml_feature_builder = None
        engine.shadow_logger = None

        return engine

    def test_manipulated_item_rejected(self, engine):
        """Manipulated junk item should be filtered out by stability filter."""
        # The 'Manipulated Junk' item has:
        # - Low value (500k)
        # - High price_vs_median_ratio (1.25 = 25% above)
        # - Low volume (300)
        # This should trigger the stability filter

        recommendations = engine.get_recommendations(
            style='active',
            capital=10_000_000,
            risk='medium',
            slots=3,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        item_names = [r['item'] for r in recommendations]
        assert 'Manipulated Junk' not in item_names, \
            "Manipulated item with high ratio + low volume should be filtered"

    def test_falling_knife_rejected_for_active(self, engine):
        """Falling knife should be rejected for active trading."""
        # The 'Falling Knife' item has:
        # - Good stability (ratio 1.0)
        # - But bad momentum: -2% 1h, -4% 4h
        # For active style, this should be rejected

        recommendations = engine.get_recommendations(
            style='active',
            capital=10_000_000,
            risk='medium',
            slots=3,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        item_names = [r['item'] for r in recommendations]
        assert 'Falling Knife' not in item_names, \
            "Item with bad momentum should be filtered for active style"

    def test_expensive_low_volume_allowed(self, engine, mock_loader, mock_predictions_df):
        """Expensive items should pass even with low volume."""
        # Torva platebody has:
        # - High value (500M)
        # - Low volume (150) - normal for expensive items
        # - Stable ratio (1.04)
        # Should pass both filters
        #
        # Note: With 495M buy price and max 30% single trade allocation (medium risk),
        # user needs at least 1.65B capital to afford 1 item. We use 2B to be safe.
        # Alternatively using high risk allows 50% allocation (needs 990M).

        # Filter predictions to ONLY include Torva platebody for this test
        # to ensure it's not filtered out by stability/trend checks
        torva_df = mock_predictions_df[
            mock_predictions_df['item_name'] == 'Torva platebody'
        ].copy()
        mock_loader.get_best_prediction_per_item.return_value = torva_df
        mock_loader.get_batch_buy_limits.return_value = {4: 2}
        mock_loader.get_batch_volumes_24h.return_value = {4: 150}
        mock_loader.get_batch_volumes_1h.return_value = {4: 5}
        mock_loader.get_batch_trends.return_value = {4: 'Stable'}

        # Use high risk which allows 50% single trade allocation (495M needs 990M capital)
        recommendations = engine.get_recommendations(
            style='passive',
            capital=1_000_000_000,  # 1B capital
            risk='high',  # High risk allows 50% allocation per trade
            slots=2,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        item_names = [r['item'] for r in recommendations]
        assert 'Torva platebody' in item_names, \
            "Expensive items should pass with low volume"

    def test_good_item_passes_all_filters(self, engine, mock_loader, mock_predictions_df):
        """Normal good items should pass all filters."""
        # Filter to ONLY Good Item for this test
        # Note: Good Item buy_price is 4.9M. With medium risk (30% max allocation),
        # we need 4.9M / 0.30 = ~16.3M capital minimum to afford 1 item.
        good_item_df = mock_predictions_df[
            mock_predictions_df['item_name'] == 'Good Item'
        ].copy()
        mock_loader.get_best_prediction_per_item.return_value = good_item_df
        mock_loader.get_batch_buy_limits.return_value = {1: 100}
        mock_loader.get_batch_volumes_24h.return_value = {1: 15000}
        mock_loader.get_batch_volumes_1h.return_value = {1: 500}
        mock_loader.get_batch_trends.return_value = {1: 'Stable'}

        # Use 20M capital so 30% allocation (6M) can afford 1 item at 4.9M
        recommendations = engine.get_recommendations(
            style='active',
            capital=20_000_000,
            risk='medium',
            slots=3,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        item_names = [r['item'] for r in recommendations]
        assert 'Good Item' in item_names, \
            "Normal stable items should pass all filters"

    def test_recovering_item_passes(self, engine, mock_loader, mock_predictions_df):
        """Item that was falling but now stable should pass."""
        # Recovering Item has:
        # - return_24h: -0.03 (was falling)
        # - return_1h: 0.001 (now stable)
        # - return_4h: -0.01 (recovering)
        # For active style, recent stability matters more

        # Filter to include Recovering Item
        filtered_df = mock_predictions_df[
            mock_predictions_df['item_name'] == 'Recovering Item'
        ].copy()
        mock_loader.get_best_prediction_per_item.return_value = filtered_df

        recommendations = engine.get_recommendations(
            style='active',
            capital=10_000_000,
            risk='medium',
            slots=3,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        item_names = [r['item'] for r in recommendations]
        assert 'Recovering Item' in item_names, \
            "Item recovering from dip should pass for active style"

    def test_filter_order_stability_then_trend(self, engine, mock_loader, mock_predictions_df):
        """Verify both filters are applied in sequence."""
        # Start with all items
        mock_loader.get_best_prediction_per_item.return_value = mock_predictions_df.copy()

        recommendations = engine.get_recommendations(
            style='active',
            capital=10_000_000,
            risk='medium',
            slots=5,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        item_names = [r['item'] for r in recommendations]

        # Manipulated Junk should be caught by stability filter
        assert 'Manipulated Junk' not in item_names

        # Falling Knife should be caught by trend filter
        assert 'Falling Knife' not in item_names

        # Good items should remain
        assert 'Good Item' in item_names or 'Recovering Item' in item_names

    def test_passive_style_more_lenient_on_trends(self, engine, mock_loader, mock_predictions_df):
        """Passive style should allow items that active rejects."""
        # Create a prediction that would fail active but pass passive
        # Moderate 24h decline but not severe
        # Note: passive mode has capital efficiency threshold, so we use
        # 50M capital to ensure the trade is viable for passive trading
        moderate_decline_df = pd.DataFrame({
            'item_id': [10],
            'item_name': ['Moderate Decline'],
            'current_high': [5_000_000],
            'current_low': [4_900_000],
            'buy_price': [4_900_000],
            'sell_price': [5_100_000],
            'hour_offset': [24],  # Long horizon for passive
            'offset_pct': [0.02],
            'fill_probability': [0.5],
            'expected_value': [0.01],
            'confidence': ['medium'],
            'price_vs_median_ratio': [1.02],
            'volume_24h': [10000],
            'return_1h': [-0.005],
            'return_4h': [-0.015],
            'return_24h': [-0.04],  # Moderate decline
        })

        mock_loader.get_best_prediction_per_item.return_value = moderate_decline_df
        mock_loader.get_batch_buy_limits.return_value = {10: 100}
        mock_loader.get_batch_volumes_24h.return_value = {10: 10000}
        mock_loader.get_batch_volumes_1h.return_value = {10: 500}
        mock_loader.get_batch_trends.return_value = {10: 'Falling'}

        # Should pass for passive style (longer horizon, more lenient)
        # Using 50M capital to meet passive mode's capital efficiency threshold
        passive_recs = engine.get_recommendations(
            style='passive',
            capital=50_000_000,
            risk='medium',
            slots=3,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        passive_names = [r['item'] for r in passive_recs]
        assert 'Moderate Decline' in passive_names, \
            "Passive style should allow moderate declines"

    def test_empty_after_both_filters(self, engine, mock_loader):
        """All items filtered should return empty list gracefully."""
        # Create only bad items
        all_bad_df = pd.DataFrame({
            'item_id': [1, 2],
            'item_name': ['Manipulated', 'Crashing'],
            'current_high': [100_000, 100_000],
            'current_low': [95_000, 95_000],
            'buy_price': [95_000, 95_000],
            'sell_price': [105_000, 105_000],
            'hour_offset': [4, 4],
            'offset_pct': [0.02, 0.02],
            'fill_probability': [0.5, 0.5],
            'expected_value': [0.01, 0.01],
            'confidence': ['medium', 'medium'],
            'price_vs_median_ratio': [1.30, 1.05],  # First manipulated
            'volume_24h': [200, 500],  # Low volumes
            'return_1h': [0.01, -0.03],  # Second crashing
            'return_4h': [0.02, -0.05],
            'return_24h': [0.05, -0.08],
        })

        mock_loader.get_best_prediction_per_item.return_value = all_bad_df
        mock_loader.get_batch_buy_limits.return_value = {1: 1000, 2: 1000}
        mock_loader.get_batch_volumes_24h.return_value = {1: 200, 2: 500}
        mock_loader.get_batch_volumes_1h.return_value = {1: 10, 2: 20}
        mock_loader.get_batch_trends.return_value = {1: 'Rising', 2: 'Falling'}

        recommendations = engine.get_recommendations(
            style='active',
            capital=10_000_000,
            risk='medium',
            slots=3,
            active_trades=[],
            exclude_ids=set(),
            exclude_item_ids=set(),
            user_id='test',
        )

        assert recommendations == [], \
            "Should return empty list when all items filtered"
