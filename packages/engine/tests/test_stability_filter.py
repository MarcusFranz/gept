"""Tests for price stability filter (anti-manipulation)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestPriceStabilityFilter:
    """Test _apply_price_stability_filter method."""

    @pytest.fixture
    def engine(self):
        """Create a RecommendationEngine with mocked dependencies."""
        from src.recommendation_engine import RecommendationEngine

        mock_loader = MagicMock()
        engine = RecommendationEngine.__new__(RecommendationEngine)
        engine.loader = mock_loader
        engine.logger = MagicMock()
        return engine

    @pytest.fixture
    def normal_predictions(self):
        """Predictions with normal price/median ratios."""
        return pd.DataFrame({
            'item_id': [1, 2, 3],
            'item_name': ['Normal Item', 'Torva platebody', 'Cheap Item'],
            'current_high': [5_000_000, 500_000_000, 100_000],
            'price_vs_median_ratio': [1.02, 1.08, 1.03],  # All within bounds
            'volume_24h': [10000, 200, 50000],
        })

    @pytest.fixture
    def manipulated_predictions(self):
        """Predictions with suspicious price spikes."""
        return pd.DataFrame({
            'item_id': [1, 2, 3],
            'item_name': ['Junk Item', 'Torva platebody', 'Mid Item'],
            'current_high': [100_000, 500_000_000, 5_000_000],
            # Junk item spiked 20% on low volume - manipulation
            # Torva 15% up but low volume is normal for expensive items
            # Mid item 15% up on low volume - suspicious
            'price_vs_median_ratio': [1.20, 1.15, 1.15],
            'volume_24h': [500, 150, 800],
        })

    def test_passes_normal_items(self, engine, normal_predictions):
        """Normal items should all pass the filter."""
        result = engine._apply_price_stability_filter(normal_predictions)
        assert len(result) == 3

    def test_rejects_cheap_manipulated_items(self, engine, manipulated_predictions):
        """Cheap items with price spikes on low volume should be rejected."""
        result = engine._apply_price_stability_filter(manipulated_predictions)
        # Junk item (100k, +20%, 500 vol) should be rejected
        assert 1 not in result['item_id'].values

    def test_allows_expensive_items_with_low_volume(self, engine, manipulated_predictions):
        """Expensive items (>100M) allowed even with low volume."""
        result = engine._apply_price_stability_filter(manipulated_predictions)
        # Torva (500M, +15%, 150 vol) should pass - expensive items have low volume
        assert 2 in result['item_id'].values

    def test_rejects_mid_tier_manipulation(self, engine, manipulated_predictions):
        """Mid-tier items with spikes on low volume should be rejected."""
        result = engine._apply_price_stability_filter(manipulated_predictions)
        # Mid item (5M, +15%, 800 vol) should be rejected
        assert 3 not in result['item_id'].values

    def test_skips_filter_if_fields_missing(self, engine):
        """Should pass through if stability fields not in predictions."""
        predictions = pd.DataFrame({
            'item_id': [1, 2],
            'item_name': ['Item A', 'Item B'],
            # No price_vs_median_ratio column
        })
        result = engine._apply_price_stability_filter(predictions)
        assert len(result) == 2

    def test_handles_none_values(self, engine):
        """Should handle None/NaN values in stability fields."""
        predictions = pd.DataFrame({
            'item_id': [1, 2],
            'item_name': ['Item A', 'Item B'],
            'current_high': [1_000_000, 2_000_000],
            'price_vs_median_ratio': [None, 1.05],
            'volume_24h': [1000, 2000],
        })
        result = engine._apply_price_stability_filter(predictions)
        # Item with None ratio should pass (can't evaluate)
        assert 1 in result['item_id'].values
