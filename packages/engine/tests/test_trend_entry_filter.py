"""Tests for trend entry filter (anti-adverse selection)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestTrendEntryFilter:
    """Test _apply_trend_entry_filter method."""

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
    def stable_predictions(self):
        """Predictions with stable/rising prices."""
        return pd.DataFrame({
            'item_id': [1, 2, 3],
            'item_name': ['Stable', 'Rising', 'Slight Dip'],
            'hour_offset': [4, 4, 4],
            'return_1h': [0.001, 0.02, -0.005],
            'return_4h': [0.005, 0.03, -0.01],
            'return_24h': [0.01, 0.05, -0.02],
        })

    @pytest.fixture
    def falling_predictions(self):
        """Predictions with falling prices."""
        return pd.DataFrame({
            'item_id': [1, 2, 3],
            'item_name': ['Crashing', 'Falling Knife', 'Stable'],
            'hour_offset': [4, 4, 4],
            'return_1h': [-0.03, -0.02, 0.001],
            'return_4h': [-0.05, -0.04, 0.005],
            'return_24h': [-0.08, -0.06, 0.01],
        })

    def test_passes_stable_items_active_style(self, engine, stable_predictions):
        """Stable/rising items should pass for active trading."""
        result = engine._apply_trend_entry_filter(stable_predictions, style='active')
        assert len(result) == 3

    def test_rejects_crashing_items_active_style(self, engine, falling_predictions):
        """Crashing items should be rejected for active trading."""
        result = engine._apply_trend_entry_filter(falling_predictions, style='active')
        assert 'Crashing' not in result['item_name'].values
        assert 'Falling Knife' not in result['item_name'].values
        assert 'Stable' in result['item_name'].values

    def test_passive_style_more_lenient(self, engine, falling_predictions):
        """Passive style should allow moderate dips."""
        falling_predictions['hour_offset'] = 24
        result = engine._apply_trend_entry_filter(falling_predictions, style='passive')
        # Only Crashing should be rejected (>-8% 24h)
        assert 'Crashing' not in result['item_name'].values

    def test_hybrid_style_moderate(self, engine, falling_predictions):
        """Hybrid style should have moderate thresholds."""
        falling_predictions['hour_offset'] = 8
        result = engine._apply_trend_entry_filter(falling_predictions, style='hybrid')
        # Items with >4% 4h drop should be rejected
        assert 'Crashing' not in result['item_name'].values

    def test_skips_filter_if_fields_missing(self, engine):
        """Should pass through if return fields not in predictions."""
        predictions = pd.DataFrame({
            'item_id': [1, 2],
            'item_name': ['Item A', 'Item B'],
            'hour_offset': [4, 4],
        })
        result = engine._apply_trend_entry_filter(predictions, style='active')
        assert len(result) == 2

    def test_handles_none_values(self, engine):
        """Should handle None/NaN values in return fields."""
        predictions = pd.DataFrame({
            'item_id': [1, 2],
            'item_name': ['Item A', 'Item B'],
            'hour_offset': [4, 4],
            'return_1h': [None, -0.01],
            'return_4h': [None, -0.02],
            'return_24h': [None, -0.03],
        })
        result = engine._apply_trend_entry_filter(predictions, style='active')
        assert 1 in result['item_id'].values
