"""Tests for materialized view fallback logic in PredictionLoader."""

from unittest.mock import MagicMock, patch

from src.prediction_loader import PredictionLoader


class TestMatviewExists:
    """Test _matview_exists detection and caching."""

    def _make_loader(self):
        """Create a PredictionLoader without connecting to a real DB."""
        loader = PredictionLoader.__new__(PredictionLoader)
        loader.engine = MagicMock()
        loader.preferred_model_id = ""
        return loader

    def test_returns_false_on_connection_error(self):
        loader = self._make_loader()
        loader.engine.connect.side_effect = Exception("connection refused")
        assert loader._matview_exists("mv_volume_24h") is False

    def test_returns_false_when_matview_missing(self):
        loader = self._make_loader()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = None
        loader.engine.connect.return_value = mock_conn

        assert loader._matview_exists("mv_volume_24h") is False

    def test_returns_true_when_matview_exists(self):
        loader = self._make_loader()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        loader.engine.connect.return_value = mock_conn

        assert loader._matview_exists("mv_volume_24h") is True

    def test_caches_result_per_view(self):
        loader = self._make_loader()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        loader.engine.connect.return_value = mock_conn

        # First call hits DB
        assert loader._matview_exists("mv_volume_24h") is True
        # Second call uses cache — no additional DB call
        assert loader._matview_exists("mv_volume_24h") is True
        assert loader.engine.connect.call_count == 1

    def test_caches_separately_per_view_name(self):
        loader = self._make_loader()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = (1,)
        loader.engine.connect.return_value = mock_conn

        loader._matview_exists("mv_volume_24h")
        loader._matview_exists("mv_volume_1h")
        # Should have 2 DB calls — one per distinct view name
        assert loader.engine.connect.call_count == 2
