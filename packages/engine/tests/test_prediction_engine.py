"""Tests for the prediction engine."""

from unittest.mock import MagicMock, patch

import pytest

from src.prediction_engine import PredictionCache, PredictionEngine


class TestPredictionCache:
    """Test cases for PredictionCache."""

    def test_set_and_get(self):
        """Test basic set and get."""
        cache = PredictionCache(ttl_seconds=60)

        cache.set("test_key", {"value": 123})
        result = cache.get("test_key")

        assert result is not None
        assert result["value"] == 123

    def test_get_missing_key(self):
        """Test getting missing key."""
        cache = PredictionCache(ttl_seconds=60)
        result = cache.get("nonexistent")
        assert result is None

    def test_expired_entry(self):
        """Test that expired entries return None."""
        cache = PredictionCache(ttl_seconds=0)  # Immediate expiry

        cache.set("test_key", {"value": 123})
        result = cache.get("test_key")

        assert result is None

    def test_clear(self):
        """Test clearing cache."""
        cache = PredictionCache(ttl_seconds=60)

        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_lru_eviction(self):
        """Test LRU eviction when max size is exceeded."""
        cache = PredictionCache(ttl_seconds=60, max_size=3)

        # Fill cache to capacity
        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        cache.set("key3", {"c": 3})

        # All should be present
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

        # Add a 4th item - should evict oldest (key1)
        cache.set("key4", {"d": 4})

        # key1 should be evicted (oldest), others remain
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_lru_access_updates_order(self):
        """Test that accessing an item moves it to end (most recently used)."""
        cache = PredictionCache(ttl_seconds=60, max_size=3)

        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        cache.set("key3", {"c": 3})

        # Access key1 - should move it to end (most recently used)
        cache.get("key1")

        # Add key4 - should evict key2 (now oldest after key1 was accessed)
        cache.set("key4", {"d": 4})

        # key2 should be evicted, key1 should remain
        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_update_existing_moves_to_end(self):
        """Test that updating an existing key moves it to end."""
        cache = PredictionCache(ttl_seconds=60, max_size=3)

        cache.set("key1", {"a": 1})
        cache.set("key2", {"b": 2})
        cache.set("key3", {"c": 3})

        # Update key1 - should move it to end
        cache.set("key1", {"a": 100})

        # Add key4 - should evict key2 (oldest after key1 was updated)
        cache.set("key4", {"d": 4})

        assert cache.get("key1") is not None
        assert cache.get("key1")["a"] == 100  # Value updated
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_max_size_default(self):
        """Test that default max_size is 1000."""
        cache = PredictionCache(ttl_seconds=60)
        assert cache.max_size == 1000


class TestPredictionEngine:
    """Test cases for PredictionEngine."""

    @pytest.fixture
    def mock_engine(self, sample_registry, mock_db_connection):
        """Create a prediction engine with mocked dependencies."""
        with patch("src.prediction_engine.DataLoader") as MockDataLoader:
            mock_loader = MagicMock()
            MockDataLoader.return_value = mock_loader

            engine = PredictionEngine(
                db_connection_string=mock_db_connection,
                registry_path=str(sample_registry),
            )

            return engine, mock_loader

    def test_init(self, mock_engine):
        """Test engine initialization."""
        engine, _ = mock_engine

        assert engine.data_loader is not None
        assert engine.feature_engine is not None
        assert engine.model_runner is not None
        assert engine.output_formatter is not None
        assert engine.cache is not None

    def test_supported_items(self, mock_engine):
        """Test getting supported items."""
        engine, _ = mock_engine
        items = engine.supported_items

        assert 554 in items
        assert 565 in items

    def test_predict_item_unsupported(self, mock_engine):
        """Test prediction for unsupported item."""
        engine, _ = mock_engine

        result = engine.predict_item(999999)
        assert result is None

    def test_predict_item_with_cache(self, mock_engine, sample_price_data):
        """Test that cache is used correctly."""
        engine, mock_loader = mock_engine

        # Setup mock to return price data
        mock_loader.get_recent_prices.return_value = sample_price_data

        # First call - should fetch from database
        engine.cache.set("item_554", {"cached": True})
        result = engine.predict_item(554, use_cache=True)

        assert result is not None
        assert result.get("cached") is True

        # Verify database was not called (cache hit)
        mock_loader.get_recent_prices.assert_not_called()

    def test_predict_item_bypass_cache(self, mock_engine, sample_price_data):
        """Test bypassing cache."""
        engine, mock_loader = mock_engine

        # Setup mock
        mock_loader.get_recent_prices.return_value = sample_price_data

        # Bypass cache
        engine.cache.set("item_554", {"cached": True})

        # This will try to predict but will fail due to no models
        engine.predict_item(554, use_cache=False)

        # Database should be called
        mock_loader.get_recent_prices.assert_called()

    def test_get_recommendations_parameters(self, mock_engine):
        """Test that recommendation parameters are handled correctly."""
        engine, _ = mock_engine

        # Mock predict_actionable to check parameters
        engine.predict_actionable = MagicMock(return_value=[])

        engine.get_recommendations(
            style="active",
            capital=10_000_000,
            risk="high",
            slots=4,
        )

        # Verify predict_actionable was called
        engine.predict_actionable.assert_called_once()

    def test_get_min_ev_for_risk(self, mock_engine):
        """Test EV threshold varies by risk."""
        engine, _ = mock_engine

        assert engine._get_min_ev_for_risk("low") > engine._get_min_ev_for_risk(
            "medium"
        )
        assert engine._get_min_ev_for_risk("medium") > engine._get_min_ev_for_risk(
            "high"
        )

    def test_get_hours_for_style(self, mock_engine):
        """Test hour preferences vary by style."""
        engine, _ = mock_engine

        passive_hours = engine._get_hours_for_style("passive")
        active_hours = engine._get_hours_for_style("active")

        # Passive prefers longer windows
        assert max(passive_hours) > max(active_hours)

    def test_get_tiers_for_risk(self, mock_engine):
        """Test tier preferences vary by risk."""
        engine, _ = mock_engine

        low_tiers = engine._get_tiers_for_risk("low")
        high_tiers = engine._get_tiers_for_risk("high")

        # Low risk is more restrictive
        assert len(low_tiers) < len(high_tiers)

    def test_health_check_structure(self, mock_engine):
        """Test health check returns expected structure."""
        engine, mock_loader = mock_engine

        # Mock health checks
        mock_loader.health_check.return_value = {
            "status": "ok",
            "component": "database",
        }

        health = engine.health_check()

        assert "status" in health
        assert "checks" in health
        assert "timestamp" in health
        assert "supported_items" in health

    def test_close(self, mock_engine):
        """Test engine cleanup."""
        engine, mock_loader = mock_engine

        engine.cache.set("test", {"value": 1})
        engine.close()

        mock_loader.close.assert_called_once()
        assert engine.cache.get("test") is None
