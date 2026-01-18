"""Tests for the OSRS Wiki API client."""

import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.wiki_api import WikiApiClient, get_wiki_api_client


class TestWikiApiClient:
    """Test cases for WikiApiClient."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        WikiApiClient.reset_instance()
        yield
        WikiApiClient.reset_instance()

    def test_init_defaults(self):
        """Test default initialization values."""
        client = WikiApiClient()

        assert client._cache_ttl == timedelta(seconds=3600)
        assert "GePT" in client._user_agent
        assert client._buy_limits == {}
        assert client._cache_timestamp is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        client = WikiApiClient(cache_ttl=1800, user_agent="Test/1.0")

        assert client._cache_ttl == timedelta(seconds=1800)
        assert client._user_agent == "Test/1.0"

    @patch("src.wiki_api.requests.get")
    def test_get_buy_limit_successful(self, mock_get):
        """Test successful buy limit retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
            {"id": 565, "name": "Blood rune", "limit": 10000},
            {"id": 13652, "name": "Dragon claws", "limit": 8},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = WikiApiClient()

        # Test retrieval
        assert client.get_buy_limit(554) == 25000
        assert client.get_buy_limit(565) == 10000
        assert client.get_buy_limit(13652) == 8

        # Unknown item should return None
        assert client.get_buy_limit(99999) is None

        # Should only make one API call (cached)
        assert mock_get.call_count == 1

    @patch("src.wiki_api.requests.get")
    def test_get_buy_limit_missing_limit(self, mock_get):
        """Test handling items without buy limit in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
            {"id": 555, "name": "Water rune"},  # No limit field
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = WikiApiClient()

        assert client.get_buy_limit(554) == 25000
        assert client.get_buy_limit(555) is None

    @patch("src.wiki_api.requests.get")
    def test_cache_expiry(self, mock_get):
        """Test cache expiry and refresh."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Use short TTL for testing
        client = WikiApiClient(cache_ttl=1)

        # First call loads cache
        assert client.get_buy_limit(554) == 25000
        assert mock_get.call_count == 1

        # Same call within TTL uses cache
        assert client.get_buy_limit(554) == 25000
        assert mock_get.call_count == 1

        # Wait for cache to expire
        time.sleep(1.1)

        # Next call should refresh cache
        assert client.get_buy_limit(554) == 25000
        assert mock_get.call_count == 2

    @patch("src.wiki_api.requests.get")
    def test_network_error_handling(self, mock_get):
        """Test graceful handling of network errors."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        client = WikiApiClient()

        # Should return None without crashing
        assert client.get_buy_limit(554) is None

    @patch("src.wiki_api.requests.get")
    def test_invalid_json_handling(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = WikiApiClient()

        # Should return None without crashing
        assert client.get_buy_limit(554) is None

    @patch("src.wiki_api.requests.get")
    def test_http_error_handling(self, mock_get):
        """Test handling of HTTP errors."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_get.return_value = mock_response

        client = WikiApiClient()

        # Should return None without crashing
        assert client.get_buy_limit(554) is None

    @patch("src.wiki_api.requests.get")
    def test_refresh_cache(self, mock_get):
        """Test manual cache refresh."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = WikiApiClient()

        # Initial load
        client.get_buy_limit(554)
        assert mock_get.call_count == 1

        # Manual refresh
        result = client.refresh_cache()
        assert result is True
        assert mock_get.call_count == 2

    @patch("src.wiki_api.requests.get")
    def test_get_cache_info(self, mock_get):
        """Test cache info retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
            {"id": 565, "name": "Blood rune", "limit": 10000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = WikiApiClient(cache_ttl=3600)

        # Before any load
        info = client.get_cache_info()
        assert info["items_cached"] == 0
        assert info["cache_age_seconds"] is None
        assert info["is_expired"] is True

        # After load
        client.get_buy_limit(554)
        info = client.get_cache_info()
        assert info["items_cached"] == 2
        assert info["cache_age_seconds"] is not None
        assert info["cache_age_seconds"] < 1  # Just loaded
        assert info["is_expired"] is False
        assert info["cache_ttl_seconds"] == 3600

    @patch("src.wiki_api.requests.get")
    def test_preserves_cache_on_error(self, mock_get):
        """Test that existing cache is preserved on refresh error."""
        import requests

        # First call succeeds
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = WikiApiClient(cache_ttl=1)
        assert client.get_buy_limit(554) == 25000

        # Second call (after expiry) fails
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")
        time.sleep(1.1)

        # Should still have cached value (cache preserved on error)
        assert client.get_buy_limit(554) == 25000


class TestWikiApiClientSingleton:
    """Test singleton pattern for WikiApiClient."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        WikiApiClient.reset_instance()
        yield
        WikiApiClient.reset_instance()

    def test_singleton_returns_same_instance(self):
        """Test that get_instance returns the same instance."""
        instance1 = WikiApiClient.get_instance()
        instance2 = WikiApiClient.get_instance()

        assert instance1 is instance2

    def test_singleton_with_different_params(self):
        """Test that singleton ignores params on subsequent calls."""
        instance1 = WikiApiClient.get_instance(cache_ttl=1000)
        instance2 = WikiApiClient.get_instance(cache_ttl=2000)

        assert instance1 is instance2
        assert instance1._cache_ttl == timedelta(seconds=1000)

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = WikiApiClient.get_instance()
        WikiApiClient.reset_instance()
        instance2 = WikiApiClient.get_instance()

        assert instance1 is not instance2

    @patch("src.wiki_api.requests.get")
    def test_thread_safety(self, mock_get):
        """Test thread-safe singleton access."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        instances = []
        errors = []

        def get_instance():
            try:
                instance = WikiApiClient.get_instance()
                # Force cache load
                instance.get_buy_limit(554)
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(instances) == 10
        # All should be same instance
        assert all(inst is instances[0] for inst in instances)


class TestGetWikiApiClient:
    """Test the convenience function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        WikiApiClient.reset_instance()
        yield
        WikiApiClient.reset_instance()

    @patch("src.config.Config")
    def test_uses_config_defaults(self, mock_config_class):
        """Test that get_wiki_api_client uses config defaults."""
        mock_config = MagicMock()
        mock_config.wiki_api_cache_ttl = 7200
        mock_config.wiki_api_user_agent = "Custom/1.0"
        mock_config_class.return_value = mock_config

        client = get_wiki_api_client()

        assert client._cache_ttl == timedelta(seconds=7200)
        assert client._user_agent == "Custom/1.0"

    @patch("src.config.Config")
    def test_override_params(self, mock_config_class):
        """Test that explicit params override config."""
        mock_config = MagicMock()
        mock_config.wiki_api_cache_ttl = 7200
        mock_config.wiki_api_user_agent = "Custom/1.0"
        mock_config_class.return_value = mock_config

        client = get_wiki_api_client(cache_ttl=1800, user_agent="Override/1.0")

        assert client._cache_ttl == timedelta(seconds=1800)
        assert client._user_agent == "Override/1.0"


class TestPredictionLoaderIntegration:
    """Test integration with PredictionLoader."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        WikiApiClient.reset_instance()
        yield
        WikiApiClient.reset_instance()

    @patch("src.wiki_api.requests.get")
    @patch("src.prediction_loader.PredictionLoader._get_db_buy_limit")
    def test_wiki_api_preferred_over_db(self, mock_db_limit, mock_wiki_get):
        """Test that Wiki API is used when available (authoritative source)."""
        from src.prediction_loader import PredictionLoader

        # DB returns limit (would be incorrect - e.g., 10000 for all items)
        mock_db_limit.return_value = 10000

        # Wiki API returns correct limit
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 554, "name": "Fire rune", "limit": 25000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_wiki_get.return_value = mock_response

        # Create loader with mock connection
        with patch("src.prediction_loader.create_engine"):
            loader = PredictionLoader("postgresql://test:test@localhost/test")

            # Replace the method to use our mock
            loader._get_db_buy_limit = mock_db_limit

            limit = loader.get_item_buy_limit(554)

            # Wiki API limit should be returned (authoritative source)
            assert limit == 25000
            mock_wiki_get.assert_called_once()
            # DB should NOT be called (Wiki API succeeded)
            mock_db_limit.assert_not_called()

    @patch("src.wiki_api.requests.get")
    @patch("src.prediction_loader.PredictionLoader._get_db_buy_limit")
    def test_fallback_to_db_when_wiki_returns_none(self, mock_db_limit, mock_wiki_get):
        """Test that DB is used as fallback when Wiki API returns None."""
        from src.prediction_loader import PredictionLoader

        # Wiki API returns item mapping but not for this item
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": 999, "name": "Other item", "limit": 1000},
        ]
        mock_response.raise_for_status = MagicMock()
        mock_wiki_get.return_value = mock_response

        # DB returns limit
        mock_db_limit.return_value = 30000

        # Create loader with mock connection
        with patch("src.prediction_loader.create_engine"):
            loader = PredictionLoader("postgresql://test:test@localhost/test")
            loader._get_db_buy_limit = mock_db_limit

            limit = loader.get_item_buy_limit(554)

            # DB limit should be returned as fallback
            assert limit == 30000
            mock_wiki_get.assert_called_once()
            mock_db_limit.assert_called_once_with(554)

    @patch("src.wiki_api.requests.get")
    @patch("src.prediction_loader.PredictionLoader._get_db_buy_limit")
    @patch("src.config.Config")
    def test_wiki_disabled_uses_db_fallback(
        self, mock_config_class, mock_db_limit, mock_wiki_get
    ):
        """Test that DB is used when Wiki API is disabled."""
        from src.prediction_loader import PredictionLoader

        # DB returns limit
        mock_db_limit.return_value = 10000

        # Wiki API disabled
        mock_config = MagicMock()
        mock_config.wiki_api_enabled = False
        mock_config_class.return_value = mock_config

        # Create loader with mock connection
        with patch("src.prediction_loader.create_engine"):
            loader = PredictionLoader("postgresql://test:test@localhost/test")
            loader._get_db_buy_limit = mock_db_limit

            limit = loader.get_item_buy_limit(554)

            # Should use DB (Wiki API disabled)
            assert limit == 10000
            mock_wiki_get.assert_not_called()
            mock_db_limit.assert_called_once_with(554)
