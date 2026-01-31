"""Tests for Redis caching layer."""

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.cache import (
    get_capital_tier,
    hash_query,
    init_cache,
)


class TestCapitalTier:
    """Test capital tier grouping."""

    def test_under_1m(self):
        """Test capital under 1M."""
        assert get_capital_tier(0) == "under_1m"
        assert get_capital_tier(500_000) == "under_1m"
        assert get_capital_tier(999_999) == "under_1m"

    def test_1m_5m(self):
        """Test capital between 1M and 5M."""
        assert get_capital_tier(1_000_000) == "1m_5m"
        assert get_capital_tier(3_000_000) == "1m_5m"
        assert get_capital_tier(4_999_999) == "1m_5m"

    def test_5m_20m(self):
        """Test capital between 5M and 20M."""
        assert get_capital_tier(5_000_000) == "5m_20m"
        assert get_capital_tier(10_000_000) == "5m_20m"
        assert get_capital_tier(19_999_999) == "5m_20m"

    def test_20m_100m(self):
        """Test capital between 20M and 100M."""
        assert get_capital_tier(20_000_000) == "20m_100m"
        assert get_capital_tier(50_000_000) == "20m_100m"
        assert get_capital_tier(99_999_999) == "20m_100m"

    def test_over_100m(self):
        """Test capital over 100M."""
        assert get_capital_tier(100_000_000) == "over_100m"
        assert get_capital_tier(500_000_000) == "over_100m"
        assert get_capital_tier(1_000_000_000) == "over_100m"


class TestHashQuery:
    """Test query hashing."""

    def test_basic_hash(self):
        """Test basic query hashing."""
        result = hash_query("fire rune")
        assert isinstance(result, str)
        assert len(result) == 12  # Truncated MD5

    def test_case_insensitive(self):
        """Test hashing is case insensitive."""
        assert hash_query("Fire Rune") == hash_query("fire rune")
        assert hash_query("FIRE RUNE") == hash_query("fire rune")

    def test_whitespace_handling(self):
        """Test whitespace is trimmed."""
        assert hash_query("  fire rune  ") == hash_query("fire rune")

    def test_different_queries(self):
        """Test different queries produce different hashes."""
        assert hash_query("fire rune") != hash_query("water rune")


class TestCacheOperations:
    """Test cache operations with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock()
        mock.ping = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock()
        mock.delete = AsyncMock()
        mock.scan_iter = AsyncMock(return_value=iter([]))
        mock.info = AsyncMock(
            return_value={"keyspace_hits": 100, "keyspace_misses": 20}
        )
        mock.close = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_init_cache_disabled(self):
        """Test cache init when disabled."""
        with patch.dict("os.environ", {"CACHE_ENABLED": "false"}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            result = await init_cache()
            assert result is False

            # Clean up
            if "CACHE_ENABLED" in os.environ:
                del os.environ["CACHE_ENABLED"]
            importlib.reload(config_module)

    @pytest.mark.asyncio
    async def test_init_cache_no_redis_url(self):
        """Test cache init without Redis URL."""
        with patch.dict(
            "os.environ",
            {"CACHE_ENABLED": "true", "REDIS_URL": ""},
            clear=False,
        ):
            import importlib
            import src.config as config_module
            import src.cache as cache_module

            importlib.reload(config_module)
            importlib.reload(cache_module)

            result = await cache_module.init_cache()
            assert result is False

            # Clean up
            importlib.reload(config_module)
            importlib.reload(cache_module)

    @pytest.mark.asyncio
    async def test_get_cached_unavailable(self):
        """Test get_cached when cache is unavailable."""
        import src.cache as cache_module

        # Ensure cache is not available
        cache_module._cache_available = False
        cache_module._redis_client = None

        result = await cache_module.get_cached("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_cached_unavailable(self):
        """Test set_cached when cache is unavailable."""
        import src.cache as cache_module

        cache_module._cache_available = False
        cache_module._redis_client = None

        result = await cache_module.set_cached("test_key", {"data": "test"}, 60)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_cache_stats_unavailable(self):
        """Test get_cache_stats when cache is unavailable."""
        import src.cache as cache_module

        cache_module._cache_available = False
        cache_module._redis_client = None

        stats = await cache_module.get_cache_stats()
        assert stats["available"] is False


class TestCacheConfig:
    """Test cache configuration."""

    def test_cache_ttl_recommendations(self):
        """Test cache TTL for recommendations."""
        with patch.dict("os.environ", {"CACHE_TTL_RECOMMENDATIONS": "45"}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.cache_ttl_recommendations == 45

            # Clean up
            if "CACHE_TTL_RECOMMENDATIONS" in os.environ:
                del os.environ["CACHE_TTL_RECOMMENDATIONS"]
            importlib.reload(config_module)

    def test_cache_ttl_items(self):
        """Test cache TTL for items."""
        with patch.dict("os.environ", {"CACHE_TTL_ITEMS": "90"}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.cache_ttl_items == 90

            # Clean up
            if "CACHE_TTL_ITEMS" in os.environ:
                del os.environ["CACHE_TTL_ITEMS"]
            importlib.reload(config_module)

    def test_cache_ttl_search(self):
        """Test cache TTL for search."""
        with patch.dict("os.environ", {"CACHE_TTL_SEARCH": "600"}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.cache_ttl_search == 600

            # Clean up
            if "CACHE_TTL_SEARCH" in os.environ:
                del os.environ["CACHE_TTL_SEARCH"]
            importlib.reload(config_module)

    def test_cache_enabled_default(self):
        """Test cache is enabled by default."""
        import os

        # Ensure env var is not set
        env_backup = os.environ.get("CACHE_ENABLED")
        if "CACHE_ENABLED" in os.environ:
            del os.environ["CACHE_ENABLED"]

        try:
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.cache_enabled is True
        finally:
            if env_backup is not None:
                os.environ["CACHE_ENABLED"] = env_backup
            import src.config as config_module

            importlib.reload(config_module)

    def test_cache_disabled(self):
        """Test cache can be disabled."""
        with patch.dict("os.environ", {"CACHE_ENABLED": "false"}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.cache_enabled is False

            # Clean up
            if "CACHE_ENABLED" in os.environ:
                del os.environ["CACHE_ENABLED"]
            importlib.reload(config_module)
