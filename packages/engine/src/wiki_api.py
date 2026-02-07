"""OSRS Wiki API client for fetching item metadata including buy limits.

This module provides a cached client for the OSRS Wiki Prices API mapping endpoint.
Buy limits are cached in memory with a configurable TTL to minimize API requests.

API Documentation: https://oldschool.runescape.wiki/w/RuneScape:Real-time_Prices
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_USER_AGENT = "GePT-Recommendation-Engine/1.0 (https://github.com/MarcusFranz/gept-recommendation-engine)"
API_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
REQUEST_TIMEOUT = 30  # seconds


class WikiApiClient:
    """Client for OSRS Wiki API with caching support.

    Fetches and caches item buy limits from the Wiki API mapping endpoint.
    Uses lazy loading - the full mapping is fetched on first access and
    cached for the configured TTL.

    Thread-safe for concurrent access.
    """

    _instance: Optional["WikiApiClient"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        """Initialize the Wiki API client.

        Args:
            cache_ttl: Cache duration in seconds (default: 3600 = 1 hour)
            user_agent: User-Agent header for API requests
        """
        self._cache_ttl = timedelta(seconds=cache_ttl)
        self._user_agent = user_agent
        self._buy_limits: dict[int, int] = {}
        self._item_names: dict[int, str] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_lock = threading.RLock()

    def get_buy_limit(self, item_id: int) -> Optional[int]:
        """Get the GE buy limit for an item.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit for the item, or None if not found
        """
        self._ensure_cache_loaded()
        return self._buy_limits.get(item_id)

    def get_item_name(self, item_id: int) -> Optional[str]:
        """Get an item's name from the Wiki mapping.

        Args:
            item_id: OSRS item ID

        Returns:
            Item name, or None if not found.
        """
        self._ensure_cache_loaded()
        return self._item_names.get(item_id)

    def _ensure_cache_loaded(self) -> None:
        """Ensure the cache is loaded and not expired."""
        with self._cache_lock:
            now = datetime.now(timezone.utc)

            # Check if cache is valid
            if (
                self._cache_timestamp is not None
                and now - self._cache_timestamp < self._cache_ttl
            ):
                return

            # Load or refresh cache
            self._load_all_limits()

    def _load_all_limits(self) -> None:
        """Fetch all item buy limits from the Wiki API.

        Populates internal caches:
        - item_id -> buy_limit mappings
        - item_id -> item_name mappings
        """
        try:
            logger.info("Fetching buy limits from OSRS Wiki API...")

            response = requests.get(
                API_URL,
                headers={"User-Agent": self._user_agent},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            items = response.json()

            # Parse response and build buy limits dict
            new_limits: dict[int, int] = {}
            new_names: dict[int, str] = {}
            for item in items:
                item_id = item.get("id")
                name = item.get("name")
                limit = item.get("limit")

                if item_id is None:
                    continue

                item_id_int = int(item_id)
                if name:
                    new_names[item_id_int] = str(name)

                if limit is not None:
                    new_limits[item_id_int] = int(limit)

            # Update cache atomically
            self._buy_limits = new_limits
            self._item_names = new_names
            self._cache_timestamp = datetime.now(timezone.utc)

            logger.info(
                f"Loaded {len(new_names)} item names and {len(new_limits)} buy limits "
                "from OSRS Wiki API"
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch buy limits from Wiki API: {e}")
            # Keep existing cache if available, don't clear on error
            if not self._buy_limits and not self._item_names:
                logger.warning("No cached Wiki mapping available after API failure")
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse Wiki API response: {e}")

    def refresh_cache(self) -> bool:
        """Force refresh the cache regardless of TTL.

        Returns:
            True if refresh succeeded, False otherwise
        """
        with self._cache_lock:
            old_count = len(self._buy_limits)
            self._cache_timestamp = None  # Force refresh
            self._load_all_limits()
            return len(self._buy_limits) > 0 and len(self._buy_limits) >= old_count

    def get_cache_info(self) -> dict:
        """Get information about the current cache state.

        Returns:
            Dictionary with cache metadata
        """
        with self._cache_lock:
            age_seconds = None
            if self._cache_timestamp is not None:
                age_seconds = (
                    datetime.now(timezone.utc) - self._cache_timestamp
                ).total_seconds()

            return {
                "items_cached": len(self._buy_limits),
                "cache_age_seconds": age_seconds,
                "cache_ttl_seconds": self._cache_ttl.total_seconds(),
                "is_expired": (
                    self._cache_timestamp is None
                    or datetime.now(timezone.utc) - self._cache_timestamp
                    >= self._cache_ttl
                ),
            }

    @classmethod
    def get_instance(
        cls,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> "WikiApiClient":
        """Get or create the singleton instance.

        Args:
            cache_ttl: Cache duration in seconds
            user_agent: User-Agent header for API requests

        Returns:
            WikiApiClient singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_ttl=cache_ttl, user_agent=user_agent)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None


def get_wiki_api_client(
    cache_ttl: Optional[int] = None,
    user_agent: Optional[str] = None,
) -> WikiApiClient:
    """Get the Wiki API client singleton.

    Convenience function that loads configuration from environment.

    Args:
        cache_ttl: Override cache TTL (uses config default if None)
        user_agent: Override User-Agent (uses config default if None)

    Returns:
        WikiApiClient singleton instance
    """
    from .config import Config

    config = Config()

    ttl = cache_ttl if cache_ttl is not None else config.wiki_api_cache_ttl
    agent = user_agent if user_agent is not None else config.wiki_api_user_agent

    return WikiApiClient.get_instance(cache_ttl=ttl, user_agent=agent)
