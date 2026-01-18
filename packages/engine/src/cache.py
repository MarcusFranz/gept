"""Redis caching layer for API responses.

Provides caching for recommendations, item lookups, and search results
to reduce database load and improve response times.
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional

import redis.asyncio as aioredis

from src.config import config

logger = logging.getLogger(__name__)

# Global Redis client (initialized lazily)
_redis_client: Optional[aioredis.Redis] = None
_cache_available = False


def get_capital_tier(capital: int) -> str:
    """Group capital into tiers for cache key generation.

    This allows caching recommendations by capital tier rather than
    exact capital value, significantly improving cache hit rates.
    """
    if capital < 1_000_000:
        return "under_1m"
    if capital < 5_000_000:
        return "1m_5m"
    if capital < 20_000_000:
        return "5m_20m"
    if capital < 100_000_000:
        return "20m_100m"
    return "over_100m"


def hash_query(query: str) -> str:
    """Hash a search query for cache key generation."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]


async def init_cache() -> bool:
    """Initialize Redis connection.

    Returns True if cache is available, False otherwise.
    """
    global _redis_client, _cache_available

    if not config.cache_enabled:
        logger.info("Cache disabled via CACHE_ENABLED=false")
        return False

    if not config.redis_url:
        logger.info("Cache disabled: REDIS_URL not configured")
        return False

    try:
        _redis_client = aioredis.from_url(
            config.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        await _redis_client.ping()
        _cache_available = True
        logger.info("Redis cache initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Redis cache unavailable: {e}")
        _redis_client = None
        _cache_available = False
        return False


async def close_cache() -> None:
    """Close Redis connection."""
    global _redis_client, _cache_available
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        _cache_available = False
        logger.info("Redis cache connection closed")


async def get_cached(key: str) -> Optional[Any]:
    """Get a value from cache.

    Returns None if cache is unavailable or key doesn't exist.
    """
    if not _cache_available or not _redis_client:
        return None

    try:
        cached = await _redis_client.get(key)
        if cached:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(cached)
        logger.debug(f"Cache MISS: {key}")
        return None
    except Exception as e:
        logger.warning(f"Cache get error for {key}: {e}")
        return None


async def set_cached(key: str, value: Any, ttl: int) -> bool:
    """Set a value in cache with TTL.

    Returns True if cached successfully, False otherwise.
    """
    if not _cache_available or not _redis_client:
        return False

    try:
        await _redis_client.set(key, json.dumps(value), ex=ttl)
        logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.warning(f"Cache set error for {key}: {e}")
        return False


async def clear_pattern(pattern: str) -> int:
    """Clear all keys matching a pattern.

    Returns number of keys deleted.
    """
    if not _cache_available or not _redis_client:
        return 0

    try:
        count = 0
        async for key in _redis_client.scan_iter(match=pattern):
            await _redis_client.delete(key)
            count += 1
        logger.info(f"Cache cleared: {count} keys matching '{pattern}'")
        return count
    except Exception as e:
        logger.warning(f"Cache clear error for pattern {pattern}: {e}")
        return 0


async def clear_all() -> int:
    """Clear all cache keys (recommendations only, not crowding).

    Returns number of keys deleted.
    """
    return await clear_pattern("recs:*") + await clear_pattern("item:*")


async def get_cache_stats() -> dict:
    """Get cache statistics for health checks."""
    if not _cache_available or not _redis_client:
        return {"available": False, "reason": "Redis not connected"}

    try:
        info = await _redis_client.info("stats")
        keyspace = await _redis_client.info("keyspace")

        # Count recommendation-related keys
        recs_count = 0
        async for _ in _redis_client.scan_iter(match="recs:*"):
            recs_count += 1

        item_count = 0
        async for _ in _redis_client.scan_iter(match="item:*"):
            item_count += 1

        return {
            "available": True,
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": (
                round(
                    info.get("keyspace_hits", 0)
                    / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                    )
                    * 100,
                    2,
                )
            ),
            "recommendation_keys": recs_count,
            "item_keys": item_count,
            "total_keys": (
                keyspace.get("db0", {}).get("keys", 0) if "db0" in keyspace else 0
            ),
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def cache_response(
    key_prefix: str,
    ttl_config_key: str = "cache_ttl_recommendations",
    key_builder: Optional[Callable[..., str]] = None,
):
    """Decorator for caching async endpoint responses.

    Args:
        key_prefix: Prefix for cache key (e.g., "recs", "item")
        ttl_config_key: Config attribute name for TTL (e.g., "cache_ttl_recommendations")
        key_builder: Optional function to build cache key from function args

    Usage:
        @cache_response("recs", "cache_ttl_recommendations")
        async def get_recommendations(...):
            ...

        @cache_response("item", "cache_ttl_items", key_builder=lambda item_id, **k: str(item_id))
        async def get_item(item_id: int, ...):
            ...
    """

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check for no_cache parameter
            no_cache = kwargs.pop("no_cache", False)
            if no_cache or not _cache_available:
                return await func(*args, **kwargs)

            # Build cache key
            if key_builder:
                cache_key = f"{key_prefix}:{key_builder(*args, **kwargs)}"
            else:
                # Default: hash all kwargs
                key_parts = ":".join(
                    f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None
                )
                cache_key = f"{key_prefix}:{hash_query(key_parts)}"

            # Try to get from cache
            cached = await get_cached(cache_key)
            if cached is not None:
                return cached

            # Execute function and cache result
            result = await func(*args, **kwargs)

            # Get TTL from config
            ttl = getattr(config, ttl_config_key, 60)
            await set_cached(cache_key, result, ttl)

            return result

        return wrapper

    return decorator


# Cache key builders for common patterns
def recommendations_key(
    style: str = "hybrid",
    risk: str = "medium",
    capital: int = 0,
    **kwargs,
) -> str:
    """Build cache key for recommendations."""
    capital_tier = get_capital_tier(capital)
    return f"{style}:{risk}:{capital_tier}"


def item_key(item_id: int, **kwargs) -> str:
    """Build cache key for item lookups."""
    return str(item_id)


def search_key(q: str, **kwargs) -> str:
    """Build cache key for search queries."""
    return hash_query(q)


def stats_key(hashed_user_id: str, period: str = "week", **kwargs) -> str:
    """Build cache key for user stats."""
    return f"{hashed_user_id}:{period}"
