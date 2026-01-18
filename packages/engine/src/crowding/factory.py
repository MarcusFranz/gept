"""Factory for creating crowding tracker instances.

Selects the appropriate implementation based on configuration:
- Redis if REDIS_URL is configured and connection succeeds
- In-memory as fallback or when Redis is not configured
"""

import logging
from typing import Optional

from .base import CrowdingTrackerBase
from .memory import InMemoryCrowdingTracker

logger = logging.getLogger(__name__)


def create_crowding_tracker(
    redis_url: Optional[str] = None,
    fallback_to_memory: bool = True,
) -> CrowdingTrackerBase:
    """Create a crowding tracker instance.

    Args:
        redis_url: Redis connection URL (e.g., redis://localhost:6379/0).
                   If None or empty, uses in-memory tracker.
        fallback_to_memory: If True, fall back to in-memory tracker when
                            Redis is unavailable. If False, raise exception.

    Returns:
        CrowdingTrackerBase implementation (Redis or in-memory)

    Raises:
        Exception: If Redis connection fails and fallback_to_memory is False
    """
    if redis_url:
        try:
            from .redis import RedisCrowdingTracker

            tracker = RedisCrowdingTracker(redis_url)
            tracker.ping()  # Verify connection
            logger.info("Using Redis-backed crowding tracker")
            return tracker
        except ImportError as e:
            error_msg = f"Redis package not installed: {e}"
            logger.warning(error_msg)
            if not fallback_to_memory:
                raise ImportError(error_msg) from e
        except Exception as e:
            error_msg = f"Redis connection failed: {e}"
            logger.warning(error_msg)
            if not fallback_to_memory:
                raise

    logger.info("Using in-memory crowding tracker")
    return InMemoryCrowdingTracker()
