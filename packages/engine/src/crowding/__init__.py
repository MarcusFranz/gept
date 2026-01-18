"""Crowding tracker package.

Provides crowding tracking implementations to prevent market manipulation
by limiting concurrent users per item.

Usage:
    from src.crowding import create_crowding_tracker

    # Uses Redis if REDIS_URL configured, otherwise in-memory
    tracker = create_crowding_tracker(redis_url=config.redis_url)

    # Record a delivery
    tracker.record_delivery(item_id=123, user_id="user_abc")

    # Check if item is crowded
    if not tracker.is_crowded(item_id=123, capacity=20):
        # Safe to recommend this item
        pass

    # Get stats for health check
    stats = tracker.get_stats()
"""

from .base import CrowdingTrackerBase
from .factory import create_crowding_tracker
from .memory import InMemoryCrowdingTracker

__all__ = [
    "CrowdingTrackerBase",
    "InMemoryCrowdingTracker",
    "create_crowding_tracker",
]

# RedisCrowdingTracker is imported lazily to avoid requiring redis package
# when not using Redis backend
