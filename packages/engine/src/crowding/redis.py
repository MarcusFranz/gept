"""Redis-backed crowding tracker implementation.

Uses Redis Sets with TTL-based expiry for automatic cleanup.
Suitable for multi-instance deployments where crowding state
must be shared across processes.
"""

import logging
from datetime import datetime, timedelta, timezone

import redis

from .base import CrowdingTrackerBase

logger = logging.getLogger(__name__)


class RedisCrowdingTracker(CrowdingTrackerBase):
    """Redis-backed implementation of crowding tracker.

    Uses Redis Sets to track unique users per item within time buckets.
    Keys automatically expire via TTL, eliminating need for manual cleanup.

    Key structure: crowding:{item_id}:{time_bucket}
    - item_id: OSRS item ID
    - time_bucket: Timestamp rounded to 15-minute intervals (YYYYMMDDHHMM)

    Each key is a Set containing user_ids. TTL is set to 4h 15m.
    """

    KEY_PREFIX = "crowding"
    BUCKET_MINUTES = 15
    # 4 hours + 15 minutes buffer for bucket overlap
    TTL_SECONDS = int(4.25 * 3600)  # 15300 seconds

    def __init__(self, redis_url: str):
        """Initialize Redis crowding tracker.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
        """
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._redis_url = redis_url

    def ping(self) -> bool:
        """Test Redis connection.

        Returns:
            True if connection is healthy

        Raises:
            redis.ConnectionError: If connection fails
        """
        return self._redis.ping()

    def _get_bucket_key(self, item_id: int, timestamp: datetime) -> str:
        """Generate Redis key for a time bucket.

        Args:
            item_id: OSRS item ID
            timestamp: Timestamp to bucket

        Returns:
            Redis key string (e.g., crowding:123:202601121430)
        """
        # Round down to nearest 15-minute bucket
        minute = (timestamp.minute // self.BUCKET_MINUTES) * self.BUCKET_MINUTES
        bucket = timestamp.replace(minute=minute, second=0, microsecond=0)
        bucket_str = bucket.strftime("%Y%m%d%H%M")
        return f"{self.KEY_PREFIX}:{item_id}:{bucket_str}"

    def _get_bucket_keys(self, item_id: int, now: datetime) -> list[str]:
        """Get all bucket keys within the rolling window.

        Args:
            item_id: OSRS item ID
            now: Current timestamp

        Returns:
            List of Redis keys for all buckets in the 4-hour window
        """
        keys = []
        # 16 buckets (4 hours * 4 per hour) + 1 for current partial bucket
        num_buckets = (
            int(self.WINDOW_DURATION.total_seconds() / 60 / self.BUCKET_MINUTES) + 1
        )

        for i in range(num_buckets):
            bucket_time = now - timedelta(minutes=i * self.BUCKET_MINUTES)
            keys.append(self._get_bucket_key(item_id, bucket_time))

        return keys

    def record_delivery(self, item_id: int, user_id: str) -> None:
        """Record that a recommendation was delivered to a user.

        Args:
            item_id: OSRS item ID
            user_id: User identifier (Discord user ID, session ID, etc.)
        """
        now = datetime.now(timezone.utc)
        key = self._get_bucket_key(item_id, now)

        # Use pipeline for atomic operation
        pipe = self._redis.pipeline()
        pipe.sadd(key, user_id)
        pipe.expire(key, self.TTL_SECONDS)
        pipe.execute()

        logger.debug(f"Recorded delivery to Redis: item={item_id}, user={user_id}")

    def get_concurrent_users(self, item_id: int) -> int:
        """Get number of concurrent users for an item in the rolling window.

        Args:
            item_id: OSRS item ID

        Returns:
            Number of unique users who received this recommendation recently
        """
        now = datetime.now(timezone.utc)
        keys = self._get_bucket_keys(item_id, now)

        # Filter to only existing keys to avoid errors
        existing_keys = [k for k in keys if self._redis.exists(k)]
        if not existing_keys:
            return 0

        # SUNION returns unique users across all buckets
        unique_users = self._redis.sunion(*existing_keys)
        return len(unique_users)

    def release_position(self, item_id: int, user_id: str) -> bool:
        """Release a user's position for an item when trade closes.

        Removes the user from all time buckets for this item,
        immediately freeing up crowding capacity.

        Args:
            item_id: OSRS item ID
            user_id: User identifier

        Returns:
            True if position was found and released, False otherwise
        """
        now = datetime.now(timezone.utc)
        keys = self._get_bucket_keys(item_id, now)

        # Filter to only existing keys
        existing_keys = [k for k in keys if self._redis.exists(k)]
        if not existing_keys:
            return False

        # Remove user from all buckets using pipeline
        pipe = self._redis.pipeline()
        for key in existing_keys:
            pipe.srem(key, user_id)
        results = pipe.execute()

        # Check if any removals succeeded
        released = any(r > 0 for r in results)

        if released:
            logger.debug(
                f"Released position from Redis: item={item_id}, user={user_id}"
            )

        return released

    def get_stats(self) -> dict:
        """Get current crowding statistics.

        Returns:
            Statistics dictionary with tracking info
        """
        # Use SCAN to find all crowding keys
        cursor = 0
        all_keys: set[str] = set()
        while True:
            cursor, keys = self._redis.scan(
                cursor, match=f"{self.KEY_PREFIX}:*", count=100
            )
            all_keys.update(keys)
            if cursor == 0:
                break

        # Group keys by item_id
        item_keys: dict[int, list[str]] = {}
        for key in all_keys:
            parts = key.split(":")
            if len(parts) >= 2:
                try:
                    item_id = int(parts[1])
                    if item_id not in item_keys:
                        item_keys[item_id] = []
                    item_keys[item_id].append(key)
                except ValueError:
                    continue

        # Calculate stats
        total_items = len(item_keys)
        total_deliveries = 0
        crowded_items = []

        for item_id, keys in item_keys.items():
            if keys:
                # Get unique users across all buckets for this item
                unique_users = self._redis.sunion(*keys)
                user_count = len(unique_users)

                # Sum individual set sizes for total deliveries
                item_deliveries = sum(self._redis.scard(k) for k in keys)
                total_deliveries += item_deliveries

                crowded_items.append(
                    {
                        "item_id": item_id,
                        "concurrent_users": user_count,
                        "total_deliveries": item_deliveries,
                    }
                )

        crowded_items.sort(key=lambda x: x["concurrent_users"], reverse=True)

        return {
            "tracked_items": total_items,
            "total_deliveries": total_deliveries,
            "window_hours": self.WINDOW_DURATION.total_seconds() / 3600,
            "most_crowded": crowded_items[:10],  # Top 10
            "backend": "redis",
        }

    def clear(self) -> None:
        """Clear all crowding keys from Redis.

        Useful for testing or manual resets.
        """
        cursor = 0
        deleted_count = 0
        while True:
            cursor, keys = self._redis.scan(
                cursor, match=f"{self.KEY_PREFIX}:*", count=100
            )
            if keys:
                self._redis.delete(*keys)
                deleted_count += len(keys)
            if cursor == 0:
                break

        logger.info(f"Cleared {deleted_count} crowding keys from Redis")

    def get_item_users(self, item_id: int) -> set[str]:
        """Get all unique users for an item (for debugging).

        Args:
            item_id: OSRS item ID

        Returns:
            Set of user_ids
        """
        now = datetime.now(timezone.utc)
        keys = self._get_bucket_keys(item_id, now)
        existing_keys = [k for k in keys if self._redis.exists(k)]

        if not existing_keys:
            return set()

        return self._redis.sunion(*existing_keys)
