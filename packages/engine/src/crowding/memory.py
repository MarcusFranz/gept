"""In-memory crowding tracker implementation.

Uses a local defaultdict for tracking - suitable for single-instance deployments
or local development. Data is lost on restart.
"""

import logging
from collections import defaultdict
from datetime import datetime, timezone

from .base import CrowdingTrackerBase

logger = logging.getLogger(__name__)

# Maximum number of items to track (prevents unbounded memory growth)
# Higher than typical item count (168) to allow for growth
MAX_TRACKED_ITEMS = 500


class InMemoryCrowdingTracker(CrowdingTrackerBase):
    """In-memory implementation of crowding tracker.

    Uses a local defaultdict for storage. Suitable for:
    - Single-instance deployments
    - Local development
    - Testing

    Note: Data is lost on restart and not shared across instances.
    """

    def __init__(self):
        """Initialize in-memory crowding tracker."""
        # Track recommendation deliveries: item_id -> [(user_id, timestamp), ...]
        self._deliveries: dict[int, list[tuple[str, datetime]]] = defaultdict(list)

    def record_delivery(self, item_id: int, user_id: str) -> None:
        """Record that a recommendation was delivered to a user.

        Args:
            item_id: OSRS item ID
            user_id: User identifier (Discord user ID, session ID, etc.)
        """
        now = datetime.now(timezone.utc)

        # Clean up old entries for this item first
        self._cleanup_old_deliveries(item_id, now)

        # Add new delivery
        self._deliveries[item_id].append((user_id, now))

        concurrent = len(self._deliveries[item_id])
        logger.debug(
            f"Recorded delivery: item={item_id}, user={user_id}, "
            f"concurrent={concurrent}"
        )

    def get_concurrent_users(self, item_id: int) -> int:
        """Get number of concurrent users for an item in the rolling window.

        Args:
            item_id: OSRS item ID

        Returns:
            Number of unique users who received this recommendation recently
        """
        now = datetime.now(timezone.utc)
        self._cleanup_old_deliveries(item_id, now)

        # Count unique users (use .get() to avoid creating entries in defaultdict)
        deliveries = self._deliveries.get(item_id, [])
        unique_users = {user_id for user_id, _ in deliveries}
        return len(unique_users)

    def release_position(self, item_id: int, user_id: str) -> bool:
        """Release a user's position for an item when trade closes.

        Removes all delivery records for this user/item pair, immediately
        freeing up crowding capacity.

        Args:
            item_id: OSRS item ID
            user_id: User identifier

        Returns:
            True if position was found and released, False otherwise
        """
        if item_id not in self._deliveries:
            return False

        original_count = len(self._deliveries[item_id])

        # Remove all entries for this user on this item
        self._deliveries[item_id] = [
            (uid, ts) for uid, ts in self._deliveries[item_id] if uid != user_id
        ]

        # Clean up empty entries
        if not self._deliveries[item_id]:
            del self._deliveries[item_id]

        released = len(self._deliveries.get(item_id, [])) < original_count

        if released:
            logger.debug(f"Released position: item={item_id}, user={user_id}")

        return released

    def _cleanup_old_deliveries(self, item_id: int, current_time: datetime) -> None:
        """Remove deliveries outside the rolling window.

        Args:
            item_id: OSRS item ID
            current_time: Current timestamp
        """
        if item_id not in self._deliveries:
            return

        cutoff_time = current_time - self.WINDOW_DURATION

        # Keep only deliveries within the window
        self._deliveries[item_id] = [
            (user_id, ts)
            for user_id, ts in self._deliveries[item_id]
            if ts > cutoff_time
        ]

        # Remove empty entries
        if not self._deliveries[item_id]:
            del self._deliveries[item_id]

    def _enforce_max_items(self) -> None:
        """Evict oldest items if over the maximum limit.

        Uses LRU-style eviction based on the most recent delivery timestamp
        for each item. This prevents unbounded memory growth.
        """
        if len(self._deliveries) <= MAX_TRACKED_ITEMS:
            return

        # Sort items by their most recent delivery timestamp (oldest first)
        items_by_age = sorted(
            self._deliveries.items(),
            key=lambda x: (
                max(ts for _, ts in x[1])
                if x[1]
                else datetime.min.replace(tzinfo=timezone.utc)
            ),
        )

        evicted_count = 0
        while len(self._deliveries) > MAX_TRACKED_ITEMS:
            oldest_item = items_by_age.pop(0)[0]
            del self._deliveries[oldest_item]
            evicted_count += 1

        if evicted_count > 0:
            logger.warning(
                f"Evicted {evicted_count} oldest items due to memory limit "
                f"(max={MAX_TRACKED_ITEMS})"
            )

    def cleanup_all(self) -> None:
        """Clean up old deliveries for all items.

        Should be called periodically to prevent memory leaks.
        Also enforces maximum item limit via LRU eviction.
        """
        now = datetime.now(timezone.utc)
        items_to_check = list(self._deliveries.keys())

        for item_id in items_to_check:
            self._cleanup_old_deliveries(item_id, now)

        # Enforce maximum items limit
        self._enforce_max_items()

        logger.debug(f"Cleanup complete: tracking {len(self._deliveries)} items")

    def get_stats(self) -> dict:
        """Get current crowding statistics.

        Returns:
            Statistics dictionary with tracking info
        """
        now = datetime.now(timezone.utc)

        # Cleanup first
        items_to_check = list(self._deliveries.keys())
        for item_id in items_to_check:
            self._cleanup_old_deliveries(item_id, now)

        total_items = len(self._deliveries)
        total_deliveries = sum(
            len(deliveries) for deliveries in self._deliveries.values()
        )

        # Find most crowded items
        crowded_items = []
        for item_id, deliveries in self._deliveries.items():
            unique_users = len({user_id for user_id, _ in deliveries})
            if unique_users > 0:
                crowded_items.append(
                    {
                        "item_id": item_id,
                        "concurrent_users": unique_users,
                        "total_deliveries": len(deliveries),
                    }
                )

        crowded_items.sort(key=lambda x: x["concurrent_users"], reverse=True)

        return {
            "tracked_items": total_items,
            "max_tracked_items": MAX_TRACKED_ITEMS,
            "total_deliveries": total_deliveries,
            "window_hours": self.WINDOW_DURATION.total_seconds() / 3600,
            "most_crowded": crowded_items[:10],  # Top 10
            "backend": "memory",
        }

    def clear(self) -> None:
        """Clear all tracking data.

        Useful for testing or manual resets.
        """
        self._deliveries.clear()
        logger.info("Cleared all crowding tracking data")
