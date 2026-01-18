"""Abstract base class for crowding trackers.

Defines the interface that all crowding tracker implementations must follow.
"""

import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class CrowdingTrackerBase(ABC):
    """Abstract base class for crowding trackers.

    Crowding capacity is determined by actual 1-hour trade volume:
    - > 50,000 volume: Unlimited (None)
    - > 10,000 volume: 50 concurrent users
    - > 1,000 volume: 20 concurrent users
    - <= 1,000 volume: 10 concurrent users

    Rolling window: 4 hours
    """

    # Rolling window duration
    WINDOW_DURATION = timedelta(hours=4)

    @abstractmethod
    def record_delivery(self, item_id: int, user_id: str) -> None:
        """Record that a recommendation was delivered to a user.

        Args:
            item_id: OSRS item ID
            user_id: User identifier (Discord user ID, session ID, etc.)
        """
        pass

    @abstractmethod
    def get_concurrent_users(self, item_id: int) -> int:
        """Get number of concurrent users for an item in the rolling window.

        Args:
            item_id: OSRS item ID

        Returns:
            Number of unique users who received this recommendation recently
        """
        pass

    def is_crowded(self, item_id: int, capacity: Optional[int] = None) -> bool:
        """Check if an item has hit its crowding limit.

        Args:
            item_id: OSRS item ID
            capacity: Maximum concurrent users (None = unlimited)

        Returns:
            True if item is crowded (at or above limit), False otherwise
        """
        # None capacity means unlimited
        if capacity is None:
            return False

        concurrent = self.get_concurrent_users(item_id)
        is_at_limit = concurrent >= capacity

        if is_at_limit:
            logger.info(
                f"Item {item_id} is crowded: {concurrent} users (limit={capacity})"
            )

        return is_at_limit

    def filter_crowded_items(
        self, candidates: list[dict], capacity_key: str = "crowding_capacity"
    ) -> list[dict]:
        """Filter out crowded items from a list of candidates.

        Args:
            candidates: List of candidate recommendations with item_id and crowding_capacity
            capacity_key: Key to access crowding capacity in candidate dict

        Returns:
            Filtered list excluding crowded items
        """
        filtered = []
        excluded_count = 0

        for candidate in candidates:
            item_id = candidate.get("item_id")
            capacity = candidate.get(capacity_key)

            if item_id is None:
                # Keep if missing item_id (fail open)
                filtered.append(candidate)
                continue

            if not self.is_crowded(item_id, capacity):
                filtered.append(candidate)
            else:
                excluded_count += 1

        if excluded_count > 0:
            logger.info(
                f"Filtered {excluded_count} crowded items "
                f"from {len(candidates)} candidates"
            )

        return filtered

    @abstractmethod
    def release_position(self, item_id: int, user_id: str) -> bool:
        """Release a user's position for an item when trade closes.

        This allows immediate release of crowding capacity when a trade
        completes or is cancelled, rather than waiting for TTL expiry.

        Args:
            item_id: OSRS item ID
            user_id: User identifier

        Returns:
            True if position was found and released, False otherwise
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get current crowding statistics.

        Returns:
            Statistics dictionary with tracking info including:
            - tracked_items: Number of items being tracked
            - total_deliveries: Total delivery records
            - window_hours: Rolling window duration in hours
            - most_crowded: List of most crowded items
            - backend: Implementation type ("memory" or "redis")
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all tracking data.

        Useful for testing or manual resets.
        """
        pass

    def cleanup_all(self) -> None:
        """Clean up old deliveries.

        Default implementation is a no-op. Redis uses TTL for automatic cleanup.
        In-memory implementation overrides this for manual cleanup.
        """
        pass
