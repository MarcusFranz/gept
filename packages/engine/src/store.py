"""Shared storage classes for recommendations and predictions."""

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional


class RecommendationStore:
    """Stores active recommendations for lookup by ID or item_id.

    Recommendations are stored with stable IDs that persist across calls
    for the same item, enabling proper rate limiting in the Discord bot.
    """

    def __init__(self, ttl_seconds: int = 900):
        """Initialize store.

        Args:
            ttl_seconds: Time-to-live for recommendations (default 15 min)
        """
        self._lock = threading.Lock()
        self.by_id: dict[str, tuple[dict, datetime]] = {}
        self.by_item_id: dict[int, str] = {}  # item_id -> recommendation id
        self.ttl = timedelta(seconds=ttl_seconds)

    def _generate_stable_id(self, item_id: int) -> str:
        """Generate a stable ID for an item.

        Uses item_id and current hour to create IDs that are stable
        within the same hour but change over time.
        """
        now = datetime.now(timezone.utc)
        hour_bucket = now.strftime("%Y%m%d%H")
        return f"rec_{item_id}_{hour_bucket}"

    def parse_rec_id(self, rec_id: str) -> Optional[int]:
        """Parse rec_id to extract item_id.

        Format: rec_{item_id}_{YYYYMMDDHH}

        Args:
            rec_id: Recommendation ID string

        Returns:
            item_id or None if invalid format
        """
        if not rec_id or not rec_id.startswith("rec_"):
            return None
        parts = rec_id.split("_")
        if len(parts) != 3:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    def store(self, recommendation: dict) -> str:
        """Store a recommendation and return its stable ID.

        Thread-safe: uses lock to prevent race conditions.

        Args:
            recommendation: Recommendation dict (will be modified with id)

        Returns:
            Stable recommendation ID
        """
        item_id = recommendation.get("itemId")
        if item_id is None:
            return recommendation.get("id", "")

        # Generate stable ID
        stable_id = self._generate_stable_id(item_id)
        recommendation["id"] = stable_id

        with self._lock:
            # Store by ID
            self.by_id[stable_id] = (recommendation, datetime.now(timezone.utc))

            # Store mapping by item_id
            self.by_item_id[item_id] = stable_id

        return stable_id

    def get_by_id(self, rec_id: str) -> Optional[dict]:
        """Get recommendation by ID.

        Thread-safe: uses lock to prevent race conditions.

        Args:
            rec_id: Recommendation ID

        Returns:
            Recommendation dict or None if not found/expired
        """
        with self._lock:
            if rec_id not in self.by_id:
                return None

            rec, timestamp = self.by_id[rec_id]
            if datetime.now(timezone.utc) - timestamp > self.ttl:
                # Expired - clean up
                item_id = rec.get("itemId")
                del self.by_id[rec_id]
                if item_id and self.by_item_id.get(item_id) == rec_id:
                    del self.by_item_id[item_id]
                return None

            return rec

    def get_by_item_id(self, item_id: int) -> Optional[dict]:
        """Get recommendation by item ID.

        Thread-safe: uses lock to prevent race conditions.

        Args:
            item_id: OSRS item ID

        Returns:
            Recommendation dict or None if not found/expired
        """
        with self._lock:
            rec_id = self.by_item_id.get(item_id)
            if rec_id is None:
                return None

            # Inline get_by_id logic to avoid nested lock acquisition
            if rec_id not in self.by_id:
                return None

            rec, timestamp = self.by_id[rec_id]
            if datetime.now(timezone.utc) - timestamp > self.ttl:
                # Expired - clean up
                item_id_from_rec = rec.get("itemId")
                del self.by_id[rec_id]
                if item_id_from_rec and self.by_item_id.get(item_id_from_rec) == rec_id:
                    del self.by_item_id[item_id_from_rec]
                return None

            return rec

    def clear(self):
        """Clear all stored recommendations.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._lock:
            self.by_id.clear()
            self.by_item_id.clear()
