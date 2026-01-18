"""Tests for crowding tracker functionality."""

from datetime import datetime, timedelta, timezone

from src.crowding import InMemoryCrowdingTracker


class TestInMemoryCrowdingTracker:
    """Test suite for InMemoryCrowdingTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = InMemoryCrowdingTracker()
        assert tracker is not None
        assert tracker.get_concurrent_users(123) == 0

    def test_record_delivery(self):
        """Test recording a delivery."""
        tracker = InMemoryCrowdingTracker()
        tracker.record_delivery(item_id=123, user_id="user1")

        assert tracker.get_concurrent_users(123) == 1

    def test_multiple_users_same_item(self):
        """Test multiple users receiving same item."""
        tracker = InMemoryCrowdingTracker()

        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=123, user_id="user2")
        tracker.record_delivery(item_id=123, user_id="user3")

        assert tracker.get_concurrent_users(123) == 3

    def test_same_user_multiple_deliveries(self):
        """Test same user receiving item multiple times (counts as 1)."""
        tracker = InMemoryCrowdingTracker()

        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=123, user_id="user1")

        # Should count unique users only
        assert tracker.get_concurrent_users(123) == 1

    def test_different_items(self):
        """Test tracking different items separately."""
        tracker = InMemoryCrowdingTracker()

        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=456, user_id="user2")

        assert tracker.get_concurrent_users(123) == 1
        assert tracker.get_concurrent_users(456) == 1

    def test_unlimited_capacity(self):
        """Test None capacity means unlimited."""
        tracker = InMemoryCrowdingTracker()

        # Add 100 users
        for i in range(100):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")

        # None capacity = unlimited, should never be crowded
        assert not tracker.is_crowded(123, capacity=None)
        assert tracker.get_concurrent_users(123) == 100

    def test_capacity_50_limit(self):
        """Test capacity of 50 concurrent users."""
        tracker = InMemoryCrowdingTracker()

        # Add 49 users - should not be crowded
        for i in range(49):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")
        assert not tracker.is_crowded(123, capacity=50)

        # Add 50th user - should be crowded
        tracker.record_delivery(item_id=123, user_id="user50")
        assert tracker.is_crowded(123, capacity=50)

    def test_capacity_20_limit(self):
        """Test capacity of 20 concurrent users."""
        tracker = InMemoryCrowdingTracker()

        # Add 19 users - should not be crowded
        for i in range(19):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")
        assert not tracker.is_crowded(123, capacity=20)

        # Add 20th user - should be crowded
        tracker.record_delivery(item_id=123, user_id="user20")
        assert tracker.is_crowded(123, capacity=20)

    def test_capacity_10_limit(self):
        """Test capacity of 10 concurrent users."""
        tracker = InMemoryCrowdingTracker()

        # Add 9 users - should not be crowded
        for i in range(9):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")
        assert not tracker.is_crowded(123, capacity=10)

        # Add 10th user - should be crowded
        tracker.record_delivery(item_id=123, user_id="user10")
        assert tracker.is_crowded(123, capacity=10)

    def test_filter_crowded_items(self):
        """Test filtering crowded items from candidates."""
        tracker = InMemoryCrowdingTracker()

        # Crowd item 123 (capacity 20)
        for i in range(20):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")

        # Keep item 456 uncrowded
        tracker.record_delivery(item_id=456, user_id="user1")

        candidates = [
            {"item_id": 123, "crowding_capacity": 20, "item_name": "Crowded Item"},
            {"item_id": 456, "crowding_capacity": 20, "item_name": "Available Item"},
            {"item_id": 789, "crowding_capacity": None, "item_name": "Unlimited Item"},
        ]

        filtered = tracker.filter_crowded_items(candidates)

        # Item 123 should be filtered out
        assert len(filtered) == 2
        assert all(c["item_id"] != 123 for c in filtered)

    def test_cleanup_old_deliveries(self):
        """Test that old deliveries are cleaned up (mocked time)."""
        tracker = InMemoryCrowdingTracker()

        # Record delivery
        tracker.record_delivery(item_id=123, user_id="user1")
        assert tracker.get_concurrent_users(123) == 1

        # Manually manipulate the timestamp to be 5 hours old
        five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
        tracker._deliveries[123][0] = ("user1", five_hours_ago)

        # Cleanup should remove it (4-hour window)
        assert tracker.get_concurrent_users(123) == 0

    def test_cleanup_all(self):
        """Test cleanup_all removes old deliveries."""
        tracker = InMemoryCrowdingTracker()

        # Add some deliveries
        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=456, user_id="user2")

        # Manually age them
        five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
        tracker._deliveries[123][0] = ("user1", five_hours_ago)
        tracker._deliveries[456][0] = ("user2", five_hours_ago)

        # Cleanup should remove old entries
        tracker.cleanup_all()
        assert 123 not in tracker._deliveries
        assert 456 not in tracker._deliveries

    def test_get_stats(self):
        """Test getting crowding statistics."""
        tracker = InMemoryCrowdingTracker()

        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=123, user_id="user2")
        tracker.record_delivery(item_id=456, user_id="user3")

        stats = tracker.get_stats()

        assert stats["tracked_items"] == 2
        assert stats["total_deliveries"] == 3
        assert stats["window_hours"] == 4.0
        assert stats["backend"] == "memory"
        assert len(stats["most_crowded"]) > 0
        assert stats["most_crowded"][0]["item_id"] == 123
        assert stats["most_crowded"][0]["concurrent_users"] == 2

    def test_clear(self):
        """Test clearing all tracking data."""
        tracker = InMemoryCrowdingTracker()

        tracker.record_delivery(item_id=123, user_id="user1")
        tracker.record_delivery(item_id=456, user_id="user2")

        tracker.clear()

        assert tracker.get_concurrent_users(123) == 0
        assert tracker.get_concurrent_users(456) == 0
        assert len(tracker._deliveries) == 0

    def test_missing_item_id_in_candidates(self):
        """Test that candidates with missing item_id are kept (fail open)."""
        tracker = InMemoryCrowdingTracker()

        candidates = [
            {"item_id": None, "crowding_capacity": 20},  # Missing item_id
            {"item_id": 456, "crowding_capacity": 20},  # Valid
            {"item_id": 789, "crowding_capacity": None},  # Unlimited
        ]

        filtered = tracker.filter_crowded_items(candidates)

        # All should be kept (fail open for missing item_id)
        assert len(filtered) == 3

    def test_missing_capacity_defaults_to_unlimited(self):
        """Test that missing crowding_capacity defaults to unlimited (not crowded)."""
        tracker = InMemoryCrowdingTracker()

        # Crowd item 123 with many users
        for i in range(100):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")

        candidates = [
            {"item_id": 123, "item_name": "No capacity key"},  # Missing capacity
        ]

        filtered = tracker.filter_crowded_items(candidates)

        # Should be kept since capacity is None (defaults to unlimited)
        assert len(filtered) == 1

    def test_custom_capacity_key(self):
        """Test using a custom capacity key."""
        tracker = InMemoryCrowdingTracker()

        # Crowd item 123
        for i in range(20):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")

        candidates = [
            {"item_id": 123, "custom_capacity": 20, "item_name": "Crowded"},
            {"item_id": 456, "custom_capacity": 20, "item_name": "Available"},
        ]

        tracker.record_delivery(item_id=456, user_id="user1")

        filtered = tracker.filter_crowded_items(
            candidates, capacity_key="custom_capacity"
        )

        # Item 123 should be filtered out
        assert len(filtered) == 1
        assert filtered[0]["item_id"] == 456

    def test_enforce_max_items(self):
        """Test that max items limit is enforced."""
        from src.crowding.memory import MAX_TRACKED_ITEMS

        tracker = InMemoryCrowdingTracker()

        # Add more items than the limit
        for i in range(MAX_TRACKED_ITEMS + 10):
            tracker.record_delivery(item_id=i, user_id=f"user{i}")

        # Run cleanup to enforce limit
        tracker.cleanup_all()

        # Should be at or below the limit
        assert len(tracker._deliveries) <= MAX_TRACKED_ITEMS

    def test_cleanup_all_removes_expired_and_enforces_limit(self):
        """Test cleanup_all handles both expiration and limit enforcement."""
        from src.crowding.memory import MAX_TRACKED_ITEMS

        tracker = InMemoryCrowdingTracker()

        # Add some fresh items
        for i in range(100):
            tracker.record_delivery(item_id=i, user_id=f"user{i}")

        # Add some old items (5 hours old - beyond 4h window)
        five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
        for i in range(100, 200):
            tracker._deliveries[i] = [(f"user{i}", five_hours_ago)]

        # Run cleanup
        tracker.cleanup_all()

        # Old items should be removed
        for i in range(100, 200):
            assert i not in tracker._deliveries, f"Item {i} should be removed (expired)"

        # Fresh items should remain (up to limit)
        assert len(tracker._deliveries) <= MAX_TRACKED_ITEMS
        assert len(tracker._deliveries) > 0  # At least some fresh items remain
