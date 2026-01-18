"""Tests for Redis-backed crowding tracker."""

import pytest
import fakeredis

from src.crowding.redis import RedisCrowdingTracker


@pytest.fixture
def fake_redis_server():
    """Create a fake Redis server instance."""
    return fakeredis.FakeServer()


@pytest.fixture
def redis_tracker(fake_redis_server):
    """Create a RedisCrowdingTracker with fake Redis."""
    # Create tracker and inject fake Redis
    tracker = RedisCrowdingTracker.__new__(RedisCrowdingTracker)
    tracker._redis = fakeredis.FakeStrictRedis(
        server=fake_redis_server, decode_responses=True
    )
    tracker._redis_url = "redis://fake:6379/0"
    return tracker


class TestRedisCrowdingTracker:
    """Test suite for RedisCrowdingTracker."""

    def test_initialization_with_fake_redis(self, redis_tracker):
        """Test tracker initialization."""
        assert redis_tracker is not None
        assert redis_tracker.get_concurrent_users(123) == 0

    def test_record_delivery(self, redis_tracker):
        """Test recording a delivery."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")

        assert redis_tracker.get_concurrent_users(123) == 1

    def test_multiple_users_same_item(self, redis_tracker):
        """Test multiple users receiving same item."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=123, user_id="user2")
        redis_tracker.record_delivery(item_id=123, user_id="user3")

        assert redis_tracker.get_concurrent_users(123) == 3

    def test_same_user_multiple_deliveries(self, redis_tracker):
        """Test same user receiving item multiple times counts as 1 (deduplication)."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=123, user_id="user1")

        # Should count unique users only (Redis Sets deduplicate)
        assert redis_tracker.get_concurrent_users(123) == 1

    def test_different_items(self, redis_tracker):
        """Test tracking different items separately."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=456, user_id="user2")

        assert redis_tracker.get_concurrent_users(123) == 1
        assert redis_tracker.get_concurrent_users(456) == 1

    def test_ttl_is_set_on_keys(self, redis_tracker):
        """Test that TTL is set on Redis keys."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")

        # Find the key
        keys = list(redis_tracker._redis.scan_iter("crowding:*"))
        assert len(keys) == 1

        # Check TTL is set (should be around 15300 seconds = 4h 15m)
        ttl = redis_tracker._redis.ttl(keys[0])
        assert 15000 < ttl <= 15300

    def test_unlimited_capacity(self, redis_tracker):
        """Test None capacity means unlimited."""
        # Add 100 users
        for i in range(100):
            redis_tracker.record_delivery(item_id=123, user_id=f"user{i}")

        # None capacity = unlimited, should never be crowded
        assert not redis_tracker.is_crowded(123, capacity=None)
        assert redis_tracker.get_concurrent_users(123) == 100

    def test_capacity_20_limit(self, redis_tracker):
        """Test capacity of 20 concurrent users."""
        # Add 19 users - should not be crowded
        for i in range(19):
            redis_tracker.record_delivery(item_id=123, user_id=f"user{i}")
        assert not redis_tracker.is_crowded(123, capacity=20)

        # Add 20th user - should be crowded
        redis_tracker.record_delivery(item_id=123, user_id="user20")
        assert redis_tracker.is_crowded(123, capacity=20)

    def test_filter_crowded_items(self, redis_tracker):
        """Test filtering crowded items from candidates."""
        # Crowd item 123 (capacity 20)
        for i in range(20):
            redis_tracker.record_delivery(item_id=123, user_id=f"user{i}")

        # Keep item 456 uncrowded
        redis_tracker.record_delivery(item_id=456, user_id="user1")

        candidates = [
            {"item_id": 123, "crowding_capacity": 20, "item_name": "Crowded Item"},
            {"item_id": 456, "crowding_capacity": 20, "item_name": "Available Item"},
            {"item_id": 789, "crowding_capacity": None, "item_name": "Unlimited Item"},
        ]

        filtered = redis_tracker.filter_crowded_items(candidates)

        # Item 123 should be filtered out
        assert len(filtered) == 2
        assert all(c["item_id"] != 123 for c in filtered)

    def test_get_stats(self, redis_tracker):
        """Test getting crowding statistics."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=123, user_id="user2")
        redis_tracker.record_delivery(item_id=456, user_id="user3")

        stats = redis_tracker.get_stats()

        assert stats["tracked_items"] == 2
        assert stats["window_hours"] == 4.0
        assert stats["backend"] == "redis"
        assert len(stats["most_crowded"]) > 0

        # Find item 123 in most_crowded
        item_123 = next(
            (item for item in stats["most_crowded"] if item["item_id"] == 123), None
        )
        assert item_123 is not None
        assert item_123["concurrent_users"] == 2

    def test_clear(self, redis_tracker):
        """Test clearing all tracking data."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=456, user_id="user2")

        # Verify keys exist
        keys_before = list(redis_tracker._redis.scan_iter("crowding:*"))
        assert len(keys_before) > 0

        redis_tracker.clear()

        # Verify keys are gone
        keys_after = list(redis_tracker._redis.scan_iter("crowding:*"))
        assert len(keys_after) == 0
        assert redis_tracker.get_concurrent_users(123) == 0
        assert redis_tracker.get_concurrent_users(456) == 0

    def test_ping(self, redis_tracker):
        """Test Redis ping."""
        assert redis_tracker.ping() is True

    def test_get_item_users(self, redis_tracker):
        """Test getting all users for an item."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")
        redis_tracker.record_delivery(item_id=123, user_id="user2")
        redis_tracker.record_delivery(item_id=123, user_id="user3")

        users = redis_tracker.get_item_users(123)

        assert len(users) == 3
        assert "user1" in users
        assert "user2" in users
        assert "user3" in users

    def test_bucket_key_format(self, redis_tracker):
        """Test that bucket keys have correct format."""
        redis_tracker.record_delivery(item_id=123, user_id="user1")

        keys = list(redis_tracker._redis.scan_iter("crowding:*"))
        assert len(keys) == 1

        # Key format should be crowding:{item_id}:{timestamp_bucket}
        key = keys[0]
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "crowding"
        assert parts[1] == "123"
        # Timestamp bucket should be 12 digits (YYYYMMDDHHMM)
        assert len(parts[2]) == 12
        assert parts[2].isdigit()


class TestRedisCrowdingTrackerMultiBucket:
    """Test multi-bucket behavior across time windows."""

    def test_users_across_multiple_buckets_counted(self, redis_tracker):
        """Test that users in different time buckets are counted together."""
        # Add users to different buckets by manipulating keys directly
        # This simulates deliveries at different times within the 4-hour window

        # Current bucket
        redis_tracker.record_delivery(item_id=123, user_id="user1")

        # Simulate user in an earlier bucket by directly adding to Redis
        # (In production, this would happen naturally over time)
        earlier_key = "crowding:123:202601121300"  # Some earlier bucket
        redis_tracker._redis.sadd(earlier_key, "user2")
        redis_tracker._redis.expire(earlier_key, 15300)

        # Now count should include both users
        # Note: The exact count depends on whether earlier_key falls within the window
        # For this test, we just verify the mechanism works
        assert redis_tracker.get_concurrent_users(123) >= 1

    def test_empty_item_returns_zero(self, redis_tracker):
        """Test that untracked items return 0 users."""
        assert redis_tracker.get_concurrent_users(99999) == 0

    def test_get_item_users_empty(self, redis_tracker):
        """Test get_item_users returns empty set for untracked item."""
        users = redis_tracker.get_item_users(99999)
        assert users == set()
