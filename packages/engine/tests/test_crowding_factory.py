"""Tests for crowding tracker factory function."""

import pytest

from src.crowding import create_crowding_tracker, InMemoryCrowdingTracker
from src.crowding.base import CrowdingTrackerBase


class TestCrowdingTrackerFactory:
    """Test suite for create_crowding_tracker factory."""

    def test_returns_memory_tracker_when_no_redis_url(self):
        """Test that factory returns in-memory tracker when no Redis URL provided."""
        tracker = create_crowding_tracker(redis_url=None)

        assert isinstance(tracker, InMemoryCrowdingTracker)
        assert isinstance(tracker, CrowdingTrackerBase)

    def test_returns_memory_tracker_for_empty_redis_url(self):
        """Test that factory returns in-memory tracker for empty Redis URL."""
        tracker = create_crowding_tracker(redis_url="")

        assert isinstance(tracker, InMemoryCrowdingTracker)

    def test_fallback_to_memory_on_connection_error(self):
        """Test fallback to memory when Redis connection fails."""
        # Use an invalid Redis URL that will fail to connect
        tracker = create_crowding_tracker(
            redis_url="redis://nonexistent-host:6379/0",
            fallback_to_memory=True,
        )

        # Should fall back to in-memory tracker
        assert isinstance(tracker, InMemoryCrowdingTracker)

    def test_raises_when_fallback_disabled(self):
        """Test that exception is raised when Redis fails and fallback disabled."""
        with pytest.raises(Exception):
            create_crowding_tracker(
                redis_url="redis://nonexistent-host:6379/0",
                fallback_to_memory=False,
            )

    def test_memory_tracker_has_correct_interface(self):
        """Test that returned tracker has correct interface."""
        tracker = create_crowding_tracker(redis_url=None)

        # Test all expected methods exist
        assert hasattr(tracker, "record_delivery")
        assert hasattr(tracker, "get_concurrent_users")
        assert hasattr(tracker, "is_crowded")
        assert hasattr(tracker, "filter_crowded_items")
        assert hasattr(tracker, "get_stats")
        assert hasattr(tracker, "clear")
        assert hasattr(tracker, "cleanup_all")

        # Test methods are callable
        tracker.record_delivery(item_id=123, user_id="test_user")
        assert tracker.get_concurrent_users(123) == 1
        assert not tracker.is_crowded(123, capacity=10)

        stats = tracker.get_stats()
        assert "backend" in stats
        assert stats["backend"] == "memory"

    def test_default_fallback_is_true(self):
        """Test that fallback_to_memory defaults to True."""
        # Should not raise even with invalid URL because fallback is True by default
        tracker = create_crowding_tracker(
            redis_url="redis://nonexistent:6379/0"
            # fallback_to_memory not specified, defaults to True
        )

        assert isinstance(tracker, InMemoryCrowdingTracker)


class TestCrowdingTrackerInterfaceConsistency:
    """Test that all implementations behave consistently."""

    def test_memory_tracker_stats_include_backend(self):
        """Test that memory tracker includes backend in stats."""
        tracker = create_crowding_tracker(redis_url=None)
        stats = tracker.get_stats()

        assert "backend" in stats
        assert stats["backend"] == "memory"

    def test_window_duration_is_4_hours(self):
        """Test that window duration is consistently 4 hours."""
        tracker = create_crowding_tracker(redis_url=None)

        stats = tracker.get_stats()
        assert stats["window_hours"] == 4.0

    def test_filter_crowded_items_interface(self):
        """Test filter_crowded_items works correctly."""
        tracker = create_crowding_tracker(redis_url=None)

        # Crowd item 123
        for i in range(20):
            tracker.record_delivery(item_id=123, user_id=f"user{i}")

        candidates = [
            {"item_id": 123, "crowding_capacity": 20},
            {"item_id": 456, "crowding_capacity": 20},
        ]

        filtered = tracker.filter_crowded_items(candidates)

        assert len(filtered) == 1
        assert filtered[0]["item_id"] == 456
