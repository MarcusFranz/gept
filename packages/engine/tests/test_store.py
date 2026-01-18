"""Tests for the store module."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.store import RecommendationStore


class TestRecommendationStore:
    """Test cases for RecommendationStore."""

    def test_store_and_get_by_id(self):
        """Test basic store and retrieval by ID."""
        store = RecommendationStore(ttl_seconds=60)

        rec = {"itemId": 554, "name": "Fire rune"}
        rec_id = store.store(rec)

        assert rec_id is not None
        assert rec_id.startswith("rec_554_")

        result = store.get_by_id(rec_id)
        assert result is not None
        assert result["itemId"] == 554
        assert result["name"] == "Fire rune"
        assert result["id"] == rec_id

    def test_store_and_get_by_item_id(self):
        """Test retrieval by item ID."""
        store = RecommendationStore(ttl_seconds=60)

        rec = {"itemId": 565, "name": "Water rune"}
        store.store(rec)

        result = store.get_by_item_id(565)
        assert result is not None
        assert result["itemId"] == 565

    def test_get_missing_id_returns_none(self):
        """Test getting missing ID returns None."""
        store = RecommendationStore(ttl_seconds=60)

        result = store.get_by_id("nonexistent")
        assert result is None

    def test_get_missing_item_id_returns_none(self):
        """Test getting missing item ID returns None."""
        store = RecommendationStore(ttl_seconds=60)

        result = store.get_by_item_id(99999)
        assert result is None

    def test_expired_recommendation_returns_none(self):
        """Test that expired recommendations return None."""
        store = RecommendationStore(ttl_seconds=0)  # Immediate expiry

        rec = {"itemId": 554, "name": "Fire rune"}
        rec_id = store.store(rec)

        # Allow time to expire
        time.sleep(0.01)

        result = store.get_by_id(rec_id)
        assert result is None

    def test_clear(self):
        """Test clearing the store."""
        store = RecommendationStore(ttl_seconds=60)

        store.store({"itemId": 554, "name": "Fire rune"})
        store.store({"itemId": 565, "name": "Water rune"})
        store.clear()

        assert store.get_by_item_id(554) is None
        assert store.get_by_item_id(565) is None

    def test_parse_rec_id(self):
        """Test parsing recommendation IDs."""
        store = RecommendationStore(ttl_seconds=60)

        # Valid ID
        item_id = store.parse_rec_id("rec_554_2026011700")
        assert item_id == 554

        # Invalid formats
        assert store.parse_rec_id("") is None
        assert store.parse_rec_id("invalid") is None
        assert store.parse_rec_id("rec_") is None
        assert store.parse_rec_id("rec_abc_123") is None

    def test_store_without_item_id(self):
        """Test storing recommendation without itemId."""
        store = RecommendationStore(ttl_seconds=60)

        rec = {"name": "No item ID", "id": "existing_id"}
        rec_id = store.store(rec)

        assert rec_id == "existing_id"

    def test_stable_id_same_hour(self):
        """Test that IDs are stable within the same hour."""
        store = RecommendationStore(ttl_seconds=60)

        rec1 = {"itemId": 554, "name": "First"}
        rec2 = {"itemId": 554, "name": "Second"}

        id1 = store.store(rec1)
        id2 = store.store(rec2)

        # Same item in same hour should have same ID
        assert id1 == id2


class TestRecommendationStoreThreadSafety:
    """Test cases for thread safety of RecommendationStore."""

    def test_has_lock(self):
        """Test that store has a lock attribute."""
        store = RecommendationStore(ttl_seconds=60)
        assert hasattr(store, "_lock")
        assert isinstance(store._lock, type(threading.Lock()))

    def test_concurrent_stores(self):
        """Test concurrent store operations don't corrupt data."""
        store = RecommendationStore(ttl_seconds=60)
        num_threads = 10
        items_per_thread = 100

        def store_items(thread_id):
            for i in range(items_per_thread):
                item_id = thread_id * 1000 + i
                store.store({"itemId": item_id, "thread": thread_id, "index": i})

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(store_items, t) for t in range(num_threads)]
            for f in futures:
                f.result()

        # Verify all items are stored correctly
        for thread_id in range(num_threads):
            for i in range(items_per_thread):
                item_id = thread_id * 1000 + i
                result = store.get_by_item_id(item_id)
                assert result is not None, f"Missing item {item_id}"
                assert result["thread"] == thread_id
                assert result["index"] == i

    def test_concurrent_reads_and_writes(self):
        """Test concurrent read and write operations."""
        store = RecommendationStore(ttl_seconds=60)
        num_items = 50
        errors = []

        # Pre-populate some items
        for i in range(num_items):
            store.store({"itemId": i, "value": i})

        def writer():
            for i in range(num_items):
                try:
                    store.store({"itemId": i + num_items, "value": i})
                except Exception as e:
                    errors.append(f"Writer error: {e}")

        def reader():
            for i in range(num_items):
                try:
                    store.get_by_item_id(i)
                except Exception as e:
                    errors.append(f"Reader error: {e}")

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"

    def test_concurrent_clear(self):
        """Test clear operation is thread safe."""
        store = RecommendationStore(ttl_seconds=60)
        errors = []

        def populate_and_clear():
            try:
                for i in range(20):
                    store.store({"itemId": i, "value": i})
                store.clear()
            except Exception as e:
                errors.append(f"Error: {e}")

        threads = [threading.Thread(target=populate_and_clear) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent clear: {errors}"
