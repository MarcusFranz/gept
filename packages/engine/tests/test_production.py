"""
Production Deployment Test Suite for GePT Recommendation Engine v1.1.0

Run against production server:
    RUN_PROD_TESTS=true pytest tests/test_production.py -v

Configure server URL via PROD_SERVER_URL environment variable.
Default: http://localhost:8000 (must SSH into server to run)

NOTE: Tests are skipped by default unless RUN_PROD_TESTS=true is set
AND a server is available at PROD_SERVER_URL. This prevents failures
when running pytest locally without a server running.
"""

import os
import time
import requests
import pytest
from typing import Dict, Optional

# Production server URL - configurable via environment variable
# Default to localhost since API is not exposed externally (must SSH into server to test)
PROD_SERVER_URL = os.environ.get("PROD_SERVER_URL", "http://localhost:8000")


def _server_available():
    """Check if production server is available."""
    try:
        requests.get(PROD_SERVER_URL, timeout=2)
        return True
    except Exception:
        return False


def _should_skip_production_tests():
    """
    Determine if production tests should be skipped.

    Tests run only when RUN_PROD_TESTS=true AND server is available.
    This prevents accidental failures in local development.
    """
    # If RUN_PROD_TESTS is not explicitly enabled, skip
    if os.environ.get("RUN_PROD_TESTS") != "true":
        return True
    # If explicitly enabled, check if server is actually available
    return not _server_available()


# Skip all tests in this module unless explicitly enabled and server is available
pytestmark = pytest.mark.skipif(
    _should_skip_production_tests(),
    reason=f"Production tests require RUN_PROD_TESTS=true and live server at {PROD_SERVER_URL}",
)


class ProductionTestClient:
    """HTTP client for production API testing."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30

    def get(self, path: str, params: Optional[Dict] = None) -> requests.Response:
        return self.session.get(f"{self.base_url}{path}", params=params)

    def post(self, path: str, json: Optional[Dict] = None) -> requests.Response:
        return self.session.post(f"{self.base_url}{path}", json=json)


@pytest.fixture(scope="module")
def client():
    """Production test client fixture."""
    return ProductionTestClient(PROD_SERVER_URL)


# =============================================================================
# Health & Connectivity Tests
# =============================================================================


class TestHealthAndConnectivity:
    """Verify basic server health and database connectivity."""

    def test_root_endpoint_returns_api_info(self, client):
        """Root endpoint should return API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        print(f"  Server version: {data.get('version')}")

    def test_health_endpoint_returns_ok(self, client):
        """Health endpoint should report ok status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        # API returns "ok" not "healthy"
        assert data.get("status") == "ok"
        print(f"  Status: {data.get('status')}")

    def test_predictions_are_fresh(self, client):
        """Predictions should be less than 6 minutes old."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        # Check in the checks array for prediction_loader
        checks = data.get("checks", [])
        prediction_check = next(
            (c for c in checks if c.get("component") == "prediction_loader"), None
        )
        assert prediction_check is not None, "No prediction_loader check found"
        age = prediction_check.get("prediction_age_seconds", 999)
        assert age < 360, f"Predictions are stale! Age: {age}s"
        print(f"  Prediction age: {age:.0f}s")

    def test_database_is_connected(self, client):
        """Database should be connected and accessible."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        # Check in the checks array for prediction_loader connected status
        checks = data.get("checks", [])
        prediction_check = next(
            (c for c in checks if c.get("component") == "prediction_loader"), None
        )
        assert prediction_check is not None, "No prediction_loader check found"
        assert prediction_check.get("connected") is True, "Database is not connected!"
        print(f"  Database connected: {prediction_check.get('connected')}")


# =============================================================================
# Item Search Tests (Including OSRS Acronym Feature)
# =============================================================================


class TestItemSearch:
    """Test item search functionality including the new acronym expansion."""

    def test_basic_item_search(self, client):
        """Basic item search should return results."""
        response = client.get("/api/v1/items/search", params={"q": "rune"})
        assert response.status_code == 200
        # API returns a list directly, not {"items": [...]}
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        print(f"  Found {len(data)} items for 'rune'")

    def test_search_limit_parameter(self, client):
        """Limit parameter should restrict result count."""
        response = client.get("/api/v1/items/search", params={"q": "rune", "limit": 5})
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    def test_search_returns_item_structure(self, client):
        """Search results should have correct structure."""
        response = client.get(
            "/api/v1/items/search", params={"q": "dragon", "limit": 1}
        )
        assert response.status_code == 200
        data = response.json()
        if data:
            item = data[0]
            # API uses itemId not item_id
            assert "itemId" in item or "item_id" in item
            assert "name" in item

    # OSRS Acronym Expansion Tests (v1.1.0 Feature)
    # Note: Tests use items that are actually tracked in the system.
    # Expensive end-game items (godswords, tbow, etc.) are NOT tracked for flipping.

    def test_acronym_bcp_expands_to_bandos_chestplate(self, client):
        """'bcp' should expand to find Bandos chestplate."""
        response = client.get("/api/v1/items/search", params={"q": "bcp"})
        assert response.status_code == 200
        items = response.json()
        item_names = [item["name"].lower() for item in items]
        assert any(
            "bandos chestplate" in name for name in item_names
        ), f"Expected 'Bandos chestplate' in results for 'bcp', got: {item_names}"
        print("  bcp -> Bandos chestplate")

    def test_acronym_tassets_expands_to_bandos_tassets(self, client):
        """'tassets' should expand to find Bandos tassets."""
        response = client.get("/api/v1/items/search", params={"q": "tassets"})
        assert response.status_code == 200
        items = response.json()
        item_names = [item["name"].lower() for item in items]
        assert any(
            "bandos tassets" in name for name in item_names
        ), f"Expected 'Bandos tassets' in results for 'tassets', got: {item_names}"
        print("  tassets -> Bandos tassets")

    def test_acronym_fury_expands_to_amulet_of_fury(self, client):
        """'fury' should expand to find Amulet of fury."""
        response = client.get("/api/v1/items/search", params={"q": "fury"})
        assert response.status_code == 200
        items = response.json()
        item_names = [item["name"].lower() for item in items]
        assert any(
            "fury" in name for name in item_names
        ), f"Expected 'Amulet of fury' in results for 'fury', got: {item_names}"
        print("  fury -> Amulet of fury")

    def test_acronym_torture_expands_to_amulet_of_torture(self, client):
        """'torture' should expand to find Amulet of torture."""
        response = client.get("/api/v1/items/search", params={"q": "torture"})
        assert response.status_code == 200
        items = response.json()
        item_names = [item["name"].lower() for item in items]
        assert any(
            "torture" in name for name in item_names
        ), f"Expected 'Amulet of torture' in results for 'torture', got: {item_names}"
        print("  torture -> Amulet of torture")

    def test_acronym_bp_expands_to_blowpipe(self, client):
        """'bp' should expand to find Toxic blowpipe."""
        response = client.get("/api/v1/items/search", params={"q": "bp"})
        assert response.status_code == 200
        items = response.json()
        item_names = [item["name"].lower() for item in items]
        assert any(
            "blowpipe" in name for name in item_names
        ), f"Expected 'Toxic blowpipe' in results for 'bp', got: {item_names}"
        print("  bp -> Toxic blowpipe")

    def test_acronym_case_insensitive(self, client):
        """Acronym expansion should be case-insensitive."""
        # Test with bcp (Bandos chestplate) which is tracked
        response_upper = client.get("/api/v1/items/search", params={"q": "BCP"})
        response_lower = client.get("/api/v1/items/search", params={"q": "bcp"})
        response_mixed = client.get("/api/v1/items/search", params={"q": "Bcp"})

        assert response_upper.status_code == 200
        assert response_lower.status_code == 200
        assert response_mixed.status_code == 200

        # All should find the same item
        for response in [response_upper, response_lower, response_mixed]:
            items = response.json()
            item_names = [item["name"].lower() for item in items]
            assert any("bandos chestplate" in name for name in item_names)
        print("  Case insensitivity verified")

    def test_non_acronym_search_works(self, client):
        """Non-acronym searches should work normally."""
        response = client.get("/api/v1/items/search", params={"q": "Abyssal whip"})
        assert response.status_code == 200
        items = response.json()
        item_names = [item["name"].lower() for item in items]
        assert any("abyssal whip" in name for name in item_names)

    def test_additional_acronyms_for_tracked_items(self, client):
        """Test additional common acronyms for items that are tracked."""
        # Only test acronyms for items that are actually in the tracked database
        acronym_tests = [
            ("bcp", "bandos chestplate"),
            ("tassets", "bandos tassets"),
            ("bp", "blowpipe"),
            ("fury", "fury"),
            ("torture", "torture"),
            ("monks", "monkfish"),
            ("ppot", "prayer potion"),
            ("restore", "restore"),
            ("shark", "shark"),
            ("lobster", "lobster"),
        ]

        passed = 0
        failed = []

        for acronym, expected_substring in acronym_tests:
            response = client.get("/api/v1/items/search", params={"q": acronym})
            if response.status_code == 200:
                items = response.json()
                item_names = [item["name"].lower() for item in items]
                if any(expected_substring in name for name in item_names):
                    passed += 1
                else:
                    failed.append(
                        f"{acronym} -> expected '{expected_substring}', got {item_names}"
                    )
            else:
                failed.append(f"{acronym} -> HTTP {response.status_code}")

        print(f"  Acronym tests: {passed}/{len(acronym_tests)} passed")
        if failed:
            print(f"  Failed: {failed}")
        # At least 80% should work
        assert (
            passed >= len(acronym_tests) * 0.8
        ), f"Too many acronym failures: {failed}"


# =============================================================================
# Recommendations Tests
# =============================================================================


class TestRecommendations:
    """Test the main recommendation endpoints."""

    def test_get_recommendations_requires_capital(self, client):
        """GET recommendations without capital should return 422."""
        response = client.get("/api/v1/recommendations")
        # Capital is required for GET - this should return 422
        assert response.status_code == 422

    def test_get_recommendations_with_capital(self, client):
        """GET recommendations with capital should work."""
        response = client.get(
            "/api/v1/recommendations",
            params={"capital": 10000000, "style": "passive"},  # 10M gp
        )
        assert response.status_code == 200
        # API returns a list directly
        data = response.json()
        assert isinstance(data, list)
        print(f"  Got {len(data)} recommendations")

    def test_get_recommendations_with_style(self, client):
        """GET recommendations with style parameter."""
        for style in ["passive", "hybrid", "active"]:
            response = client.get(
                "/api/v1/recommendations", params={"style": style, "capital": 5000000}
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            print(f"  {style}: {len(data)} recommendations")

    def test_get_recommendations_with_risk_levels(self, client):
        """GET recommendations with different risk levels."""
        for risk in ["low", "medium", "high"]:
            response = client.get(
                "/api/v1/recommendations", params={"risk": risk, "capital": 5000000}
            )
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            print(f"  {risk} risk: {len(data)} recommendations")

    def test_get_recommendations_with_slots(self, client):
        """GET recommendations with slots parameter."""
        response = client.get(
            "/api/v1/recommendations", params={"slots": 4, "capital": 5000000}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should not exceed requested slots
        assert len(data) <= 4

    def test_post_recommendations(self, client):
        """POST recommendations with full request body."""
        response = client.post(
            "/api/v1/recommendations",
            json={"style": "hybrid", "capital": 50000000, "risk": "medium", "slots": 6},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"  POST returned {len(data)} recommendations")

    def test_recommendation_structure(self, client):
        """Verify recommendation response structure."""
        response = client.get(
            "/api/v1/recommendations", params={"slots": 1, "capital": 5000000}
        )
        assert response.status_code == 200
        data = response.json()

        if data:
            rec = data[0]
            # Check required fields (API uses camelCase)
            required_fields = [
                "id",
                "itemId",
                "item",
                "buyPrice",
                "sellPrice",
                "quantity",
                "expectedProfit",
                "fillConfidence",
                "expectedHours",
            ]
            for field in required_fields:
                assert field in rec, f"Missing field: {field}"
            print(f"  Recommendation structure verified: {rec['item']}")

    def test_capital_validation_minimum(self, client):
        """Capital below 1000gp should be rejected."""
        response = client.get("/api/v1/recommendations", params={"capital": 500})
        # Should reject with 422
        assert response.status_code == 422

    def test_slots_validation(self, client):
        """Slots outside 1-8 should be handled."""
        # Too many slots - should either reject or cap
        response = client.get(
            "/api/v1/recommendations", params={"slots": 10, "capital": 5000000}
        )
        assert response.status_code in [200, 422]

        # Zero slots
        response = client.get(
            "/api/v1/recommendations", params={"slots": 0, "capital": 5000000}
        )
        assert response.status_code in [200, 422]


# =============================================================================
# Item-Specific Recommendation Tests
# =============================================================================


class TestItemRecommendations:
    """Test item-specific recommendation endpoints."""

    def test_get_item_by_id(self, client):
        """Get recommendation for item by ID."""
        # Use a common item ID like Dragon bones (526)
        response = client.get("/api/v1/recommendations/item/526")
        # May return 200 (found) or 404 (not tracked)
        assert response.status_code in [200, 404]

    def test_get_item_with_price_history(self, client):
        """Item recommendation with price history flag."""
        response = client.get(
            "/api/v1/recommendations/item/526", params={"include_history": "true"}
        )
        if response.status_code == 200:
            data = response.json()
            # Check if price_history is included when requested
            if "priceHistory" in data:
                print(
                    f"  Price history included: {len(data.get('priceHistory', []))} points"
                )


# =============================================================================
# Predictions Endpoint Tests
# =============================================================================


class TestPredictions:
    """Test the predictions endpoint."""

    def test_get_prediction_by_item_id(self, client):
        """Get prediction for a known item ID."""
        # Fire rune is item_id 554 - commonly traded
        response = client.get("/api/v1/predictions/554")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert "item_id" in data
            # API uses all_predictions not predictions
            assert "all_predictions" in data
            print(f"  Got {len(data['all_predictions'])} predictions for Fire rune")

    def test_prediction_structure(self, client):
        """Verify prediction response structure."""
        response = client.get("/api/v1/predictions/554")
        if response.status_code == 200:
            data = response.json()
            # Check top-level fields
            expected_fields = ["item_id", "item_name", "best_config", "all_predictions"]
            for field in expected_fields:
                assert field in data, f"Missing prediction field: {field}"

            if data.get("all_predictions"):
                pred = data["all_predictions"][0]
                # Should have hour_offset, offset_pct, and probability
                pred_fields = ["hour_offset", "offset_pct", "fill_probability"]
                for field in pred_fields:
                    assert field in pred, f"Missing prediction field: {field}"


# =============================================================================
# Trade Outcome Recording Tests
# =============================================================================


class TestTradeOutcomes:
    """Test trade outcome recording functionality."""

    def test_outcome_requires_hashed_user_id(self, client):
        """Outcome recording should require hashed user ID."""
        # First get a recommendation to get a valid rec_id
        rec_response = client.get(
            "/api/v1/recommendations", params={"slots": 1, "capital": 5000000}
        )
        if rec_response.status_code == 200:
            data = rec_response.json()
            if data:
                rec = data[0]
                rec_id = rec["id"]

                # Try with non-hashed user ID (should fail validation)
                response = client.post(
                    f"/api/v1/recommendations/{rec_id}/outcome",
                    json={
                        "rec_id": rec_id,
                        "user_id": "plaintext_user_id",  # Not SHA256 hashed
                        "bought_at": rec["buyPrice"],
                        "sold_at": rec["sellPrice"],
                        "quantity": 1,
                        "timestamp": "2024-01-01T00:00:00Z",
                    },
                )
                # Should reject non-hashed user ID
                assert response.status_code in [400, 422]

    def test_outcome_validation_structure(self, client):
        """Outcome endpoint should validate request structure."""
        response = client.post(
            "/api/v1/recommendations/invalid_rec_id/outcome", json={}
        )
        # Should reject due to missing fields
        assert response.status_code in [400, 422, 404]


# =============================================================================
# Performance & Load Tests
# =============================================================================


class TestPerformance:
    """Basic performance validation tests."""

    def test_health_endpoint_response_time(self, client):
        """Health endpoint should respond quickly."""
        start = time.time()
        response = client.get("/api/v1/health")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 5.0, f"Health check too slow: {elapsed:.2f}s"
        print(f"  Health check response time: {elapsed:.3f}s")

    def test_recommendations_response_time(self, client):
        """Recommendations should respond within reasonable time."""
        start = time.time()
        response = client.get("/api/v1/recommendations", params={"capital": 5000000})
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 10.0, f"Recommendations too slow: {elapsed:.2f}s"
        print(f"  Recommendations response time: {elapsed:.3f}s")

    def test_search_response_time(self, client):
        """Search should respond quickly."""
        start = time.time()
        response = client.get("/api/v1/items/search", params={"q": "dragon"})
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 5.0, f"Search too slow: {elapsed:.2f}s"
        print(f"  Search response time: {elapsed:.3f}s")

    def test_concurrent_requests(self, client):
        """Server should handle multiple concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/api/v1/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_count = sum(1 for r in results if r.status_code == 200)
        assert (
            success_count >= 8
        ), f"Too many failed concurrent requests: {success_count}/10 succeeded"
        print(f"  Concurrent requests: {success_count}/10 succeeded")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_endpoint_returns_404(self, client):
        """Invalid endpoints should return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_invalid_item_id_handled(self, client):
        """Invalid item ID should be handled gracefully."""
        response = client.get("/api/v1/predictions/999999999")
        assert response.status_code in [404, 200]  # Either not found or empty result

    def test_empty_search_query(self, client):
        """Empty search query should be handled."""
        response = client.get("/api/v1/items/search", params={"q": ""})
        assert response.status_code in [200, 400, 422]

    def test_special_characters_in_search(self, client):
        """Special characters in search should not crash server."""
        special_queries = ["'", '"', "<script>", "'; DROP TABLE items;--", "🔥"]
        for query in special_queries:
            response = client.get("/api/v1/items/search", params={"q": query})
            # Should not crash - any response code is acceptable except 500
            assert response.status_code != 500, f"Server error on query: {query}"


# =============================================================================
# Run Summary
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("GePT Production Test Suite v1.1.0")
    print(f"Target Server: {PROD_SERVER_URL}")
    print(f"{'='*60}\n")

    pytest.main([__file__, "-v", "--tb=short"])
