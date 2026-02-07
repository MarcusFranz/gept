"""Tests for the API endpoints."""

import os
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestAPI:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self, sample_registry, mock_db_connection):
        """Create test client with mocked engine."""
        # Patch environment variables and engine
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [{"status": "ok", "component": "test"}],
                    "timestamp": "2026-01-08T12:00:00Z",
                    "supported_items": 2,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                mock_engine.get_recommendations.return_value = []
                mock_engine.get_recommendation_by_item_id.return_value = None
                mock_engine.get_recommendation_by_item_name.return_value = None
                mock_engine.get_recommendation_by_id.return_value = None
                mock_engine.get_prediction_for_item.return_value = None
                MockEngine.return_value = mock_engine

                # Import after patching
                from src.api import app, limiter
                import src.api as api_module

                api_module.app.state.runtime.engine = mock_engine

                # Reset rate limiter storage to avoid rate limit issues across tests
                limiter.reset()

                yield TestClient(app), mock_engine

    def test_root(self, client):
        """Test root endpoint."""
        test_client, _ = client
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        test_client, mock_engine = client
        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "checks" in data

    def test_recommendations_get_endpoint(self, client):
        """Test GET recommendations endpoint."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
        )

        assert response.status_code == 200
        # Should return list
        assert isinstance(response.json(), list)

    def test_item_prediction_by_id(self, client):
        """Test getting prediction for a specific item."""
        test_client, mock_engine = client

        mock_engine.get_prediction_for_item.return_value = {
            "item_id": 554,
            "item_name": "Fire rune",
            "best_config": {"hour_offset": 4, "offset_pct": 0.015},
            "fill_probability": 0.72,
            "expected_value": 0.008,
            "buy_price": 4,
            "sell_price": 6,
            "confidence": "high",
            "all_predictions": [],
        }

        response = test_client.get("/api/v1/predictions/554")

        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == 554
        assert data["item_name"] == "Fire rune"

    def test_item_prediction_not_found(self, client):
        """Test prediction for item that doesn't exist."""
        test_client, mock_engine = client
        mock_engine.get_prediction_for_item.return_value = None

        response = test_client.get("/api/v1/predictions/999999")

        assert response.status_code == 404

    def test_recommendations_post(self, client):
        """Test POST recommendations endpoint."""
        test_client, mock_engine = client

        mock_engine.get_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.12,
                "fillConfidence": "Good",

                "trend": "Stable",
            }
        ]

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_recommendations_includes_rationale_fields(self, client):
        """Test recommendations response includes reason and isRecommended fields."""
        test_client, mock_engine = client

        mock_engine.get_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.12,
                "fillConfidence": "Good",

                "trend": "Stable",
                "reason": "Stable trend, high volume, 4h window",
                "isRecommended": True,
            }
        ]

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        rec = data[0]
        # Verify rationale fields are present
        assert "reason" in rec
        assert rec["reason"] == "Stable trend, high volume, 4h window"
        assert "isRecommended" in rec
        assert rec["isRecommended"] is True

    def test_recommendations_get_passthrough(self, client):
        """Test GET recommendations endpoint with different params."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?style=passive&capital=5000000&risk=low&slots=2"
        )

        assert response.status_code == 200

    def test_recommendations_get_single_slot(self, client):
        """Test GET recommendations with slots=1 for slot-by-slot methodology."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = [
            {
                "id": "rec_554_2026010112",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 10000,
                "capitalRequired": 40000,
                "expectedProfit": 20000,
                "confidence": "high",
                "fillProbability": 0.85,
                "fillConfidence": "Strong",
                "trend": "Stable",
                "expectedHours": 2,

                "reason": "High volume, stable prices",
            }
        ]

        response = test_client.get("/api/v1/recommendations?capital=5000000&slots=1")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Verify slots=1 was passed to engine
        call_args = mock_engine.get_recommendations.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["slots"] == 1

    def test_recommendations_get_single_slot_with_exclude(self, client):
        """Test single-slot request with active item exclusions."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        # Request single slot while excluding items already being traded
        response = test_client.get(
            "/api/v1/recommendations?capital=5000000&slots=1&exclude_item_ids=554,565,4151"
        )

        assert response.status_code == 200
        # Verify both slots=1 and exclude_item_ids were passed correctly
        call_args = mock_engine.get_recommendations.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert kwargs["slots"] == 1
        assert kwargs["exclude_item_ids"] == {554, 565, 4151}

    def test_recommendations_get_with_exclude(self, client):
        """Test GET recommendations endpoint with exclude parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
            "&exclude=rec_554_2026010112,rec_565_2026010112"
        )

        assert response.status_code == 200
        # Verify exclude_ids was parsed and passed correctly
        call_args = mock_engine.get_recommendations.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "exclude_ids" in kwargs
        assert kwargs["exclude_ids"] == {"rec_554_2026010112", "rec_565_2026010112"}

    def test_recommendations_get_with_empty_exclude(self, client):
        """Test GET recommendations endpoint with empty exclude parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
            "&exclude="
        )

        assert response.status_code == 200
        # Empty exclude should result in empty set
        call_args = mock_engine.get_recommendations.call_args
        _, kwargs = call_args
        assert kwargs["exclude_ids"] == set()

    def test_recommendations_get_with_whitespace_exclude(self, client):
        """Test GET recommendations with whitespace in exclude parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
            "&exclude= rec_554_2026010112 , rec_565_2026010112 "
        )

        assert response.status_code == 200
        # Whitespace should be stripped
        call_args = mock_engine.get_recommendations.call_args
        _, kwargs = call_args
        assert "rec_554_2026010112" in kwargs["exclude_ids"]
        assert "rec_565_2026010112" in kwargs["exclude_ids"]

    def test_recommendations_get_exclude_item_ids(self, client):
        """Test GET recommendations with item ID exclusions."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&exclude_item_ids=536,5295,4151"
        )

        assert response.status_code == 200
        # Verify exclude_item_ids was passed to engine
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("exclude_item_ids") == {536, 5295, 4151}

    def test_recommendations_get_exclude_item_ids_empty(self, client):
        """Test GET recommendations with no item ID exclusions."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get("/api/v1/recommendations?capital=10000000")

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("exclude_item_ids") == set()

    def test_recommendations_get_exclude_item_ids_invalid(self, client):
        """Test GET recommendations with invalid item IDs (non-numeric ignored)."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&exclude_item_ids=536,invalid,4151"
        )

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        # Only valid numeric IDs should be included
        assert call_kwargs.get("exclude_item_ids") == {536, 4151}

    def test_recommendations_get_error_returns_500(self, client):
        """Test GET recommendations returns 500 on engine errors."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.side_effect = Exception(
            "Database connection failed"
        )

        response = test_client.get("/api/v1/recommendations?capital=10000000")

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_recommendations_validation(self, client):
        """Test request validation for recommendations."""
        test_client, _ = client

        # Invalid style
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "invalid",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
            },
        )
        assert response.status_code == 422

        # Capital too low
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 100,  # Below 1000 minimum
                "risk": "medium",
                "slots": 4,
            },
        )
        assert response.status_code == 422

        # Too many slots (exceeds maximum of 20 for any tier)
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 21,  # Max is 20 even for unlimited tier
            },
        )
        assert response.status_code == 422

    def test_tier_based_slot_limits_post(self, client):
        """Test tier-based slot limits for POST endpoint."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        # Free tier (default) - max 8 slots, 9 should fail
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 9,
            },
        )
        assert response.status_code == 400
        assert "Maximum 8 slots allowed for free tier" in response.json()["detail"]

        # Free tier explicit - max 8 slots
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 9,
                "user_tier": "free",
            },
        )
        assert response.status_code == 400
        assert "Maximum 8 slots allowed for free tier" in response.json()["detail"]

        # Premium tier - also max 8 slots
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 9,
                "user_tier": "premium",
            },
        )
        assert response.status_code == 400
        assert "Maximum 8 slots allowed for premium tier" in response.json()["detail"]

        # Unlimited tier - max 20 slots, 9 should succeed
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 9,
                "user_tier": "unlimited",
            },
        )
        assert response.status_code == 200

        # Unlimited tier - 20 slots should succeed
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 20,
                "user_tier": "unlimited",
            },
        )
        assert response.status_code == 200

    def test_tier_based_slot_limits_get(self, client):
        """Test tier-based slot limits for GET endpoint."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        # Free tier (default) - max 8 slots, 9 should fail
        response = test_client.get("/api/v1/recommendations?capital=10000000&slots=9")
        assert response.status_code == 400
        assert "Maximum 8 slots allowed for free tier" in response.json()["detail"]

        # Premium tier - also max 8 slots
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&slots=9&user_tier=premium"
        )
        assert response.status_code == 400
        assert "Maximum 8 slots allowed for premium tier" in response.json()["detail"]

        # Unlimited tier - 20 slots should succeed
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&slots=20&user_tier=unlimited"
        )
        assert response.status_code == 200

        # Unlimited tier passes slots to engine correctly
        mock_engine.get_recommendations.assert_called()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("slots") == 20

    def test_recommendations_get_with_min_offset_pct(self, client):
        """Test GET recommendations with min_offset_pct parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&min_offset_pct=0.0175"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("min_offset_pct") == 0.0175

    def test_recommendations_get_with_max_offset_pct(self, client):
        """Test GET recommendations with max_offset_pct parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&max_offset_pct=0.0200"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("max_offset_pct") == 0.0200

    def test_recommendations_get_with_offset_range(self, client):
        """Test GET recommendations with both min and max offset_pct."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?capital=10000000"
            "&min_offset_pct=0.015&max_offset_pct=0.02"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("min_offset_pct") == 0.015
        assert call_kwargs.get("max_offset_pct") == 0.02

    def test_recommendations_get_offset_pct_precedence(self, client):
        """Test that offset_pct takes precedence over min/max when all provided."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?capital=10000000"
            "&offset_pct=0.0175&min_offset_pct=0.0125&max_offset_pct=0.025"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        # All three params should be passed to engine
        assert call_kwargs.get("offset_pct") == 0.0175
        assert call_kwargs.get("min_offset_pct") == 0.0125
        assert call_kwargs.get("max_offset_pct") == 0.025

    def test_recommendations_get_offset_validation_too_low(self, client):
        """Test offset_pct validation rejects values below 0.0125."""
        test_client, _ = client

        # min_offset_pct too low
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&min_offset_pct=0.01"
        )
        assert response.status_code == 422

        # max_offset_pct too low
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&max_offset_pct=0.01"
        )
        assert response.status_code == 422

    def test_recommendations_get_offset_validation_too_high(self, client):
        """Test offset_pct validation rejects values above 0.0250."""
        test_client, _ = client

        # min_offset_pct too high
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&min_offset_pct=0.03"
        )
        assert response.status_code == 422

        # max_offset_pct too high
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&max_offset_pct=0.03"
        )
        assert response.status_code == 422

    def test_recommendations_post_with_offset_params(self, client):
        """Test POST recommendations with offset parameters."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "offset_pct": 0.02,
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("offset_pct") == 0.02

    def test_recommendations_post_with_offset_range(self, client):
        """Test POST recommendations with min/max offset parameters."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "passive",
                "capital": 5000000,
                "risk": "low",
                "slots": 3,
                "min_offset_pct": 0.0125,
                "max_offset_pct": 0.0175,
            },
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("min_offset_pct") == 0.0125
        assert call_kwargs.get("max_offset_pct") == 0.0175

    def test_recommendations_post_offset_validation(self, client):
        """Test POST request validation for offset parameters."""
        test_client, _ = client

        # offset_pct too low
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "offset_pct": 0.01,  # Below 0.0125 minimum
            },
        )
        assert response.status_code == 422

        # offset_pct too high
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "offset_pct": 0.03,  # Above 0.0250 maximum
            },
        )
        assert response.status_code == 422

        # min_offset_pct too high
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "min_offset_pct": 0.03,  # Above 0.0250 maximum
            },
        )
        assert response.status_code == 422

    def test_recommendations_get_with_metadata(self, client):
        """Test GET recommendations with freshness metadata."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []
        mock_engine.get_freshness_metadata.return_value = {
            "inference_at": "2026-01-14T12:00:00+00:00",
            "inference_age_seconds": 127.5,
            "stale": False,
            "stale_threshold_seconds": 300,
        }

        response = test_client.get(
            "/api/v1/recommendations?"
            "style=active&capital=10000000&risk=medium&slots=4&include_metadata=true"
        )

        assert response.status_code == 200
        data = response.json()
        # Should return wrapped response with metadata
        assert "recommendations" in data
        assert "metadata" in data
        assert isinstance(data["recommendations"], list)
        assert data["metadata"]["inference_at"] == "2026-01-14T12:00:00+00:00"
        assert data["metadata"]["inference_age_seconds"] == 127.5
        assert data["metadata"]["stale"] is False
        assert data["metadata"]["stale_threshold_seconds"] == 300

    def test_recommendations_get_without_metadata(self, client):
        """Test GET recommendations without metadata returns list."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/api/v1/recommendations?"
            "style=active&capital=10000000&risk=medium&slots=4"
        )

        assert response.status_code == 200
        # Should return plain list (not wrapped)
        assert isinstance(response.json(), list)

    def test_recommendations_post_with_metadata(self, client):
        """Test POST recommendations with freshness metadata."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []
        mock_engine.get_freshness_metadata.return_value = {
            "inference_at": "2026-01-14T12:00:00+00:00",
            "inference_age_seconds": 450.0,
            "stale": True,
            "stale_threshold_seconds": 300,
        }

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "include_metadata": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Should return wrapped response with metadata
        assert "recommendations" in data
        assert "metadata" in data
        assert data["metadata"]["stale"] is True  # Over threshold
        assert data["metadata"]["inference_age_seconds"] == 450.0

    def test_recommendations_post_without_metadata(self, client):
        """Test POST recommendations without metadata returns list."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
            },
        )

        assert response.status_code == 200
        # Should return plain list (not wrapped)
        assert isinstance(response.json(), list)

    # ==================== User ID Normalization Tests ====================

    def test_recommendations_post_normalizes_raw_user_id(self, client):
        """Test that raw Discord IDs are hashed server-side."""
        import hashlib

        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        raw_discord_id = "123456789012345678"  # Not a valid SHA256
        expected_hash = hashlib.sha256(raw_discord_id.encode()).hexdigest()

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "userId": raw_discord_id,
            },
        )

        assert response.status_code == 200
        # Verify the engine received the hashed user_id
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["user_id"] == expected_hash

    def test_recommendations_post_preserves_valid_sha256(self, client):
        """Test that valid SHA256 hashes are passed through unchanged."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        valid_sha256 = "a" * 64  # Valid SHA256 hash

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "userId": valid_sha256,
            },
        )

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["user_id"] == valid_sha256

    def test_recommendations_get_normalizes_raw_user_id(self, client):
        """Test GET endpoint normalizes raw user IDs."""
        import hashlib

        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        raw_user_id = "my_discord_user"
        expected_hash = hashlib.sha256(raw_user_id.encode()).hexdigest()

        response = test_client.get(
            f"/api/v1/recommendations?capital=10000000&user_id={raw_user_id}"
        )

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["user_id"] == expected_hash

    def test_recommendations_get_preserves_valid_sha256(self, client):
        """Test GET endpoint preserves valid SHA256 hashes."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        valid_sha256 = "b" * 64

        response = test_client.get(
            f"/api/v1/recommendations?capital=10000000&user_id={valid_sha256}"
        )

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["user_id"] == valid_sha256

    def test_recommendations_post_handles_empty_user_id(self, client):
        """Test that empty user_id is handled correctly."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "userId": "",
            },
        )

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["user_id"] is None

    def test_recommendations_post_handles_none_user_id(self, client):
        """Test that None user_id is handled correctly."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                # No userId field
            },
        )

        assert response.status_code == 200
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["user_id"] is None

    def test_refresh_item_prices(self, client):
        """Test refreshing prices for an item."""
        from datetime import datetime, timezone

        test_client, mock_engine = client

        # Mock loader methods
        mock_engine.loader.get_item_name.return_value = "Dragon bones"
        mock_engine.loader.get_latest_price.return_value = {
            "timestamp": datetime(2026, 1, 9, 12, 15, 0, tzinfo=timezone.utc),
            "high": 2089,
            "low": 2015,
            "high_time": datetime(2026, 1, 9, 12, 14, 0, tzinfo=timezone.utc),
            "low_time": datetime(2026, 1, 9, 12, 13, 0, tzinfo=timezone.utc),
        }

        response = test_client.get("/api/v1/recommendations/item/536/refresh")

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 536
        assert data["itemName"] == "Dragon bones"
        assert data["buyPrice"] == 2089
        assert data["sellPrice"] == 2015
        assert "updatedAt" in data
        assert data["updatedAt"].endswith("Z")

    def test_refresh_item_prices_item_not_found(self, client):
        """Test refresh prices for non-existent item."""
        test_client, mock_engine = client

        # Mock loader returning None for unknown item
        mock_engine.loader.get_item_name.return_value = None

        response = test_client.get("/api/v1/recommendations/item/999999/refresh")

        assert response.status_code == 404
        assert "Item not found" in response.json()["detail"]

    def test_refresh_item_prices_no_price_data(self, client):
        """Test refresh prices when no price data available."""
        test_client, mock_engine = client

        # Item exists but no price data
        mock_engine.loader.get_item_name.return_value = "Dragon bones"
        mock_engine.loader.get_latest_price.return_value = None

        response = test_client.get("/api/v1/recommendations/item/536/refresh")

        assert response.status_code == 404
        assert "No price data" in response.json()["detail"]

    def test_item_search(self, client):
        """Test item search endpoint."""
        test_client, mock_engine = client

        # Mock search_items_by_name to return dragon items
        mock_engine.search_items_by_name.return_value = [
            {"item_id": 536, "item_name": "Dragon bones"},
            {"item_id": 4587, "item_name": "Dragon scimitar"},
        ]

        # Test search for "dragon"
        response = test_client.get("/api/v1/items/search?q=dragon&limit=5")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5
        # Should include dragon items
        item_names = [item["name"] for item in data]
        assert any("Dragon" in name for name in item_names)

    def test_item_search_fuzzy(self, client):
        """Test fuzzy matching in item search."""
        test_client, mock_engine = client

        # Mock search_items_by_name to return blood rune (fuzzy match for "blod")
        mock_engine.search_items_by_name.return_value = [
            {"item_id": 565, "item_name": "Blood rune"},
        ]

        # Test typo tolerance: "blod" should match "Blood"
        response = test_client.get("/api/v1/items/search?q=blod&limit=5")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should find Blood rune despite typo
        assert len(data) > 0

    def test_item_search_case_insensitive(self, client):
        """Test case-insensitive search."""
        test_client, mock_engine = client

        # Mock search_items_by_name to return fire rune
        mock_engine.search_items_by_name.return_value = [
            {"item_id": 554, "item_name": "Fire rune"},
        ]

        # Test with different cases
        response_lower = test_client.get("/api/v1/items/search?q=fire&limit=5")
        response_upper = test_client.get("/api/v1/items/search?q=FIRE&limit=5")
        response_mixed = test_client.get("/api/v1/items/search?q=FiRe&limit=5")

        assert response_lower.status_code == 200
        assert response_upper.status_code == 200
        assert response_mixed.status_code == 200

        # All should return the same result
        data_lower = response_lower.json()
        data_upper = response_upper.json()
        data_mixed = response_mixed.json()

        assert len(data_lower) == len(data_upper) == len(data_mixed)
        if len(data_lower) > 0:
            assert (
                data_lower[0]["itemId"]
                == data_upper[0]["itemId"]
                == data_mixed[0]["itemId"]
            )

    def test_item_search_limit(self, client):
        """Test search limit parameter."""
        test_client, mock_engine = client

        # Mock search_items_by_name to return 3 items (respecting limit)
        mock_engine.search_items_by_name.return_value = [
            {"item_id": 1, "item_name": "Item 1"},
            {"item_id": 2, "item_name": "Item 2"},
            {"item_id": 3, "item_name": "Item 3"},
        ]

        # Test with limit=3
        response = test_client.get("/api/v1/items/search?q=item&limit=3")

        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 3

    def test_item_search_empty_results(self, client):
        """Test search with no matching items."""
        test_client, mock_engine = client

        # Mock search_items_by_name to return empty list
        mock_engine.search_items_by_name.return_value = []

        response = test_client.get("/api/v1/items/search?q=nonexistent&limit=5")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_item_search_validation(self, client):
        """Test search parameter validation."""
        test_client, _ = client

        # Missing query parameter
        response = test_client.get("/api/v1/items/search")
        assert response.status_code == 422

        # Empty query
        response = test_client.get("/api/v1/items/search?q=")
        assert response.status_code == 422

        # Limit too high
        response = test_client.get("/api/v1/items/search?q=test&limit=100")
        assert response.status_code == 422

        # Limit too low
        response = test_client.get("/api/v1/items/search?q=test&limit=0")
        assert response.status_code == 422

    def test_item_search_acronym_expansion(self, client):
        """Test OSRS acronym expansion in search."""
        test_client, mock_engine = client

        # Mock search to return Armadyl godsword for "ags" acronym
        mock_engine.search_items_by_name.return_value = [
            {"item_id": 11802, "item_name": "Armadyl godsword"},
        ]

        # Test "ags" acronym
        response = test_client.get("/api/v1/items/search?q=ags&limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        # Verify the method was called
        mock_engine.search_items_by_name.assert_called()
        # The expansion happens in search_items_by_name, so we verify results
        assert any("Armadyl godsword" in item["name"] for item in data)

    def test_item_search_acronym_case_insensitive(self, client):
        """Test acronym expansion is case-insensitive."""
        test_client, mock_engine = client

        mock_engine.search_items_by_name.return_value = [
            {"item_id": 11802, "item_name": "Armadyl godsword"},
        ]

        # Test various cases - all should work
        for query in ["AGS", "ags", "Ags", "aGs"]:
            mock_engine.search_items_by_name.reset_mock()
            response = test_client.get(f"/api/v1/items/search?q={query}&limit=5")
            assert response.status_code == 200
            data = response.json()
            assert len(data) >= 1

    def test_item_search_non_acronym_unchanged(self, client):
        """Test that non-acronym queries work normally."""
        test_client, mock_engine = client

        mock_engine.search_items_by_name.return_value = [
            {"item_id": 536, "item_name": "Dragon bones"},
        ]

        # Regular query should work as before
        response = test_client.get("/api/v1/items/search?q=dragon&limit=5")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any("Dragon" in item["name"] for item in data)

    def test_trade_outcome_success(self, client):
        """Test successful trade outcome reporting."""
        test_client, mock_engine = client

        # Mock the outcome database connection (separate from predictions db)
        import src.api as api_module

        mock_outcome_engine = MagicMock()
        mock_conn = MagicMock()
        mock_outcome_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_outcome_engine.connect.return_value.__exit__ = MagicMock(return_value=None)
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "a" * 64,  # Valid SHA256 hash
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_123",
                "buyPrice": 43250,
                "sellPrice": 44890,
                "quantity": 1100,
                "actualProfit": 1602000,
                "reportedAt": "2026-01-09T12:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "Outcome recorded"

    def test_trade_outcome_database_unavailable(self, client):
        """Test trade outcome when outcome database is not available."""
        test_client, _ = client

        # Ensure outcome database is not available
        import src.api as api_module

        api_module.app.state.runtime.outcome_db_engine = None

        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "a" * 64,
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_123",
                "buyPrice": 43250,
                "sellPrice": 44890,
                "quantity": 1100,
                "actualProfit": 1602000,
                "reportedAt": "2026-01-09T12:00:00Z",
            },
        )

        assert response.status_code == 503
        assert "not available" in response.json()["detail"].lower()

    def test_trade_outcome_rec_id_mismatch(self, client):
        """Test trade outcome with mismatched rec_id."""
        test_client, _ = client

        # Set up mock outcome database
        import src.api as api_module

        mock_outcome_engine = MagicMock()
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "a" * 64,
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_456",  # Doesn't match URL
                "buyPrice": 43250,
                "sellPrice": 44890,
                "quantity": 1100,
                "actualProfit": 1602000,
                "reportedAt": "2026-01-09T12:00:00Z",
            },
        )

        assert response.status_code == 400
        assert "does not match" in response.json()["detail"]

    def test_trade_outcome_invalid_user_id(self, client):
        """Test trade outcome with invalid (unhashed) user ID."""
        test_client, _ = client

        # Set up mock outcome database
        import src.api as api_module

        mock_outcome_engine = MagicMock()
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "123456789",  # Not a SHA256 hash
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_123",
                "buyPrice": 43250,
                "sellPrice": 44890,
                "quantity": 1100,
                "actualProfit": 1602000,
                "reportedAt": "2026-01-09T12:00:00Z",
            },
        )

        assert response.status_code == 400
        assert "SHA256" in response.json()["detail"]

    def test_trade_outcome_invalid_timestamp(self, client):
        """Test trade outcome with invalid timestamp."""
        test_client, _ = client

        # Set up mock outcome database
        import src.api as api_module

        mock_outcome_engine = MagicMock()
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "a" * 64,
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_123",
                "buyPrice": 43250,
                "sellPrice": 44890,
                "quantity": 1100,
                "actualProfit": 1602000,
                "reportedAt": "not-a-timestamp",
            },
        )

        assert response.status_code == 400
        assert "timestamp" in response.json()["detail"].lower()

    def test_trade_outcome_validation(self, client):
        """Test trade outcome field validation."""
        test_client, _ = client

        # Negative buy price
        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "a" * 64,
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_123",
                "buyPrice": 0,  # Must be >= 1
                "sellPrice": 44890,
                "quantity": 1100,
                "actualProfit": 1602000,
                "reportedAt": "2026-01-09T12:00:00Z",
            },
        )
        assert response.status_code == 422

        # Zero quantity
        response = test_client.post(
            "/api/v1/recommendations/rec_123/outcome",
            json={
                "userId": "a" * 64,
                "itemId": 5295,
                "itemName": "Ranarr seed",
                "recId": "rec_123",
                "buyPrice": 43250,
                "sellPrice": 44890,
                "quantity": 0,  # Must be >= 1
                "actualProfit": 1602000,
                "reportedAt": "2026-01-09T12:00:00Z",
            },
        )
        assert response.status_code == 422

    def test_user_stats_success(self, client):
        """Test successful user stats retrieval."""
        test_client, _ = client

        # Mock the outcome_db_engine
        from unittest.mock import MagicMock, patch

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

        # Different mock results for different queries
        call_count = [0]

        def mock_execute(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            # First call: aggregate stats (total_trades, total_profit, winning_trades)
            if call_count[0] == 1:
                result.fetchone.return_value = (10, 5000000, 7)
            # Second call: best flip (item_name, profit)
            elif call_count[0] == 2:
                result.fetchone.return_value = ("Ranarr seed", 2100000)
            # Third call: worst flip (item_name, profit)
            elif call_count[0] == 3:
                result.fetchone.return_value = ("Dragon bones", -200000)
            # Fourth call: previous period stats
            elif call_count[0] == 4:
                result.fetchone.return_value = (8, 3000000, 5)
            else:
                result.fetchone.return_value = None
            return result

        mock_conn.execute = mock_execute

        with patch("src.api.outcome_db_engine", mock_engine):
            valid_hash = "a" * 64
            response = test_client.get(f"/api/v1/users/{valid_hash}/stats?period=week")

        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "week"
        assert "startDate" in data
        assert "endDate" in data
        assert data["totalProfit"] == 5000000
        assert data["totalTrades"] == 10
        assert data["winRate"] == 0.7
        assert data["bestFlip"]["itemName"] == "Ranarr seed"
        assert data["bestFlip"]["profit"] == 2100000
        assert data["worstFlip"]["itemName"] == "Dragon bones"
        assert data["worstFlip"]["profit"] == -200000

    def test_user_stats_invalid_hash(self, client):
        """Test user stats with invalid hash format."""
        test_client, _ = client

        from unittest.mock import MagicMock, patch

        mock_engine = MagicMock()
        with patch("src.api.outcome_db_engine", mock_engine):
            # Too short
            response = test_client.get("/api/v1/users/abc123/stats")
            assert response.status_code == 400
            assert "SHA256" in response.json()["detail"]

            # Not hex
            invalid_hash = "g" * 64  # 'g' is not a hex character
            response = test_client.get(f"/api/v1/users/{invalid_hash}/stats")
            assert response.status_code == 400
            assert "SHA256" in response.json()["detail"]

    def test_user_stats_db_unavailable(self, client):
        """Test user stats when outcome DB is not available."""
        test_client, _ = client

        from unittest.mock import patch

        with patch("src.api.outcome_db_engine", None):
            valid_hash = "a" * 64
            response = test_client.get(f"/api/v1/users/{valid_hash}/stats")

        assert response.status_code == 503
        assert "not available" in response.json()["detail"]

    def test_user_stats_no_data(self, client):
        """Test user stats for user with no trades."""
        test_client, _ = client

        from unittest.mock import MagicMock, patch

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

        # Mock query results: no trades
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (0, 0, 0)
        mock_conn.execute.return_value = mock_result

        with patch("src.api.outcome_db_engine", mock_engine):
            valid_hash = "b" * 64
            response = test_client.get(f"/api/v1/users/{valid_hash}/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["totalTrades"] == 0
        assert data["totalProfit"] == 0
        assert data["winRate"] == 0.0
        assert data["bestFlip"] is None
        assert data["worstFlip"] is None

    def test_user_stats_all_periods(self, client):
        """Test user stats with different period values."""
        test_client, _ = client

        from unittest.mock import MagicMock, patch

        def create_mock_engine():
            """Create a fresh mock engine with proper query handling."""
            mock_engine = MagicMock()
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

            call_count = [0]

            def mock_execute(query, params=None):
                call_count[0] += 1
                result = MagicMock()
                # Aggregate stats query
                if call_count[0] == 1:
                    result.fetchone.return_value = (5, 2000000, 3)
                # Best flip query
                elif call_count[0] == 2:
                    result.fetchone.return_value = ("Test item", 500000)
                # Worst flip query
                elif call_count[0] == 3:
                    result.fetchone.return_value = ("Bad item", -100000)
                # Previous period stats
                elif call_count[0] == 4:
                    result.fetchone.return_value = (4, 1500000, 2)
                else:
                    result.fetchone.return_value = (0, 0, 0)
                return result

            mock_conn.execute = mock_execute
            return mock_engine

        valid_hash = "c" * 64

        # Test week - needs fresh mock for each request
        with patch("src.api.outcome_db_engine", create_mock_engine()):
            response = test_client.get(f"/api/v1/users/{valid_hash}/stats?period=week")
            assert response.status_code == 200
            assert response.json()["period"] == "week"

        # Test month
        with patch("src.api.outcome_db_engine", create_mock_engine()):
            response = test_client.get(f"/api/v1/users/{valid_hash}/stats?period=month")
            assert response.status_code == 200
            assert response.json()["period"] == "month"

        # Test all (no comparison to previous period)
        with patch("src.api.outcome_db_engine", create_mock_engine()):
            response = test_client.get(f"/api/v1/users/{valid_hash}/stats?period=all")
            assert response.status_code == 200
            assert response.json()["period"] == "all"
            # "all" period should not have comparison
            assert response.json()["comparisonToPreviousPeriod"] is None

    # ==================== Feedback Endpoint Tests ====================

    def test_feedback_success(self, client):
        """Test successful feedback submission."""
        test_client, _ = client

        # Mock the outcome database connection
        import src.api as api_module

        mock_outcome_engine = MagicMock()
        mock_conn = MagicMock()
        mock_outcome_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_outcome_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

        # Mock the INSERT ... RETURNING id
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (42,)
        mock_conn.execute.return_value = mock_result

        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "price_too_high",
                "side": "buy",
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["feedbackId"] == 42
        assert data["message"] == "Feedback recorded"

    def test_feedback_with_optional_fields(self, client):
        """Test feedback with all optional fields populated."""
        test_client, _ = client

        import src.api as api_module

        mock_outcome_engine = MagicMock()
        mock_conn = MagicMock()
        mock_outcome_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_outcome_engine.connect.return_value.__exit__ = MagicMock(return_value=None)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (43,)
        mock_conn.execute.return_value = mock_result

        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "price_too_high",
                "recId": "rec_536_2026011510",
                "side": "buy",
                "notes": "Price was 5% higher than expected",
                "recommendedPrice": 2000,
                "actualPrice": 2100,
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        assert response.status_code == 200

    def test_feedback_invalid_user_id(self, client):
        """Test feedback with non-hashed user ID."""
        test_client, _ = client

        import src.api as api_module

        mock_outcome_engine = MagicMock()
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "123456",  # Not SHA256
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "price_too_high",
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        assert response.status_code == 400
        assert "SHA256" in response.json()["detail"]

    def test_feedback_invalid_type(self, client):
        """Test feedback with invalid feedback type."""
        test_client, _ = client

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "invalid_type",
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        # Pydantic validation failure
        assert response.status_code == 422

    def test_feedback_invalid_rec_id_format(self, client):
        """Test feedback with malformed rec_id."""
        test_client, _ = client

        import src.api as api_module

        mock_outcome_engine = MagicMock()
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "price_too_high",
                "recId": "invalid_format",
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        assert response.status_code == 400
        assert "recId format" in response.json()["detail"]

    def test_feedback_database_unavailable(self, client):
        """Test feedback when database not available."""
        test_client, _ = client

        import src.api as api_module

        api_module.app.state.runtime.outcome_db_engine = None

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "price_too_high",
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        assert response.status_code == 503
        assert "not available" in response.json()["detail"].lower()

    def test_feedback_notes_max_length(self, client):
        """Test that notes over 500 chars are rejected by Pydantic validation."""
        test_client, _ = client

        long_notes = "x" * 600  # Over 500 chars

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "other",
                "notes": long_notes,
                "submittedAt": "2026-01-15T10:00:00Z",
            },
        )

        # Should fail Pydantic validation (max_length=500)
        assert response.status_code == 422

    def test_feedback_invalid_timestamp(self, client):
        """Test feedback with invalid timestamp."""
        test_client, _ = client

        import src.api as api_module

        mock_outcome_engine = MagicMock()
        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.post(
            "/api/v1/feedback",
            json={
                "userId": "a" * 64,
                "itemId": 536,
                "itemName": "Dragon bones",
                "feedbackType": "price_too_high",
                "submittedAt": "not-a-timestamp",
            },
        )

        assert response.status_code == 400
        assert "timestamp" in response.json()["detail"].lower()

    def test_feedback_analytics_success(self, client):
        """Test successful feedback analytics retrieval."""
        test_client, _ = client

        import src.api as api_module

        mock_outcome_engine = MagicMock()
        mock_conn = MagicMock()
        mock_outcome_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_outcome_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

        call_count = [0]

        def mock_execute(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # Type breakdown
                result.fetchall.return_value = [
                    ("price_too_high", 15),
                    ("did_not_fill", 10),
                    ("volume_too_low", 5),
                ]
            else:  # Top items
                result.fetchall.return_value = [
                    (536, "Dragon bones", 12),
                    (4151, "Abyssal whip", 8),
                ]
            return result

        mock_conn.execute = mock_execute

        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.get("/api/v1/feedback/analytics?period=week")

        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "week"
        assert data["totalFeedback"] == 30
        assert len(data["byType"]) == 3
        assert data["byType"][0]["feedbackType"] == "price_too_high"
        assert data["byType"][0]["count"] == 15

    def test_feedback_analytics_by_item(self, client):
        """Test feedback analytics filtered by item."""
        test_client, _ = client

        import src.api as api_module

        mock_outcome_engine = MagicMock()
        mock_conn = MagicMock()
        mock_outcome_engine.connect.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_outcome_engine.connect.return_value.__exit__ = MagicMock(return_value=None)

        call_count = [0]

        def mock_execute(query, params=None):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:  # Type breakdown query
                result.fetchall.return_value = [("price_too_high", 5)]
            else:  # Top items query
                result.fetchall.return_value = [(536, "Dragon bones", 5)]
            return result

        mock_conn.execute = mock_execute

        api_module.app.state.runtime.outcome_db_engine = mock_outcome_engine

        response = test_client.get("/api/v1/feedback/analytics?item_id=536")

        assert response.status_code == 200

    def test_feedback_analytics_database_unavailable(self, client):
        """Test analytics when database not available."""
        test_client, _ = client

        import src.api as api_module

        api_module.app.state.runtime.outcome_db_engine = None

        response = test_client.get("/api/v1/feedback/analytics")

        assert response.status_code == 503

    # ==================== Order Update Endpoint Tests ====================

    def test_order_update_success(self, client):
        """Test successful order evaluation."""
        test_client, mock_engine = client

        # Mock the evaluate_active_order method
        mock_engine.evaluate_active_order.return_value = {
            "action": "wait",
            "confidence": 0.75,
            "current_fill_probability": 0.65,
            "recommendations": {
                "adjust_price": {
                    "suggested_price": 985000,
                    "new_fill_probability": 0.75,
                    "cost_difference": 5000,
                },
                "wait": {"estimated_fill_time_minutes": 60},
                "abort_retry": None,
                "liquidate": {"instant_price": 980000, "loss_amount": 0},
            },
            "reasoning": "Fill probability is strong at 65%. Waiting is recommended.",
        }

        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 980000,
                "quantity": 1,
                "time_elapsed_minutes": 30,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "wait"
        assert data["confidence"] == 0.75
        assert data["current_fill_probability"] == 0.65
        assert "recommendations" in data
        assert "reasoning" in data

    def test_order_update_with_user_id(self, client):
        """Test order evaluation with valid user ID."""
        test_client, mock_engine = client

        mock_engine.evaluate_active_order.return_value = {
            "action": "adjust_price",
            "confidence": 0.6,
            "current_fill_probability": 0.35,
            "recommendations": {
                "adjust_price": {
                    "suggested_price": 990000,
                    "new_fill_probability": 0.6,
                    "cost_difference": 10000,
                },
                "wait": {"estimated_fill_time_minutes": 120},
                "abort_retry": None,
                "liquidate": {"instant_price": 980000, "loss_amount": 0},
            },
            "reasoning": "Price adjustment recommended.",
        }

        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 980000,
                "quantity": 1,
                "time_elapsed_minutes": 60,
                "user_id": "a" * 64,  # Valid SHA256 hash
            },
        )

        assert response.status_code == 200
        # Verify user_id was passed to engine
        mock_engine.evaluate_active_order.assert_called_once()
        call_kwargs = mock_engine.evaluate_active_order.call_args[1]
        assert call_kwargs["user_id"] == "a" * 64

    def test_order_update_invalid_user_id(self, client):
        """Test order evaluation with invalid user ID."""
        test_client, mock_engine = client

        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 980000,
                "quantity": 1,
                "time_elapsed_minutes": 30,
                "user_id": "invalid",  # Not SHA256
            },
        )

        assert response.status_code == 400
        assert "SHA256" in response.json()["detail"]

    def test_order_update_sell_order(self, client):
        """Test evaluation of a sell order."""
        test_client, mock_engine = client

        mock_engine.evaluate_active_order.return_value = {
            "action": "liquidate",
            "confidence": 0.7,
            "current_fill_probability": 0.1,
            "recommendations": {
                "adjust_price": {
                    "suggested_price": 1005000,
                    "new_fill_probability": 0.5,
                    "cost_difference": 25000,
                },
                "wait": {"estimated_fill_time_minutes": 240},
                "abort_retry": None,
                "liquidate": {"instant_price": 980000, "loss_amount": 50000},
            },
            "reasoning": "Low fill probability. Consider liquidating.",
        }

        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "sell",
                "user_price": 1030000,
                "quantity": 5,
                "time_elapsed_minutes": 180,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "liquidate"
        assert data["recommendations"]["liquidate"]["loss_amount"] == 50000

    def test_order_update_validation_errors(self, client):
        """Test validation of order update request fields."""
        test_client, _ = client

        # Invalid order_type
        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "invalid",
                "user_price": 980000,
                "quantity": 1,
                "time_elapsed_minutes": 30,
            },
        )
        assert response.status_code == 422

        # Zero quantity
        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 980000,
                "quantity": 0,
                "time_elapsed_minutes": 30,
            },
        )
        assert response.status_code == 422

        # Negative time elapsed
        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 980000,
                "quantity": 1,
                "time_elapsed_minutes": -1,
            },
        )
        assert response.status_code == 422

        # Zero price
        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 0,
                "quantity": 1,
                "time_elapsed_minutes": 30,
            },
        )
        assert response.status_code == 422

    def test_order_update_engine_error(self, client):
        """Test order update when engine raises an error."""
        test_client, mock_engine = client

        mock_engine.evaluate_active_order.side_effect = Exception(
            "Database connection failed"
        )

        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 980000,
                "quantity": 1,
                "time_elapsed_minutes": 30,
            },
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_order_update_with_alternatives(self, client):
        """Test order evaluation that returns alternatives."""
        test_client, mock_engine = client

        mock_engine.evaluate_active_order.return_value = {
            "action": "abort_retry",
            "confidence": 0.65,
            "current_fill_probability": 0.08,
            "recommendations": {
                "adjust_price": None,
                "wait": {"estimated_fill_time_minutes": 300},
                "abort_retry": {
                    "alternative_items": [
                        {
                            "item_id": 11802,
                            "item_name": "Armadyl godsword",
                            "expected_profit": 50000,
                            "fill_probability": 0.5,
                            "expected_hours": 4,
                        },
                        {
                            "item_id": 536,
                            "item_name": "Dragon bones",
                            "expected_profit": 20000,
                            "fill_probability": 0.7,
                            "expected_hours": 2,
                        },
                    ]
                },
                "liquidate": {"instant_price": 950000, "loss_amount": 0},
            },
            "reasoning": "Better opportunities available.",
        }

        response = test_client.post(
            "/api/v1/recommendations/update",
            json={
                "item_id": 4151,
                "order_type": "buy",
                "user_price": 900000,
                "quantity": 1,
                "time_elapsed_minutes": 180,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "abort_retry"
        assert len(data["recommendations"]["abort_retry"]["alternative_items"]) == 2


class TestItemPriceLookup:
    """Test cases for the item price lookup endpoint (issue #135)."""

    @pytest.fixture
    def client(self, sample_registry, mock_db_connection):
        """Create test client with mocked engine."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.health_check.return_value = {"status": "ok"}
                MockEngine.return_value = mock_engine

                from src.api import app, limiter
                import src.api as api_module

                api_module.app.state.runtime.engine = mock_engine
                limiter.reset()

                yield TestClient(app), mock_engine

    def test_price_lookup_buy_side(self, client):
        """Test price lookup for buy side."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = {
            "itemId": 554,
            "itemName": "Fire rune",
            "side": "buy",
            "recommendedPrice": 4,
            "currentMarketPrice": 5,
            "offsetPercent": 0.02,
            "fillProbability": 0.72,
            "expectedValue": 0.008,
            "timeWindowHours": 24,
            "volume24h": 100000,
            "trend": "Stable",
            "isRecommended": True,
            "warning": None,
            "marginGp": 2,
            "marginPercent": 0.5,
            "buyLimit": 10000,
        }

        response = test_client.get("/api/v1/items/554/price?side=buy")

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 554
        assert data["side"] == "buy"
        assert data["recommendedPrice"] == 4
        assert data["isRecommended"] is True
        assert data["marginGp"] == 2
        assert data["buyLimit"] == 10000

    def test_price_lookup_sell_side(self, client):
        """Test price lookup for sell side."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = {
            "itemId": 554,
            "itemName": "Fire rune",
            "side": "sell",
            "recommendedPrice": 6,
            "currentMarketPrice": 5,
            "offsetPercent": 0.02,
            "fillProbability": 0.65,
            "expectedValue": 0.006,
            "timeWindowHours": 24,
            "volume24h": 100000,
            "trend": "Stable",
            "isRecommended": True,
            "warning": None,
            "marginGp": 2,
            "marginPercent": 0.5,
            "buyLimit": 10000,
        }

        response = test_client.get("/api/v1/items/554/price?side=sell")

        assert response.status_code == 200
        data = response.json()
        assert data["side"] == "sell"
        assert data["recommendedPrice"] == 6

    def test_price_lookup_with_window(self, client):
        """Test price lookup with custom window parameter."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = {
            "itemId": 554,
            "itemName": "Fire rune",
            "side": "buy",
            "recommendedPrice": 4,
            "currentMarketPrice": 5,
            "offsetPercent": 0.015,
            "fillProbability": 0.80,
            "expectedValue": 0.009,
            "timeWindowHours": 4,
            "volume24h": 100000,
            "trend": "Rising",
            "isRecommended": True,
            "warning": None,
            "marginGp": 2,
            "marginPercent": 0.5,
            "buyLimit": 10000,
        }

        response = test_client.get("/api/v1/items/554/price?side=buy&window=4")

        assert response.status_code == 200
        data = response.json()
        assert data["timeWindowHours"] == 4
        mock_engine.get_item_price_lookup.assert_called_once_with(
            item_id=554,
            side="buy",
            window_hours=4,
            offset_pct=None,
            include_price_history=False,
        )

    def test_price_lookup_with_offset(self, client):
        """Test price lookup with custom offset parameter."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = {
            "itemId": 554,
            "itemName": "Fire rune",
            "side": "buy",
            "recommendedPrice": 4,
            "currentMarketPrice": 5,
            "offsetPercent": 0.015,
            "fillProbability": 0.80,
            "expectedValue": 0.009,
            "timeWindowHours": 24,
            "volume24h": 100000,
            "trend": "Stable",
            "isRecommended": True,
            "warning": None,
            "marginGp": 2,
            "marginPercent": 0.5,
            "buyLimit": 10000,
        }

        response = test_client.get("/api/v1/items/554/price?side=buy&offset=0.015")

        assert response.status_code == 200
        mock_engine.get_item_price_lookup.assert_called_once_with(
            item_id=554,
            side="buy",
            window_hours=24,
            offset_pct=0.015,
            include_price_history=False,
        )

    def test_price_lookup_item_not_found(self, client):
        """Test price lookup returns 404 for unknown item."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = None

        response = test_client.get("/api/v1/items/99999999/price")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_price_lookup_with_warning(self, client):
        """Test price lookup includes warning when below thresholds."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = {
            "itemId": 123,
            "itemName": "King worm",
            "side": "buy",
            "recommendedPrice": 100,
            "currentMarketPrice": 102,
            "offsetPercent": 0.02,
            "fillProbability": 0.02,  # Very low
            "expectedValue": 0.001,  # Very low
            "timeWindowHours": 24,
            "volume24h": 50,
            "trend": "Falling",
            "isRecommended": False,
            "warning": "Low fill probability (2.0%) - Low expected value (0.10%). Consider waiting for better conditions.",
            "marginGp": 1,
            "marginPercent": 0.01,
            "buyLimit": 1000,
        }

        response = test_client.get("/api/v1/items/123/price")

        assert response.status_code == 200
        data = response.json()
        assert data["isRecommended"] is False
        assert data["warning"] is not None
        assert "Low fill probability" in data["warning"]

    def test_price_lookup_invalid_item_id(self, client):
        """Test price lookup returns 422 for invalid item_id."""
        test_client, _ = client

        # item_id must be > 0
        response = test_client.get("/api/v1/items/0/price")
        assert response.status_code == 422

    def test_price_lookup_invalid_side(self, client):
        """Test price lookup returns 422 for invalid side."""
        test_client, _ = client

        response = test_client.get("/api/v1/items/554/price?side=invalid")
        assert response.status_code == 422

    def test_price_lookup_invalid_window(self, client):
        """Test price lookup returns 422 for invalid window."""
        test_client, _ = client

        # window must be 1-48
        response = test_client.get("/api/v1/items/554/price?window=100")
        assert response.status_code == 422

    def test_price_lookup_invalid_offset(self, client):
        """Test price lookup returns 422 for invalid offset."""
        test_client, _ = client

        # offset must be 0.01-0.03
        response = test_client.get("/api/v1/items/554/price?offset=0.5")
        assert response.status_code == 422

    def test_price_lookup_with_price_history(self, client):
        """Test price lookup includes price history when requested."""
        test_client, mock_engine = client

        mock_engine.get_item_price_lookup.return_value = {
            "itemId": 554,
            "itemName": "Fire rune",
            "side": "buy",
            "recommendedPrice": 4,
            "currentMarketPrice": 5,
            "offsetPercent": 0.02,
            "fillProbability": 0.72,
            "expectedValue": 0.008,
            "timeWindowHours": 24,
            "volume24h": 100000,
            "trend": "Stable",
            "isRecommended": True,
            "warning": None,
            "marginGp": 1,
            "marginPercent": 0.25,
            "buyLimit": 25000,
            "priceHistory": [
                {"timestamp": "2024-01-01T00:00:00Z", "high": 5, "low": 4},
                {"timestamp": "2024-01-01T01:00:00Z", "high": 6, "low": 4},
            ],
        }

        response = test_client.get("/api/v1/items/554/price?include_price_history=true")

        assert response.status_code == 200
        data = response.json()
        assert data["priceHistory"] is not None
        assert len(data["priceHistory"]) == 2
        mock_engine.get_item_price_lookup.assert_called_once_with(
            item_id=554,
            side="buy",
            window_hours=24,
            offset_pct=None,
            include_price_history=True,
        )


class TestExpandAcronym:
    """Test expand_acronym function directly."""

    def test_known_acronyms(self):
        """Test expansion of known acronyms."""
        from src.prediction_loader import expand_acronym

        # Godswords
        assert expand_acronym("ags") == "armadyl godsword"
        assert expand_acronym("bgs") == "bandos godsword"
        assert expand_acronym("sgs") == "saradomin godsword"
        assert expand_acronym("zgs") == "zamorak godsword"

        # Melee weapons
        assert expand_acronym("tbow") == "twisted bow"
        assert expand_acronym("dhl") == "dragon hunter lance"
        assert expand_acronym("dhcb") == "dragon hunter crossbow"
        assert expand_acronym("sang") == "sanguinesti staff"
        assert expand_acronym("dwh") == "dragon warhammer"
        assert expand_acronym("gmaul") == "granite maul"
        assert expand_acronym("scythe") == "scythe of vitur"

        # Ranged weapons
        assert expand_acronym("bp") == "toxic blowpipe"
        assert expand_acronym("bowfa") == "bow of faerdhinen"
        assert expand_acronym("acb") == "armadyl crossbow"
        assert expand_acronym("zcb") == "zaryte crossbow"

        # Armor
        assert expand_acronym("bcp") == "bandos chestplate"
        assert expand_acronym("tassets") == "bandos tassets"
        assert expand_acronym("prims") == "primordial boots"
        assert expand_acronym("bgloves") == "barrows gloves"
        assert expand_acronym("serp") == "serpentine helm"

        # Shields
        assert expand_acronym("dfs") == "dragonfire shield"
        assert expand_acronym("ely") == "elysian spirit shield"
        assert expand_acronym("avernic") == "avernic defender"

        # Jewelry
        assert expand_acronym("fury") == "amulet of fury"
        assert expand_acronym("torture") == "amulet of torture"
        assert expand_acronym("anguish") == "necklace of anguish"
        assert expand_acronym("occult") == "occult necklace"

        # Potions
        assert expand_acronym("ppot") == "prayer potion"
        assert expand_acronym("scb") == "super combat potion"
        assert expand_acronym("sara brew") == "saradomin brew"

    def test_case_insensitive(self):
        """Test that acronym lookup is case-insensitive."""
        from src.prediction_loader import expand_acronym

        assert expand_acronym("AGS") == "armadyl godsword"
        assert expand_acronym("Ags") == "armadyl godsword"
        assert expand_acronym("aGs") == "armadyl godsword"
        assert expand_acronym("TBOW") == "twisted bow"

    def test_unknown_queries_unchanged(self):
        """Test that unknown queries are returned unchanged."""
        from src.prediction_loader import expand_acronym

        assert expand_acronym("dragon") == "dragon"
        assert expand_acronym("fire rune") == "fire rune"
        assert expand_acronym("twisted bow") == "twisted bow"

    def test_whitespace_trimmed(self):
        """Test that leading/trailing whitespace is trimmed."""
        from src.prediction_loader import expand_acronym

        assert expand_acronym("  ags  ") == "armadyl godsword"
        assert expand_acronym(" tbow ") == "twisted bow"

    def test_empty_string(self):
        """Test handling of empty string."""
        from src.prediction_loader import expand_acronym

        assert expand_acronym("") == ""
        assert expand_acronym("   ") == "   "  # Only spaces not in dict


class TestWebFrontendEndpoints:
    """Test cases for web frontend compatibility endpoints (Issue #147)."""

    @pytest.fixture
    def client(self, sample_registry, mock_db_connection):
        """Create test client with mocked engine."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [{"status": "ok", "component": "test"}],
                    "timestamp": "2026-01-08T12:00:00Z",
                    "supported_items": 2,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                mock_engine.get_recommendations.return_value = []
                mock_engine.get_recommendation_by_item_id.return_value = None
                mock_engine.search_items_by_name.return_value = []
                MockEngine.return_value = mock_engine

                from src.api import app, limiter
                import src.api as api_module

                api_module.app.state.runtime.engine = mock_engine
                limiter.reset()

                yield TestClient(app), mock_engine

    def test_health_alias_endpoint(self, client):
        """Test /health alias returns same as /api/v1/health."""
        test_client, mock_engine = client
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "checks" in data

    def test_recommendations_web_endpoint(self, client):
        """Test /recommendations web frontend endpoint."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.72,
                "fillConfidence": "Good",
                "trend": "Stable",
            }
        ]

        response = test_client.get(
            "/recommendations?capital=10000000&style=active&risk=medium&count=4"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["itemId"] == 554

    def test_recommendations_web_with_user_hash(self, client):
        """Test /recommendations with user_hash parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        user_hash = "a" * 64  # SHA256 hash
        response = test_client.get(
            f"/recommendations?capital=10000000&user_hash={user_hash}"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("user_id") == user_hash

    def test_recommendations_web_with_exclude_items(self, client):
        """Test /recommendations with exclude_items parameter."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/recommendations?capital=10000000&exclude_items=554,565"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("exclude_item_ids") == {554, 565}

    def test_recommendations_web_with_offset_filters(self, client):
        """Test /recommendations with min/max offset parameters."""
        test_client, mock_engine = client
        mock_engine.get_recommendations.return_value = []

        response = test_client.get(
            "/recommendations?capital=10000000&min_offset_pct=0.015&max_offset_pct=0.02"
        )

        assert response.status_code == 200
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs.get("min_offset_pct") == 0.015
        assert call_kwargs.get("max_offset_pct") == 0.02

    def test_item_web_endpoint(self, client):
        """Test /item/{item_id} web frontend endpoint."""
        test_client, mock_engine = client
        mock_engine.get_recommendation_by_item_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
            "capitalRequired": 4000,
            "expectedProfit": 200,
            "expectedHours": 4,
            "confidence": "high",
            "fillProbability": 0.72,
            "fillConfidence": "Good",
            "trend": "Stable",
        }

        response = test_client.get("/item/554")

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 554

    def test_item_web_with_user_context(self, client):
        """Test /item/{item_id} with user context parameters."""
        test_client, mock_engine = client
        mock_engine.get_recommendation_by_item_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
            "capitalRequired": 4000,
            "expectedProfit": 200,
            "expectedHours": 4,
            "confidence": "high",
            "fillProbability": 0.72,
            "fillConfidence": "Good",
            "trend": "Stable",
        }

        response = test_client.get(
            "/item/554?capital=5000000&risk=low&style=passive&include_price_history=true"
        )

        assert response.status_code == 200
        mock_engine.get_recommendation_by_item_id.assert_called_once_with(
            item_id=554,
            capital=5000000,
            risk="low",
            style="passive",
            slots=None,
            include_price_history=True,
        )

    def test_item_web_not_found(self, client):
        """Test /item/{item_id} returns 404 when item not found."""
        test_client, mock_engine = client
        mock_engine.get_recommendation_by_item_id.return_value = None

        response = test_client.get("/item/999999")

        assert response.status_code == 404

    def test_search_items_web_endpoint(self, client):
        """Test /search-items web frontend endpoint."""
        test_client, mock_engine = client
        mock_engine.search_items_by_name.return_value = [
            {"item_id": 554, "item_name": "Fire rune"},
            {"item_id": 555, "item_name": "Water rune"},
        ]

        response = test_client.get("/search-items?q=rune&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["itemId"] == 554
        assert data[0]["name"] == "Fire rune"

    def test_search_items_web_validation(self, client):
        """Test /search-items validates query parameter."""
        test_client, mock_engine = client

        # Missing query parameter
        response = test_client.get("/search-items")
        assert response.status_code == 422

    def test_trade_outcome_web_endpoint(self, client, mock_db_connection):
        """Test /trade-outcome web frontend endpoint."""
        test_client, mock_engine = client

        # We need to mock the outcome database for this test
        with patch("src.api.outcome_db_engine") as mock_outcome_db:
            mock_conn = MagicMock()
            mock_outcome_db.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_outcome_db.connect.return_value.__exit__ = MagicMock(return_value=None)

            import src.api as api_module

            api_module.app.state.runtime.outcome_db_engine = mock_outcome_db

            response = test_client.post(
                "/trade-outcome",
                json={
                    "rec_id": "rec_554_2026010112",
                    "userId": "a" * 64,  # Valid SHA256 hash
                    "itemId": 554,
                    "itemName": "Fire rune",
                    "buyPrice": 4,
                    "sellPrice": 6,
                    "quantity": 1000,
                    "actualProfit": 2000,
                    "reportedAt": "2026-01-08T12:00:00Z",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_trade_outcome_web_invalid_user_id(self, client):
        """Test /trade-outcome rejects invalid user ID."""
        test_client, mock_engine = client

        with patch("src.api.outcome_db_engine") as mock_outcome_db:
            mock_outcome_db.__bool__ = MagicMock(return_value=True)

            import src.api as api_module

            api_module.app.state.runtime.outcome_db_engine = mock_outcome_db

            response = test_client.post(
                "/trade-outcome",
                json={
                    "rec_id": "rec_554_2026010112",
                    "userId": "invalid_hash",  # Not a valid SHA256
                    "itemId": 554,
                    "itemName": "Fire rune",
                    "buyPrice": 4,
                    "sellPrice": 6,
                    "quantity": 1000,
                    "actualProfit": 2000,
                    "reportedAt": "2026-01-08T12:00:00Z",
                },
            )

            assert response.status_code == 400
            assert "SHA256" in response.json()["detail"]

    def test_root_includes_web_endpoints(self, client):
        """Test root endpoint lists web frontend endpoints."""
        test_client, _ = client
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "web_frontend_endpoints" in data
        assert "recommendations" in data["web_frontend_endpoints"]
        assert "item" in data["web_frontend_endpoints"]
        assert "search" in data["web_frontend_endpoints"]
        assert "trade_outcome" in data["web_frontend_endpoints"]
        assert "health" in data["web_frontend_endpoints"]


class TestTradeUpdatesEndpoint:
    """Test cases for trade updates polling endpoint (Issue #148)."""

    @pytest.fixture
    def client(self, sample_registry, mock_db_connection):
        """Create test client with mocked engine."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [{"status": "ok", "component": "test"}],
                    "timestamp": "2026-01-08T12:00:00Z",
                    "supported_items": 2,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                mock_engine.get_recommendation_by_id.return_value = None
                mock_engine.evaluate_active_order.return_value = {
                    "action": "wait",
                    "confidence": 0.75,
                    "current_fill_probability": 0.65,
                    "recommendations": {"wait": {"estimated_fill_time_minutes": 30}},
                    "reasoning": "Price is stable, wait for fill",
                }
                MockEngine.return_value = mock_engine

                from src.api import app, limiter
                import src.api as api_module

                api_module.app.state.runtime.engine = mock_engine
                limiter.reset()

                yield TestClient(app), mock_engine

    def test_trade_updates_no_updates(self, client):
        """Test trade updates returns empty when all trades are HOLD."""
        test_client, mock_engine = client

        # Mock recommendation exists but evaluation returns "wait"
        mock_engine.get_recommendation_by_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
        }
        mock_engine.evaluate_active_order.return_value = {
            "action": "wait",
            "confidence": 0.75,
            "current_fill_probability": 0.65,
            "recommendations": {"wait": {"estimated_fill_time_minutes": 30}},
            "reasoning": "Price is stable, wait for fill",
        }

        response = test_client.get("/api/v1/trades/updates?tradeIds=rec_554_123")

        assert response.status_code == 200
        data = response.json()
        assert data["updates"] == []
        assert data["nextCheckIn"] == 60  # No updates, longer interval

    def test_trade_updates_with_adjust_price(self, client):
        """Test trade updates returns ADJUST_PRICE update."""
        test_client, mock_engine = client

        mock_engine.get_recommendation_by_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
        }
        mock_engine.evaluate_active_order.return_value = {
            "action": "adjust_price",
            "confidence": 0.85,
            "current_fill_probability": 0.45,
            "recommendations": {
                "adjust_price": {
                    "suggested_price": 5,
                    "new_fill_probability": 0.75,
                    "cost_difference": -1000,
                }
            },
            "reasoning": "Market shifted, reduce sell price for faster fill",
        }

        response = test_client.get("/api/v1/trades/updates?tradeIds=rec_554_123")

        assert response.status_code == 200
        data = response.json()
        assert len(data["updates"]) == 1
        update = data["updates"][0]
        assert update["type"] == "ADJUST_PRICE"
        assert update["tradeId"] == "rec_554_123"
        assert update["newSellPrice"] == 5
        assert update["originalSellPrice"] == 6
        assert update["urgency"] == "high"  # confidence >= 0.85
        assert data["nextCheckIn"] == 15  # High urgency, short interval

    def test_trade_updates_with_switch_item(self, client):
        """Test trade updates returns SWITCH_ITEM update."""
        test_client, mock_engine = client

        mock_engine.get_recommendation_by_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
        }
        mock_engine.evaluate_active_order.return_value = {
            "action": "abort_retry",
            "confidence": 0.92,
            "current_fill_probability": 0.25,
            "recommendations": {
                "abort_retry": {
                    "alternative_items": [
                        {
                            "item_id": 565,
                            "item_name": "Water rune",
                            "expected_profit": 5000,
                            "fill_probability": 0.85,
                            "expected_hours": 4,
                        }
                    ]
                }
            },
            "reasoning": "Better opportunity found with higher EV",
        }

        response = test_client.get("/api/v1/trades/updates?tradeIds=rec_554_123")

        assert response.status_code == 200
        data = response.json()
        assert len(data["updates"]) == 1
        update = data["updates"][0]
        assert update["type"] == "SWITCH_ITEM"
        assert update["newItem"]["itemId"] == 565
        assert update["newItem"]["item"] == "Water rune"
        assert update["newItem"]["expectedProfit"] == 5000

    def test_trade_updates_with_sell_now(self, client):
        """Test trade updates returns SELL_NOW (liquidate) update."""
        test_client, mock_engine = client

        mock_engine.get_recommendation_by_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
        }
        mock_engine.evaluate_active_order.return_value = {
            "action": "liquidate",
            "confidence": 0.78,
            "current_fill_probability": 0.15,
            "recommendations": {"liquidate": {"instant_price": 5, "loss_amount": 1000}},
            "reasoning": "Market unfavorable, sell now to minimize loss",
        }

        response = test_client.get("/api/v1/trades/updates?tradeIds=rec_554_123")

        assert response.status_code == 200
        data = response.json()
        assert len(data["updates"]) == 1
        update = data["updates"][0]
        assert update["type"] == "SELL_NOW"
        assert update["adjustedSellPrice"] == 5
        assert update["urgency"] == "medium"  # 0.65 <= confidence < 0.85

    def test_trade_updates_multiple_trades(self, client):
        """Test trade updates handles multiple trades."""
        test_client, mock_engine = client

        # First trade: wait (no update)
        # Second trade: adjust_price (should return update)
        call_count = [0]

        def mock_get_rec(rec_id):
            return {
                "id": rec_id,
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
            }

        def mock_evaluate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "action": "wait",
                    "confidence": 0.75,
                    "current_fill_probability": 0.65,
                    "recommendations": {"wait": {"estimated_fill_time_minutes": 30}},
                    "reasoning": "Wait for fill",
                }
            else:
                return {
                    "action": "adjust_price",
                    "confidence": 0.70,
                    "current_fill_probability": 0.45,
                    "recommendations": {
                        "adjust_price": {
                            "suggested_price": 5,
                            "new_fill_probability": 0.75,
                            "cost_difference": -1000,
                        }
                    },
                    "reasoning": "Adjust price",
                }

        mock_engine.get_recommendation_by_id.side_effect = mock_get_rec
        mock_engine.evaluate_active_order.side_effect = mock_evaluate

        response = test_client.get(
            "/api/v1/trades/updates?tradeIds=rec_554_123,rec_554_456"
        )

        assert response.status_code == 200
        data = response.json()
        # Only second trade should have an update (first is HOLD)
        assert len(data["updates"]) == 1
        assert data["updates"][0]["tradeId"] == "rec_554_456"

    def test_trade_updates_missing_trade(self, client):
        """Test trade updates handles missing trades gracefully."""
        test_client, mock_engine = client
        mock_engine.get_recommendation_by_id.return_value = None

        response = test_client.get("/api/v1/trades/updates?tradeIds=rec_nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data["updates"] == []
        assert data["nextCheckIn"] == 60

    def test_trade_updates_empty_trade_ids(self, client):
        """Test trade updates rejects empty trade IDs."""
        test_client, _ = client

        response = test_client.get("/api/v1/trades/updates?tradeIds=")

        assert response.status_code == 400
        assert "No valid trade IDs" in response.json()["detail"]

    def test_trade_updates_too_many_trades(self, client):
        """Test trade updates rejects more than 20 trade IDs."""
        test_client, _ = client

        trade_ids = ",".join([f"rec_{i}_123" for i in range(25)])
        response = test_client.get(f"/api/v1/trades/updates?tradeIds={trade_ids}")

        assert response.status_code == 400
        assert "Maximum 20" in response.json()["detail"]

    def test_trade_updates_invalid_user_id(self, client):
        """Test trade updates rejects invalid user ID."""
        test_client, _ = client

        response = test_client.get(
            "/api/v1/trades/updates?tradeIds=rec_554_123&user_id=invalid"
        )

        assert response.status_code == 400
        assert "SHA256" in response.json()["detail"]

    def test_trade_updates_with_valid_user_id(self, client):
        """Test trade updates accepts valid user ID."""
        test_client, mock_engine = client
        mock_engine.get_recommendation_by_id.return_value = {
            "id": "rec_554_123",
            "itemId": 554,
            "item": "Fire rune",
            "buyPrice": 4,
            "sellPrice": 6,
            "quantity": 1000,
        }
        mock_engine.evaluate_active_order.return_value = {
            "action": "wait",
            "confidence": 0.75,
            "current_fill_probability": 0.65,
            "recommendations": {"wait": {"estimated_fill_time_minutes": 30}},
            "reasoning": "Wait for fill",
        }

        user_id = "a" * 64
        response = test_client.get(
            f"/api/v1/trades/updates?tradeIds=rec_554_123&user_id={user_id}"
        )

        assert response.status_code == 200
        # Verify user_id was passed to evaluate_active_order
        mock_engine.evaluate_active_order.assert_called_once()
        call_kwargs = mock_engine.evaluate_active_order.call_args[1]
        assert call_kwargs.get("user_id") == user_id


class TestPriceHistoryEndpoint:
    """Test cases for price history endpoint (Issue #152)."""

    @pytest.fixture
    def client(self, sample_registry, mock_db_connection):
        """Create test client with mocked engine."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.loader = MagicMock()
                mock_engine.loader.get_extended_price_history.return_value = []
                mock_engine.loader.get_item_name.return_value = "Fire rune"
                mock_engine.loader.get_item_trend.return_value = "Stable"
                MockEngine.return_value = mock_engine

                from src.api import app, limiter
                import src.api as api_module

                api_module.app.state.runtime.engine = mock_engine
                limiter.reset()

                yield TestClient(app), mock_engine

    def test_price_history_success(self, client):
        """Test successful price history retrieval."""
        test_client, mock_engine = client

        mock_engine.loader.get_extended_price_history.return_value = [
            {
                "timestamp": "2024-01-15T10:00:00",
                "high": 5,
                "low": 4,
                "avgHigh": 5,
                "avgLow": 4,
            },
            {
                "timestamp": "2024-01-15T11:00:00",
                "high": 6,
                "low": 4,
                "avgHigh": 6,
                "avgLow": 4,
            },
        ]
        mock_engine.loader.get_item_name.return_value = "Fire rune"
        mock_engine.loader.get_item_trend.return_value = "Rising"

        response = test_client.get("/api/v1/items/554/price-history")

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 554
        assert data["itemName"] == "Fire rune"
        assert data["trend"] == "Rising"
        assert len(data["history"]) == 2
        assert data["history"][0]["high"] == 5
        assert data["history"][0]["low"] == 4

    def test_price_history_with_hours_param(self, client):
        """Test price history with custom hours parameter."""
        test_client, mock_engine = client

        mock_engine.loader.get_extended_price_history.return_value = [
            {
                "timestamp": "2024-01-15T10:00:00",
                "high": 5,
                "low": 4,
                "avgHigh": 5,
                "avgLow": 4,
            }
        ]
        mock_engine.loader.get_item_name.return_value = "Fire rune"
        mock_engine.loader.get_item_trend.return_value = "Stable"

        response = test_client.get("/api/v1/items/554/price-history?hours=48")

        assert response.status_code == 200
        mock_engine.loader.get_extended_price_history.assert_called_once_with(
            554, hours=48
        )

    def test_price_history_item_not_found(self, client):
        """Test price history for non-existent item."""
        test_client, mock_engine = client

        mock_engine.loader.get_extended_price_history.return_value = []
        mock_engine.loader.get_item_name.return_value = None

        response = test_client.get("/api/v1/items/999999/price-history")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_price_history_empty_history(self, client):
        """Test price history returns empty for item with no price data."""
        test_client, mock_engine = client

        mock_engine.loader.get_extended_price_history.return_value = []
        mock_engine.loader.get_item_name.return_value = "Fire rune"

        response = test_client.get("/api/v1/items/554/price-history")

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 554
        assert data["itemName"] == "Fire rune"
        assert data["history"] == []
        assert data["trend"] == "Stable"
        assert data["change24h"] == 0.0

    def test_price_history_calculates_change(self, client):
        """Test 24h change percentage calculation."""
        test_client, mock_engine = client

        # 10% increase: first avg = 100, last avg = 110
        mock_engine.loader.get_extended_price_history.return_value = [
            {
                "timestamp": "2024-01-15T10:00:00",
                "high": 110,
                "low": 90,
                "avgHigh": 110,
                "avgLow": 90,
            },
            {
                "timestamp": "2024-01-15T11:00:00",
                "high": 120,
                "low": 100,
                "avgHigh": 120,
                "avgLow": 100,
            },
        ]
        mock_engine.loader.get_item_name.return_value = "Fire rune"
        mock_engine.loader.get_item_trend.return_value = "Rising"

        response = test_client.get("/api/v1/items/554/price-history")

        assert response.status_code == 200
        data = response.json()
        # First avg = (110+90)/2 = 100
        # Last avg = (120+100)/2 = 110
        # Change = (110-100)/100 * 100 = 10%
        assert data["change24h"] == 10.0

    def test_price_history_hours_validation(self, client):
        """Test hours parameter validation."""
        test_client, _ = client

        # Hours too high
        response = test_client.get("/api/v1/items/554/price-history?hours=200")
        assert response.status_code == 422

        # Hours too low
        response = test_client.get("/api/v1/items/554/price-history?hours=0")
        assert response.status_code == 422

    def test_price_history_trend_types(self, client):
        """Test all trend types are handled."""
        test_client, mock_engine = client

        mock_engine.loader.get_extended_price_history.return_value = [
            {
                "timestamp": "2024-01-15T10:00:00",
                "high": 5,
                "low": 4,
                "avgHigh": 5,
                "avgLow": 4,
            }
        ]
        mock_engine.loader.get_item_name.return_value = "Fire rune"

        for trend in ["Rising", "Stable", "Falling"]:
            mock_engine.loader.get_item_trend.return_value = trend
            response = test_client.get("/api/v1/items/554/price-history")
            assert response.status_code == 200
            assert response.json()["trend"] == trend


class TestCORSConfiguration:
    """Test cases for CORS configuration (Issue #149)."""

    def test_cors_config_default(self):
        """Test default CORS origins are localhost only."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove CORS env vars if present
            import os

            env_backup = {}
            for key in ["CORS_ORIGINS", "CORS_ORIGIN_REGEX"]:
                if key in os.environ:
                    env_backup[key] = os.environ.pop(key)

            try:
                # Re-import to get fresh config
                import importlib
                import src.config as config_module

                importlib.reload(config_module)

                # Default should be localhost
                assert "http://localhost:3000" in config_module.config.cors_origins
                assert "http://localhost:8080" in config_module.config.cors_origins
                assert config_module.config.cors_origin_regex == ""
            finally:
                # Restore env vars
                for key, value in env_backup.items():
                    os.environ[key] = value
                importlib.reload(config_module)

    def test_cors_config_with_origins(self):
        """Test CORS origins from environment variable."""
        test_origins = "https://gept.gg,https://www.gept.gg,http://localhost:3000"

        with patch.dict("os.environ", {"CORS_ORIGINS": test_origins}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert "https://gept.gg" in config_module.config.cors_origins
            assert "https://www.gept.gg" in config_module.config.cors_origins
            assert "http://localhost:3000" in config_module.config.cors_origins
            assert len(config_module.config.cors_origins) == 3

            # Reload to reset
            import os

            if "CORS_ORIGINS" in os.environ:
                del os.environ["CORS_ORIGINS"]
            importlib.reload(config_module)

    def test_cors_config_with_regex(self):
        """Test CORS regex pattern from environment variable."""
        test_regex = r"https://gept-gg-.*\.vercel\.app"

        with patch.dict("os.environ", {"CORS_ORIGIN_REGEX": test_regex}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.cors_origin_regex == test_regex

            # Reload to reset
            import os

            if "CORS_ORIGIN_REGEX" in os.environ:
                del os.environ["CORS_ORIGIN_REGEX"]
            importlib.reload(config_module)


class TestAPIAuthentication:
    """Test cases for API key authentication (Issue #150)."""

    @pytest.fixture
    def client_with_auth(self, sample_registry, mock_db_connection):
        """Create test client with API key authentication enabled."""
        test_api_key = "test-secret-api-key-12345"
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
                "INTERNAL_API_KEY": test_api_key,
            },
        ):
            # Reload config to pick up the new env var
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [{"status": "ok", "component": "test"}],
                    "timestamp": "2026-01-08T12:00:00Z",
                    "supported_items": 2,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                mock_engine.get_recommendations.return_value = []
                mock_engine.get_recommendation_by_item_id.return_value = None
                mock_engine.search_items_by_name.return_value = []
                MockEngine.return_value = mock_engine

                # Reload api module to pick up config changes
                import src.api as api_module

                importlib.reload(api_module)
                api_module.app.state.runtime.engine = mock_engine

                # Reset rate limiter storage
                api_module.limiter.reset()

                yield TestClient(api_module.app), mock_engine, test_api_key

            # Clean up
            if "INTERNAL_API_KEY" in os.environ:
                del os.environ["INTERNAL_API_KEY"]
            importlib.reload(config_module)
            importlib.reload(api_module)

    def test_health_endpoint_public_with_auth_enabled(self, client_with_auth):
        """Test health endpoint is accessible without API key even when auth is enabled."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_alias_public_with_auth_enabled(self, client_with_auth):
        """Test /health alias is accessible without API key."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_recommendations_requires_api_key(self, client_with_auth):
        """Test recommendations endpoint requires API key when auth is enabled."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&style=active"
        )

        assert response.status_code == 401
        data = response.json()
        assert "Missing API key" in data["detail"]

    def test_recommendations_with_valid_api_key(self, client_with_auth):
        """Test recommendations endpoint works with valid API key."""
        test_client, mock_engine, api_key = client_with_auth
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&style=active",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_recommendations_with_invalid_api_key(self, client_with_auth):
        """Test recommendations endpoint rejects invalid API key."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get(
            "/api/v1/recommendations?capital=10000000&style=active",
            headers={"X-API-Key": "wrong-api-key"},
        )

        assert response.status_code == 401
        data = response.json()
        assert "Invalid API key" in data["detail"]

    def test_item_search_requires_api_key(self, client_with_auth):
        """Test item search endpoint requires API key."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get("/api/v1/items/search?q=fire")

        assert response.status_code == 401

    def test_item_search_with_valid_api_key(self, client_with_auth):
        """Test item search endpoint works with valid API key."""
        test_client, mock_engine, api_key = client_with_auth
        response = test_client.get(
            "/api/v1/items/search?q=fire",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200

    def test_web_endpoint_requires_api_key(self, client_with_auth):
        """Test web frontend endpoint requires API key."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get("/recommendations?capital=10000000")

        assert response.status_code == 401

    def test_web_endpoint_with_valid_api_key(self, client_with_auth):
        """Test web frontend endpoint works with valid API key."""
        test_client, mock_engine, api_key = client_with_auth
        response = test_client.get(
            "/recommendations?capital=10000000",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200

    def test_validate_rejects_missing_api_key(self):
        """Test config.validate() rejects missing INTERNAL_API_KEY."""
        import importlib
        import os

        env_backup = os.environ.get("INTERNAL_API_KEY")
        if "INTERNAL_API_KEY" in os.environ:
            del os.environ["INTERNAL_API_KEY"]

        try:
            import src.config as config_module
            importlib.reload(config_module)

            errors = config_module.config.validate()
            assert any("INTERNAL_API_KEY" in e for e in errors)
        finally:
            if env_backup is not None:
                os.environ["INTERNAL_API_KEY"] = env_backup
            import src.config as config_module
            importlib.reload(config_module)

    def test_validate_rejects_missing_webhook_config_when_monitor_enabled(self):
        """Test config.validate() rejects missing webhook config when monitor enabled."""
        import importlib
        import os

        with patch.dict(
            os.environ,
            {
                "PRICE_DROP_MONITOR_ENABLED": "true",
                "WEBHOOK_SECRET": "",
                "WEB_APP_WEBHOOK_URL": "",
                "INTERNAL_API_KEY": "test-api-key",
                "DB_CONNECTION_STRING": "postgresql://test:test@localhost:5432/test",
            },
            clear=False,
        ):
            import src.config as config_module
            importlib.reload(config_module)

            errors = config_module.config.validate()
            assert any("WEBHOOK_SECRET" in e for e in errors)
            assert any("WEB_APP_WEBHOOK_URL" in e for e in errors)

    def test_validate_allows_missing_webhook_config_when_monitor_disabled(self):
        """Test config.validate() allows missing webhook config when monitor disabled."""
        import importlib
        import os

        with patch.dict(
            os.environ,
            {
                "PRICE_DROP_MONITOR_ENABLED": "false",
                "TRADE_WEBHOOKS_ENABLED": "false",
                "WEBHOOK_SECRET": "",
                "WEB_APP_WEBHOOK_URL": "",
                "INTERNAL_API_KEY": "test-api-key",
                "DB_CONNECTION_STRING": "postgresql://test:test@localhost:5432/test",
            },
            clear=False,
        ):
            import src.config as config_module
            importlib.reload(config_module)

            errors = config_module.config.validate()
            assert not any("WEBHOOK_SECRET" in e for e in errors)
            assert not any("WEB_APP_WEBHOOK_URL" in e for e in errors)

    def test_validate_rejects_missing_webhook_secret_when_trade_webhooks_enabled(self):
        """Test config.validate() rejects missing webhook secret when trade webhooks enabled."""
        import importlib
        import os

        with patch.dict(
            os.environ,
            {
                "PRICE_DROP_MONITOR_ENABLED": "false",
                "TRADE_WEBHOOKS_ENABLED": "true",
                "WEBHOOK_SECRET": "",
                "WEB_APP_WEBHOOK_URL": "",
                "INTERNAL_API_KEY": "test-api-key",
                "DB_CONNECTION_STRING": "postgresql://test:test@localhost:5432/test",
            },
            clear=False,
        ):
            import src.config as config_module
            importlib.reload(config_module)

            errors = config_module.config.validate()
            assert any("WEBHOOK_SECRET" in e for e in errors)
            assert not any("WEB_APP_WEBHOOK_URL" in e for e in errors)

    def test_root_endpoint_public(self, client_with_auth):
        """Test root endpoint is always public."""
        test_client, mock_engine, _ = client_with_auth
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data

    def test_config_internal_api_key(self):
        """Test INTERNAL_API_KEY config field."""
        test_key = "my-secret-key-abcd1234"

        with patch.dict("os.environ", {"INTERNAL_API_KEY": test_key}, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.internal_api_key == test_key

            # Clean up
            import os

            if "INTERNAL_API_KEY" in os.environ:
                del os.environ["INTERNAL_API_KEY"]
            importlib.reload(config_module)

    def test_config_internal_api_key_default_empty(self):
        """Test INTERNAL_API_KEY defaults to empty string (caught by validate)."""
        import os

        # Ensure env var is not set
        env_backup = os.environ.get("INTERNAL_API_KEY")
        if "INTERNAL_API_KEY" in os.environ:
            del os.environ["INTERNAL_API_KEY"]

        try:
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.internal_api_key == ""
            # validate() should flag this as an error
            errors = config_module.config.validate()
            assert any("INTERNAL_API_KEY" in e for e in errors)
        finally:
            # Restore env var if it was set
            if env_backup is not None:
                os.environ["INTERNAL_API_KEY"] = env_backup
            import src.config as config_module

            importlib.reload(config_module)


class TestRateLimiting:
    """Test cases for configurable rate limiting (Issue #154)."""

    def test_get_rate_limit_key_user_hash_header(self):
        """Test rate limit key from X-User-Hash header."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {"X-User-Hash": "a" * 64}  # Valid SHA256 hash
        mock_request.query_params = {}
        mock_request.client = None

        key = get_rate_limit_key(mock_request)
        assert key == f"user:{'a' * 64}"

    def test_get_rate_limit_key_user_id_query_param(self):
        """Test rate limit key from user_id query parameter."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {"user_id": "b" * 64}
        mock_request.client = None

        key = get_rate_limit_key(mock_request)
        assert key == f"user:{'b' * 64}"

    def test_get_rate_limit_key_x_forwarded_for_trusted_proxy(self):
        """Test rate limit key from X-Forwarded-For when client is a trusted proxy."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8, 9.10.11.12"}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"  # trusted proxy

        key = get_rate_limit_key(mock_request)
        assert key == "ip:1.2.3.4"

    def test_get_rate_limit_key_x_forwarded_for_untrusted(self):
        """Test X-Forwarded-For is ignored when client is not a trusted proxy."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {"X-Forwarded-For": "1.2.3.4"}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "10.0.0.50"  # not a trusted proxy

        key = get_rate_limit_key(mock_request)
        # Should use direct client IP, not the spoofed forwarded header
        assert key == "ip:10.0.0.50"

    def test_get_rate_limit_key_client_ip(self):
        """Test rate limit key from client IP."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"

        key = get_rate_limit_key(mock_request)
        assert key == "ip:192.168.1.1"

    def test_get_rate_limit_key_fallback_unknown(self):
        """Test rate limit key fallback to unknown."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {}
        mock_request.query_params = {}
        mock_request.client = None

        key = get_rate_limit_key(mock_request)
        assert key == "ip:unknown"

    def test_get_rate_limit_key_priority_user_over_ip(self):
        """Test user hash takes priority over IP."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {
            "X-User-Hash": "c" * 64,
            "X-Forwarded-For": "1.2.3.4",
        }
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"

        key = get_rate_limit_key(mock_request)
        assert key == f"user:{'c' * 64}"

    def test_get_rate_limit_key_invalid_hash_falls_through(self):
        """Test invalid hash (wrong length) falls through to IP."""
        from src.api import get_rate_limit_key

        mock_request = MagicMock()
        mock_request.headers = {"X-User-Hash": "tooshort"}
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "192.168.1.1"

        key = get_rate_limit_key(mock_request)
        assert key == "ip:192.168.1.1"

    def test_rate_limit_config_defaults(self):
        """Test rate limit config fields have proper defaults."""
        import os

        # Save current env vars
        env_backup = {}
        for key in [
            "RATE_LIMIT_RECOMMENDATIONS",
            "RATE_LIMIT_SEARCH",
            "RATE_LIMIT_ITEMS",
            "RATE_LIMIT_HEALTH",
            "RATE_LIMIT_TRADE_UPDATES",
            "RATE_LIMIT_OUTCOMES",
        ]:
            env_backup[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]

        try:
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            # Check defaults
            assert config_module.config.rate_limit_recommendations == "60/minute"
            assert config_module.config.rate_limit_search == "10/second;200/minute"
            assert config_module.config.rate_limit_items == "120/minute"
            assert config_module.config.rate_limit_health == "60/minute"
            assert config_module.config.rate_limit_trade_updates == "120/minute"
            assert config_module.config.rate_limit_outcomes == "30/minute"
        finally:
            # Restore env vars
            for key, value in env_backup.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]
            importlib.reload(config_module)

    def test_rate_limit_config_custom_values(self):
        """Test rate limit config fields accept custom values."""
        custom_values = {
            "RATE_LIMIT_RECOMMENDATIONS": "100/minute",
            "RATE_LIMIT_SEARCH": "20/second;400/minute",
            "RATE_LIMIT_ITEMS": "200/minute",
            "RATE_LIMIT_HEALTH": "120/minute",
            "RATE_LIMIT_TRADE_UPDATES": "180/minute",
            "RATE_LIMIT_OUTCOMES": "60/minute",
        }

        with patch.dict("os.environ", custom_values, clear=False):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.rate_limit_recommendations == "100/minute"
            assert config_module.config.rate_limit_search == "20/second;400/minute"
            assert config_module.config.rate_limit_items == "200/minute"
            assert config_module.config.rate_limit_health == "120/minute"
            assert config_module.config.rate_limit_trade_updates == "180/minute"
            assert config_module.config.rate_limit_outcomes == "60/minute"

            # Clean up
            importlib.reload(config_module)


class TestEnhancedHealthChecks:
    """Test cases for enhanced health check endpoints (Issue #155)."""

    @pytest.fixture
    def client_with_engine(self, sample_registry, mock_db_connection):
        """Create test client with engine initialized."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565}
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [
                        {
                            "status": "ok",
                            "component": "database",
                            "prediction_age_seconds": 120,
                        }
                    ],
                    "timestamp": "2026-01-17T12:00:00Z",
                    "supported_items": 2,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                MockEngine.return_value = mock_engine

                import src.api as api_module

                importlib.reload(api_module)
                api_module.app.state.runtime.engine = mock_engine
                api_module.app.state.runtime.is_ready = True
                api_module.app.state.runtime.startup_time = time.time() - 3600  # 1 hour ago

                # Reset rate limiter storage
                api_module.limiter.reset()

                yield TestClient(api_module.app), mock_engine

            importlib.reload(config_module)

    def test_healthz_liveness_endpoint(self, client_with_engine):
        """Test /healthz lightweight liveness probe."""
        test_client, _ = client_with_engine
        response = test_client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_healthz_no_database_check(self, sample_registry, mock_db_connection):
        """Test /healthz responds even without engine initialized."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            import importlib
            import src.api as api_module
            import src.config as config_module

            importlib.reload(config_module)
            importlib.reload(api_module)
            api_module.app.state.runtime.engine = None  # Simulate engine not initialized
            api_module.limiter.reset()

            client = TestClient(api_module.app)
            response = client.get("/healthz")

            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    def test_ready_endpoint_all_ok(self, client_with_engine):
        """Test /ready returns ok when all checks pass."""
        test_client, _ = client_with_engine
        response = test_client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "database" in data["checks"]
        assert "predictions" in data["checks"]
        assert "redis" in data["checks"]
        assert data["checks"]["database"]["status"] == "ok"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_ready_endpoint_degraded_stale_predictions(
        self, sample_registry, mock_db_connection
    ):
        """Test /ready returns degraded when predictions are stale."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
                "PREDICTION_STALE_SECONDS": "60",  # 1 minute threshold
            },
        ):
            import importlib
            import src.config as config_module

            importlib.reload(config_module)

            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [
                        {
                            "status": "ok",
                            "component": "database",
                            "prediction_age_seconds": 300,  # 5 minutes, exceeds threshold
                        }
                    ],
                    "timestamp": "2026-01-17T12:00:00Z",
                    "supported_items": 2,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                MockEngine.return_value = mock_engine

                import src.api as api_module

                importlib.reload(api_module)
                api_module.app.state.runtime.engine = mock_engine
                api_module.app.state.runtime.is_ready = True
                api_module.app.state.runtime.startup_time = time.time()
                api_module.limiter.reset()

                client = TestClient(api_module.app)
                response = client.get("/ready")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "degraded"
                assert data["checks"]["predictions"]["status"] == "warning"

            importlib.reload(config_module)

    def test_startup_endpoint_ready(self, client_with_engine):
        """Test /startup returns ok when ready."""
        test_client, _ = client_with_engine
        response = test_client.get("/startup")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["message"] == "Application ready"

    def test_startup_endpoint_not_ready(self, sample_registry, mock_db_connection):
        """Test /startup returns 503 when not ready."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            import importlib
            import src.api as api_module
            import src.config as config_module

            importlib.reload(config_module)
            importlib.reload(api_module)
            api_module.app.state.runtime.is_ready = False  # Simulate still starting
            api_module.limiter.reset()

            client = TestClient(api_module.app)
            response = client.get("/startup")

            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "starting"

    def test_health_includes_uptime(self, client_with_engine):
        """Test /api/v1/health includes uptime_seconds."""
        test_client, _ = client_with_engine
        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_includes_version(self, client_with_engine):
        """Test /api/v1/health includes version."""
        test_client, _ = client_with_engine
        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == "2.0.0"

    def test_ready_includes_version(self, client_with_engine):
        """Test /ready includes version."""
        test_client, _ = client_with_engine
        response = test_client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "2.0.0"


class TestReturnAllRecommendations:
    """Test cases for return_all parameter (issue #184)."""

    @pytest.fixture
    def client(self, sample_registry, mock_db_connection):
        """Create test client with mocked engine."""
        with patch.dict(
            "os.environ",
            {
                "DB_CONNECTION_STRING": mock_db_connection,
                "MODEL_REGISTRY_PATH": str(sample_registry),
            },
        ):
            with patch("src.api.RecommendationEngine") as MockEngine:
                mock_engine = MagicMock()
                mock_engine.supported_items = {554, 565, 566}
                mock_engine.health_check.return_value = {
                    "status": "ok",
                    "checks": [{"status": "ok", "component": "test"}],
                    "timestamp": "2026-01-17T12:00:00Z",
                    "supported_items": 3,
                    "recommendation_store_size": 0,
                    "crowding_stats": {},
                }
                mock_engine.get_recommendations.return_value = []
                mock_engine.get_all_recommendations.return_value = []
                MockEngine.return_value = mock_engine

                from src.api import app, limiter
                import src.api as api_module

                api_module.app.state.runtime.engine = mock_engine
                limiter.reset()

                yield TestClient(app), mock_engine

    def test_post_return_all_false_uses_get_recommendations(self, client):
        """Test that return_all=false calls get_recommendations (default behavior)."""
        test_client, mock_engine = client

        mock_engine.get_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.12,
                "fillConfidence": "Good",

                "trend": "Stable",
            }
        ]

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "return_all": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Default behavior: returns list
        assert isinstance(data, list)
        mock_engine.get_recommendations.assert_called_once()
        mock_engine.get_all_recommendations.assert_not_called()

    def test_post_return_all_true_uses_get_all_recommendations(self, client):
        """Test that return_all=true calls get_all_recommendations."""
        test_client, mock_engine = client

        mock_engine.get_all_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.12,
                "fillConfidence": "Good",

                "trend": "Stable",
                "_score": 0.0842,
            },
            {
                "id": "rec_565_123",
                "itemId": 565,
                "item": "Blood rune",
                "buyPrice": 400,
                "sellPrice": 420,
                "quantity": 100,
                "capitalRequired": 40000,
                "expectedProfit": 1600,
                "expectedHours": 2,
                "confidence": "high",
                "fillProbability": 0.15,
                "fillConfidence": "Strong",

                "trend": "Stable",
                "_score": 0.0925,
            },
        ]

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "return_all": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        # return_all=true: returns AllRecommendationsResponse format
        assert "recommendations" in data
        assert "generated_at" in data
        assert "valid_until" in data
        assert "total_count" in data
        assert data["total_count"] == 2
        assert len(data["recommendations"]) == 2
        mock_engine.get_all_recommendations.assert_called_once()
        mock_engine.get_recommendations.assert_not_called()

    def test_get_return_all_true_uses_get_all_recommendations(self, client):
        """Test GET endpoint with return_all=true."""
        test_client, mock_engine = client

        mock_engine.get_all_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.12,
                "fillConfidence": "Good",

                "trend": "Stable",
                "_score": 0.0842,
            }
        ]

        response = test_client.get(
            "/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4&return_all=true"
        )

        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "generated_at" in data
        assert "valid_until" in data
        assert "total_count" in data
        mock_engine.get_all_recommendations.assert_called_once()

    def test_return_all_response_has_valid_timestamps(self, client):
        """Test that generated_at and valid_until are valid ISO timestamps."""
        test_client, mock_engine = client

        mock_engine.get_all_recommendations.return_value = []

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "return_all": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify timestamps are ISO format
        from datetime import datetime

        generated_at = datetime.fromisoformat(
            data["generated_at"].replace("Z", "+00:00")
        )
        valid_until = datetime.fromisoformat(data["valid_until"].replace("Z", "+00:00"))

        # valid_until should be 5 minutes after generated_at
        delta = valid_until - generated_at
        assert delta.total_seconds() == 300  # 5 minutes

    def test_return_all_total_count_matches_recommendations_length(self, client):
        """Test that total_count matches the actual number of recommendations."""
        test_client, mock_engine = client

        mock_engine.get_all_recommendations.return_value = [
            {
                "id": f"rec_{i}_123",
                "itemId": i,
                "item": f"Item {i}",
                "buyPrice": 100,
                "sellPrice": 110,
                "quantity": 100,
                "capitalRequired": 10000,
                "expectedProfit": 800,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.10,
                "fillConfidence": "Good",

                "trend": "Stable",
                "_score": 0.05 + i * 0.01,
            }
            for i in range(5)
        ]

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                "return_all": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == len(data["recommendations"])
        assert data["total_count"] == 5

    def test_return_all_skips_slot_validation(self, client):
        """Test that return_all=true doesn't validate slots against user tier."""
        test_client, mock_engine = client

        mock_engine.get_all_recommendations.return_value = []

        # This would fail slot validation for free tier (max 8) without return_all
        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 15,  # Exceeds free tier limit
                "user_tier": "free",
                "return_all": True,
            },
        )

        # Should succeed because return_all=true skips slot validation
        assert response.status_code == 200

    def test_default_behavior_without_return_all(self, client):
        """Test backward compatibility - default behavior works same as before."""
        test_client, mock_engine = client

        mock_engine.get_recommendations.return_value = [
            {
                "id": "rec_554_123",
                "itemId": 554,
                "item": "Fire rune",
                "buyPrice": 4,
                "sellPrice": 6,
                "quantity": 1000,
                "capitalRequired": 4000,
                "expectedProfit": 200,
                "expectedHours": 4,
                "confidence": "high",
                "fillProbability": 0.12,
                "fillConfidence": "Good",

                "trend": "Stable",
            }
        ]

        response = test_client.post(
            "/api/v1/recommendations",
            json={
                "style": "active",
                "capital": 10000000,
                "risk": "medium",
                "slots": 4,
                # Note: return_all is omitted (defaults to False)
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Default: returns list (not AllRecommendationsResponse)
        assert isinstance(data, list)
        mock_engine.get_recommendations.assert_called_once()
