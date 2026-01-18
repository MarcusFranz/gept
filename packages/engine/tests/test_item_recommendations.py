"""Tests for item-specific recommendation endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


class TestItemRecommendations:
    """Test cases for item-specific recommendation endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked recommendation engine."""
        with patch("src.api.RecommendationEngine"):
            mock_engine = MagicMock()

            # Mock health check
            mock_engine.health_check.return_value = {
                "status": "ok",
                "checks": [{"status": "ok", "component": "test"}],
                "timestamp": "2026-01-09T00:00:00Z",
                "recommendation_store_size": 0,
            }

            from src.api import app
            import src.api as api_module

            # Set the engine before creating client
            api_module.engine = mock_engine

            yield TestClient(app), mock_engine

    def test_get_recommendation_by_item_id_with_context(self, client):
        """Test getting recommendation by item ID with user context."""
        test_client, mock_engine = client

        # Mock engine response
        mock_engine.get_recommendation_by_item_id.return_value = {
            "id": "rec_536_2026010923",
            "itemId": 536,
            "item": "Dragon bones",
            "buyPrice": 2015,
            "sellPrice": 2089,
            "quantity": 10000,
            "capitalRequired": 20150000,
            "expectedProfit": 740000,
            "confidence": "medium",
            "fillProbability": 0.12,
            "fillConfidence": "Good",
            "volumeTier": "Very High",
            "trend": "Stable",
            "expectedHours": 4,
            "reason": "Stable trend, very high volume, 4h window",
        }

        response = test_client.get(
            "/api/v1/recommendations/item/536",
            params={
                "capital": 50000000,
                "risk": "medium",
                "style": "active",
                "slots": 4,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 536
        assert data["item"] == "Dragon bones"
        assert data["buyPrice"] == 2015
        assert "reason" in data

        # Verify engine was called with correct parameters
        mock_engine.get_recommendation_by_item_id.assert_called_once_with(
            item_id=536,
            capital=50000000,
            risk="medium",
            style="active",
            slots=4,
            include_price_history=False,
        )

    def test_get_recommendation_by_item_id_without_context(self, client):
        """Test getting cached recommendation by item ID without user context."""
        test_client, mock_engine = client

        # Mock cached recommendation
        mock_engine.get_recommendation_by_item_id.return_value = {
            "id": "rec_536_2026010923",
            "itemId": 536,
            "item": "Dragon bones",
            "buyPrice": 2015,
            "sellPrice": 2089,
            "quantity": 5000,
            "capitalRequired": 10075000,
            "expectedProfit": 370000,
            "confidence": "high",
            "fillProbability": 0.18,
            "fillConfidence": "Strong",
            "volumeTier": "Very High",
            "trend": "Stable",
            "expectedHours": 2,
        }

        response = test_client.get("/api/v1/recommendations/item/536")

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 536

        # Verify engine was called without context parameters
        mock_engine.get_recommendation_by_item_id.assert_called_once_with(
            item_id=536,
            capital=None,
            risk=None,
            style=None,
            slots=None,
            include_price_history=False,
        )

    def test_get_recommendation_item_not_recommended(self, client):
        """Test when item is not recommended."""
        test_client, mock_engine = client

        # Mock "not recommended" response
        mock_engine.get_recommendation_by_item_id.return_value = {
            "itemId": 536,
            "item": "Dragon bones",
            "isRecommended": False,
            "reason": "Spread too low (1.2%) - below minimum threshold",
            "currentBuyPrice": 2015,
            "currentSellPrice": 2039,
            "spread": 0.012,
        }

        response = test_client.get(
            "/api/v1/recommendations/item/536",
            params={
                "capital": 50000000,
                "risk": "low",
                "style": "passive",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isRecommended"] is False
        assert "reason" in data
        assert "Spread too low" in data["reason"]

    def test_get_recommendation_by_item_id_not_found(self, client):
        """Test when item recommendation not found."""
        test_client, mock_engine = client

        mock_engine.get_recommendation_by_item_id.return_value = None

        response = test_client.get("/api/v1/recommendations/item/999999")

        assert response.status_code == 404

    def test_get_recommendation_by_item_name(self, client):
        """Test getting recommendation by item name."""
        test_client, mock_engine = client

        # Mock successful lookup by name
        mock_engine.get_recommendation_by_item_name.return_value = {
            "id": "rec_536_2026010923",
            "itemId": 536,
            "item": "Dragon bones",
            "buyPrice": 2015,
            "sellPrice": 2089,
            "quantity": 10000,
            "capitalRequired": 20150000,
            "expectedProfit": 740000,
            "confidence": "medium",
            "fillProbability": 0.12,
            "fillConfidence": "Good",
            "volumeTier": "Very High",
            "trend": "Stable",
            "expectedHours": 4,
            "reason": "Stable trend, very high volume, 4h window",
        }

        response = test_client.get(
            "/api/v1/recommendations/item",
            params={
                "name": "dragon bones",
                "capital": 50000000,
                "risk": "medium",
                "style": "active",
                "slots": 4,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["itemId"] == 536
        assert data["item"] == "Dragon bones"

        # Verify engine was called correctly
        mock_engine.get_recommendation_by_item_name.assert_called_once_with(
            item_name="dragon bones",
            capital=50000000,
            risk="medium",
            style="active",
            slots=4,
        )

    def test_get_recommendation_by_item_name_fuzzy_match(self, client):
        """Test fuzzy matching for item names."""
        test_client, mock_engine = client

        # Mock fuzzy match result
        mock_engine.get_recommendation_by_item_name.return_value = {
            "id": "rec_536_2026010923",
            "itemId": 536,
            "item": "Dragon bones",
            "buyPrice": 2015,
            "sellPrice": 2089,
            "quantity": 10000,
            "capitalRequired": 20150000,
            "expectedProfit": 740000,
            "confidence": "medium",
            "fillProbability": 0.12,
            "fillConfidence": "Good",
            "volumeTier": "Very High",
            "trend": "Stable",
            "expectedHours": 4,
        }

        # Test with partial name
        response = test_client.get(
            "/api/v1/recommendations/item",
            params={
                "name": "drag bon",
                "capital": 10000000,
                "risk": "medium",
                "style": "hybrid",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["item"] == "Dragon bones"

    def test_get_recommendation_by_item_name_not_found(self, client):
        """Test when item name is not found."""
        test_client, mock_engine = client

        # Mock "not found" response
        mock_engine.get_recommendation_by_item_name.return_value = {
            "error": "Item not found",
            "query": "nonexistent item",
            "suggestions": ["Dragon bones", "Dragon scimitar"],
        }

        response = test_client.get(
            "/api/v1/recommendations/item",
            params={
                "name": "nonexistent item",
                "capital": 10000000,
            },
        )

        assert response.status_code == 404

    def test_get_recommendation_insufficient_capital(self, client):
        """Test when user has insufficient capital."""
        test_client, mock_engine = client

        mock_engine.get_recommendation_by_item_id.return_value = {
            "itemId": 13652,
            "item": "Dragonfire shield",
            "isRecommended": False,
            "reason": "Insufficient capital (need at least 3,500,000 gp)",
            "currentBuyPrice": 3500000,
            "currentSellPrice": 3585000,
        }

        response = test_client.get(
            "/api/v1/recommendations/item/13652",
            params={
                "capital": 1000000,  # Only 1M
                "risk": "medium",
                "style": "active",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isRecommended"] is False
        assert "Insufficient capital" in data["reason"]

    def test_recommendation_respects_user_risk_level(self, client):
        """Test that recommendations respect user's risk tolerance."""
        test_client, mock_engine = client

        # Low risk should reject high-uncertainty items
        mock_engine.get_recommendation_by_item_id.return_value = {
            "itemId": 999,
            "item": "Risky item",
            "isRecommended": False,
            "reason": "Fill probability (2.5%) too low for low risk",
            "currentBuyPrice": 1000,
            "currentSellPrice": 1100,
            "spread": 0.10,
        }

        response = test_client.get(
            "/api/v1/recommendations/item/999",
            params={
                "capital": 10000000,
                "risk": "low",
                "style": "hybrid",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isRecommended"] is False
        assert "too low for low risk" in data["reason"]

    def test_recommendation_respects_trading_style(self, client):
        """Test that recommendations respect user's trading style."""
        test_client, mock_engine = client

        # Passive style should reject quick flips
        mock_engine.get_recommendation_by_item_id.return_value = {
            "itemId": 888,
            "item": "Quick flip item",
            "isRecommended": False,
            "reason": "Time window (2h) doesn't match passive style (8-48h)",
            "currentBuyPrice": 5000,
            "currentSellPrice": 5200,
            "spread": 0.04,
        }

        response = test_client.get(
            "/api/v1/recommendations/item/888",
            params={
                "capital": 10000000,
                "risk": "medium",
                "style": "passive",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["isRecommended"] is False
        assert "doesn't match passive style" in data["reason"]

    def test_recommendation_default_parameters(self, client):
        """Test that default parameters are applied correctly."""
        test_client, mock_engine = client

        mock_engine.get_recommendation_by_item_name.return_value = {
            "id": "rec_536_2026010923",
            "itemId": 536,
            "item": "Dragon bones",
            "buyPrice": 2015,
            "sellPrice": 2089,
            "quantity": 5000,
            "capitalRequired": 10075000,
            "expectedProfit": 370000,
            "confidence": "medium",
            "fillProbability": 0.12,
            "fillConfidence": "Good",
            "volumeTier": "Very High",
            "trend": "Stable",
            "expectedHours": 4,
        }

        # Only provide required parameters
        response = test_client.get(
            "/api/v1/recommendations/item",
            params={
                "name": "dragon bones",
                "capital": 10000000,
            },
        )

        assert response.status_code == 200

        # Verify defaults were used
        mock_engine.get_recommendation_by_item_name.assert_called_once_with(
            item_name="dragon bones",
            capital=10000000,
            risk="medium",  # default
            style="hybrid",  # default
            slots=4,  # default
        )
