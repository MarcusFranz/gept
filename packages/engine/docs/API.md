# GePT Recommendation Engine API Reference

**Version:** 2.0.0
**Base URL:** `http://localhost:8000`

## Overview

OSRS Grand Exchange flipping recommendation API. Provides optimized trade recommendations based on ML predictions, user capital, trading style, and risk tolerance.

## Authentication

All endpoints require an `X-API-Key` header. The engine reads the key from `INTERNAL_API_KEY` in its environment.

Example header:

```
X-API-Key: <your-internal-api-key>
```

---

## Endpoints

### Health Check

```
GET /api/v1/health
```

Check system health status including database connectivity and prediction freshness.

**Response:**
```json
{
  "status": "ok",
  "checks": [
    {
      "status": "ok",
      "component": "prediction_loader",
      "message": "Predictions are 248s old",
      "prediction_age_seconds": 247.89,
      "connected": true
    },
    {
      "status": "ok",
      "component": "model_registry",
      "message": "45 active, 3 deprecated models",
      "stats": {"active": 45, "deprecated": 3, "sunset": 0, "archived": 0, "total": 48}
    }
  ],
  "timestamp": "2026-01-10T06:04:56.939875+00:00",
  "recommendation_store_size": 0,
  "crowding_stats": {
    "tracked_items": 0,
    "total_deliveries": 0,
    "window_hours": 4.0
  },
  "model_registry_stats": {
    "active": 45,
    "deprecated": 3,
    "sunset": 0,
    "archived": 0,
    "total": 48
  }
}
```

---

### Get Recommendations

#### POST (Preferred)

```
POST /api/v1/recommendations
```

Main endpoint for Discord bot to fetch flip recommendations with full context.

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "style": "active",
    "capital": 10000000,
    "risk": "medium",
    "slots": 4
  }'
```

**Request Body:**
```json
{
  "style": "active",
  "capital": 10000000,
  "risk": "medium",
  "slots": 4,
  "activeTrades": [
    {"itemId": 536, "quantity": 1000, "buyPrice": 20000}
  ],
  "userId": "sha256_hashed_discord_id"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `style` | string | No | `passive`, `hybrid`, `active` (default: `hybrid`) |
| `capital` | integer | **Yes** | Available GP (minimum: 1000) |
| `risk` | string | No | `low`, `medium`, `high` (default: `medium`) |
| `slots` | integer | No | Available GE slots, 1-8 (default: 4) |
| `activeTrades` | array | No | Currently tracked trades (auto-excluded) |
| `userId` | string | No | Hashed user ID for crowding tracking |

#### GET (Alternative)

```
GET /api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4
```

Alternative endpoint using query parameters.

**Example:**
```bash
curl -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `style` | string | No | `passive`, `hybrid`, `active` (default: `hybrid`) |
| `capital` | integer | **Yes** | Available GP (minimum: 1000) |
| `risk` | string | No | `low`, `medium`, `high` (default: `medium`) |
| `slots` | integer | No | Available GE slots, 1-8 (default: 4) |
| `exclude` | string | No | Comma-separated recommendation IDs to exclude |
| `exclude_item_ids` | string | No | Comma-separated item IDs to exclude (e.g., `536,5295,4151`) |
| `user_id` | string | No | Hashed user ID for crowding tracking |

**Response:**
```json
[
  {
    "id": "rec_565_2026010812",
    "itemId": 565,
    "item": "Blood rune",
    "buyPrice": 245,
    "sellPrice": 260,
    "quantity": 10000,
    "capitalRequired": 2450000,
    "expectedProfit": 22500,
    "confidence": "high",
    "volumeTier": "High",
    "trend": "Stable",
    "expectedHours": 4,
    "volume24h": 150000,
    "priceHistory": null,
    "modelId": 123,
    "modelStatus": "ACTIVE"
  }
]
```

---

### Get Recommendation by Item ID

```
GET /api/v1/recommendations/item/{item_id}
```

Get recommendation for a specific OSRS item by its ID.

**Parameters:**

| Parameter | Location | Type | Required | Description |
|-----------|----------|------|----------|-------------|
| `item_id` | path | integer | **Yes** | OSRS item ID (e.g., 4151 for Abyssal whip) |
| `capital` | query | integer | No | Available GP for sizing |
| `risk` | query | string | No | Risk tolerance |
| `style` | query | string | No | Trading style |
| `slots` | query | integer | No | Available GE slots |
| `include_price_history` | query | boolean | No | Include 24h price data (default: false) |

---

### Get Recommendation by Item Name

```
GET /api/v1/recommendations/item?name=dragon+bones&capital=10000000
```

Search for item by name with fuzzy matching.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | **Yes** | Item name to search for |
| `capital` | integer | **Yes** | Available GP |
| `risk` | string | No | Risk tolerance (default: `medium`) |
| `style` | string | No | Trading style (default: `hybrid`) |
| `slots` | integer | No | Available GE slots (default: 4) |

---

### Get Recommendation by ID

```
GET /api/v1/recommendations/{rec_id}
```

Retrieve a specific stored recommendation by its ID. Used when user clicks "Mark Ordered" in Discord.

---

### Search Items

```
GET /api/v1/items/search?q=dragon&limit=10
```

Search for items by name with fuzzy matching for Discord autocomplete.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | **Yes** | Search query (minimum 1 character) |
| `limit` | integer | No | Max results, 1-25 (default: 10) |

**Response:**
```json
[
  {"itemId": 536, "name": "Dragon bones", "category": "Prayer"},
  {"itemId": 11943, "name": "Dragon claws", "category": "Weapons"}
]
```

---

### Get Item Predictions

```
GET /api/v1/predictions/{item_id}
```

Get full prediction details for a specific item including all hour/offset combinations.

**Response:**
```json
{
  "item_id": 565,
  "item_name": "Blood rune",
  "best_config": {"hour_offset": 4, "offset_pct": 0.015},
  "fill_probability": 0.72,
  "expected_value": 0.008,
  "buy_price": 245,
  "sell_price": 260,
  "confidence": "high",
  "all_predictions": [...],
  "model_id": 123,
  "model_status": "ACTIVE",
  "mean_auc": 0.78
}
```

---

### Report Trade Outcome

```
POST /api/v1/recommendations/{rec_id}/outcome
```

Record a trade outcome for ML feedback loop.

**Request Body:**
```json
{
  "userId": "sha256_hashed_discord_id",
  "itemId": 5295,
  "itemName": "Ranarr seed",
  "recId": "rec_5295_2026010812",
  "buyPrice": 43250,
  "sellPrice": 44890,
  "quantity": 1100,
  "actualProfit": 1602000,
  "reportedAt": "2026-01-09T12:00:00Z"
}
```

**Privacy:** User IDs must be SHA256 hashed. No Discord IDs or PII stored.

---

### Submit Feedback

```
POST /api/v1/feedback
```

Submit structured feedback on a recommendation for model improvement.

**Request Body:**
```json
{
  "userId": "sha256_hashed_discord_id",
  "itemId": 536,
  "itemName": "Dragon bones",
  "feedbackType": "price_too_high",
  "recId": "rec_536_2026011510",
  "side": "buy",
  "notes": "Price jumped 10% right after recommendation",
  "recommendedPrice": 2000,
  "actualPrice": 2200,
  "submittedAt": "2026-01-15T10:00:00Z"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `userId` | string | **Yes** | SHA256 hash of Discord ID (64 hex chars) |
| `itemId` | integer | **Yes** | OSRS item ID |
| `itemName` | string | **Yes** | Item name |
| `feedbackType` | string | **Yes** | One of: `price_too_high`, `price_too_low`, `volume_too_low`, `filled_quickly`, `filled_slowly`, `did_not_fill`, `spread_too_wide`, `price_manipulation`, `other` |
| `recId` | string | No | Recommendation ID to link feedback |
| `side` | string | No | `buy` or `sell` |
| `notes` | string | No | Free-text notes (max 500 chars) |
| `recommendedPrice` | integer | No | Price from recommendation |
| `actualPrice` | integer | No | Actual price encountered |
| `submittedAt` | string | **Yes** | ISO 8601 timestamp |

**Response:**
```json
{
  "success": true,
  "message": "Feedback recorded",
  "feedbackId": 42
}
```

See [Feedback Documentation](./feedback.md) for detailed feedback type definitions.

---

### Feedback Analytics

```
GET /api/v1/feedback/analytics?period=week
```

Get aggregated feedback statistics for model improvement analysis.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period` | string | No | `week` (7 days), `month` (30 days), or `all` (default: `week`) |
| `item_id` | integer | No | Filter by specific item ID |

**Response:**
```json
{
  "period": "week",
  "startDate": "2026-01-08",
  "endDate": "2026-01-15",
  "totalFeedback": 150,
  "byType": [
    {"feedbackType": "price_too_high", "count": 45, "percentage": 30.0},
    {"feedbackType": "did_not_fill", "count": 32, "percentage": 21.3}
  ],
  "topItems": [
    {"itemId": 536, "itemName": "Dragon bones", "count": 23},
    {"itemId": 4151, "itemName": "Abyssal whip", "count": 18}
  ]
}
```

---

## Parameter Reference

### Trading Style

| Style | Hour Range | Description |
|-------|------------|-------------|
| `active` | 1-4 hours | Quick flips, requires monitoring |
| `hybrid` | 2-12 hours | Medium-term trades |
| `passive` | 8-48 hours | Overnight/passive trades |

### Risk Level

| Risk | Min EV | Fill Probability | Description |
|------|--------|------------------|-------------|
| `low` | 0.8% | 8-25% | Higher confidence, stable items |
| `medium` | 0.5% | 5-30% | Balanced approach |
| `high` | 0.3% | 3-35% | Accepts more uncertainty |

### Confidence Tiers

| Confidence | Criteria |
|------------|----------|
| `high` | Model AUC > 0.75, data < 2 min old, tier 1 items |
| `medium` | Model AUC > 0.60, data < 5 min old |
| `low` | Model AUC > 0.52, data < 10 min old |

### Volume Tiers

| Tier | Spread | Crowding Limit |
|------|--------|----------------|
| `Very High` | < 1% | Unlimited |
| `High` | < 2% | 50 concurrent users |
| `Medium` | < 5% | 20 concurrent users |
| `Low` | >= 5% | 10 concurrent users |

---

## Error Responses

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["query", "capital"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 401 Unauthorized

Returned when the `X-API-Key` header is missing or invalid.

```json
{
  "detail": "Missing API key. Include X-API-Key header."
}
```

### 429 Too Many Requests

Returned when the request rate exceeds the configured limit.

### 503 Service Unavailable

Returned when the recommendation engine is not initialized or database is unavailable.

---

## OpenAPI Specification

The full OpenAPI 3.1 specification is available at:
- **Interactive docs:** `/docs` (Swagger UI)
- **Alternative docs:** `/redoc` (ReDoc)
- **JSON spec:** `/openapi.json`
- **Static spec:** [`openapi.json`](./openapi.json)
