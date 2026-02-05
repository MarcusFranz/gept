# Trade Outcome Reporting

This document explains how to report trade outcomes to the prediction engine for model improvement.

## Overview

The trade outcome endpoint allows Discord bot users to report the results of their trades (profit/loss/break-even). This feedback loop is critical for:

- **Model improvement**: Learn which recommendations succeed vs fail
- **Confidence calibration**: Adjust confidence scoring based on real outcomes
- **Crowding detection**: Identify over-crowded items where many users report losses
- **Accuracy tracking**: Monitor recommendation quality over time

## Endpoint

```
POST /api/v1/recommendations/{recId}/outcome
```

## Authentication

All requests require an `X-API-Key` header. The endpoint also validates that user IDs are properly hashed for privacy.

## Request Format

### Headers

```
Content-Type: application/json
X-API-Key: <your-internal-api-key>
```

### URL Parameters

- `recId` (string, required): The recommendation ID that led to this trade

### Request Body

```json
{
  "userId": "a1b2c3d4e5f6...",  // SHA256 hash (64 hex chars)
  "itemId": 5295,
  "itemName": "Ranarr seed",
  "recId": "rec_5295_2026010923",
  "buyPrice": 43250,
  "sellPrice": 44890,
  "quantity": 1100,
  "actualProfit": 1602000,
  "reportedAt": "2026-01-09T12:00:00Z"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `userId` | string | Yes | SHA256 hash of Discord user ID (64 hex characters) |
| `itemId` | integer | Yes | OSRS item ID |
| `itemName` | string | Yes | Item name (for human readability) |
| `recId` | string | Yes | Recommendation ID (must match URL parameter) |
| `buyPrice` | integer | Yes | Buy price in gp (≥1) |
| `sellPrice` | integer | Yes | Sell price in gp (≥1) |
| `quantity` | integer | Yes | Quantity traded (≥1) |
| `actualProfit` | integer | Yes | Actual profit/loss after GE tax (can be negative) |
| `reportedAt` | string | Yes | ISO 8601 timestamp when user reported outcome |

## Response Format

### Success Response (200 OK)

```json
{
  "success": true,
  "message": "Outcome recorded"
}
```

### Error Responses

#### 400 Bad Request - Invalid user ID
```json
{
  "detail": "userId must be SHA256 hash (64 hex characters)"
}
```

#### 400 Bad Request - rec_id mismatch
```json
{
  "detail": "rec_id in URL does not match recId in request body"
}
```

#### 400 Bad Request - Invalid timestamp
```json
{
  "detail": "Invalid reportedAt timestamp: ..."
}
```

#### 422 Unprocessable Entity - Validation error
```json
{
  "detail": [
    {
      "loc": ["body", "buyPrice"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

#### 401 Unauthorized - Missing/invalid API key
```json
{
  "detail": "Missing API key. Include X-API-Key header."
}
```

#### 503 Service Unavailable - Outcome database not available
```json
{
  "detail": "Outcome database not available"
}
```

> **Note**: Trade outcomes are stored in a separate database (`gept_bot`) from predictions (`osrs_data`). If the outcome database is not configured or unavailable, this endpoint returns 503.

## Privacy and Security

### User ID Hashing

**CRITICAL**: Never send Discord user IDs directly. Always hash them first:

```javascript
// Node.js example
const crypto = require('crypto');

function hashUserId(discordId) {
  return crypto
    .createHash('sha256')
    .update(discordId.toString())
    .digest('hex');
}

const hashedId = hashUserId('123456789012345678');
```

```python
# Python example
import hashlib

def hash_user_id(discord_id: str) -> str:
    return hashlib.sha256(discord_id.encode()).hexdigest()

hashed_id = hash_user_id('123456789012345678')
```

### Data Usage

- Trade outcomes are stored only for ML training purposes
- No personally identifiable information (PII) is stored
- User IDs are stored as SHA256 hashes (one-way, non-reversible)
- Data retention: Indefinite for model training
- GDPR compliance: Users can request data export/deletion (to be implemented)

## Example Usage

### cURL

```bash
curl -X POST "https://api.example.com/api/v1/recommendations/rec_5295_2026010923/outcome" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "userId": "a1b2c3d4e5f6...",
    "itemId": 5295,
    "itemName": "Ranarr seed",
    "recId": "rec_5295_2026010923",
    "buyPrice": 43250,
    "sellPrice": 44890,
    "quantity": 1100,
    "actualProfit": 1602000,
    "reportedAt": "2026-01-09T12:00:00Z"
  }'
```

### JavaScript (fetch)

```javascript
async function reportTradeOutcome(outcome) {
  const response = await fetch(
    `https://api.example.com/api/v1/recommendations/${outcome.recId}/outcome`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify({
        userId: hashUserId(discordUserId),
        itemId: outcome.itemId,
        itemName: outcome.itemName,
        recId: outcome.recId,
        buyPrice: outcome.buyPrice,
        sellPrice: outcome.sellPrice,
        quantity: outcome.quantity,
        actualProfit: outcome.actualProfit,
        reportedAt: new Date().toISOString(),
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to report outcome: ${response.statusText}`);
  }

  return await response.json();
}
```

## Database Schema

Trade outcomes are stored in a **separate database** (`gept_bot`) from predictions:

**Connection String:** `OUTCOME_DB_CONNECTION_STRING` environment variable

```sql
CREATE TABLE trade_outcomes (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,      -- SHA256 hash of Discord user ID
    rec_id VARCHAR(50) NOT NULL,             -- Recommendation ID from the engine
    item_id INTEGER NOT NULL,                -- OSRS item ID
    item_name VARCHAR(100) NOT NULL,         -- Item name at time of trade
    buy_price INTEGER NOT NULL,              -- Actual buy price (gp)
    sell_price INTEGER NOT NULL,             -- Actual sell price (gp)
    quantity INTEGER NOT NULL,               -- Quantity traded
    actual_profit BIGINT NOT NULL,           -- Actual profit/loss (gp)
    reported_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_trade_outcomes_item_id ON trade_outcomes (item_id);
CREATE INDEX idx_trade_outcomes_user_id_hash ON trade_outcomes (user_id_hash);
CREATE INDEX idx_trade_outcomes_rec_id ON trade_outcomes (rec_id);
CREATE INDEX idx_trade_outcomes_reported_at ON trade_outcomes (reported_at);
CREATE INDEX idx_trade_outcomes_created_at ON trade_outcomes (created_at);
```

### Environment Configuration

The API requires two database connections:
- `DB_CONNECTION_STRING` - Predictions database (`osrs_data`, read-only)
- `OUTCOME_DB_CONNECTION_STRING` - Outcome database (`gept_bot`, write-only)

## Integration with Discord Bot

The Discord bot should call this endpoint in two places:

1. **`/report` command**: When users manually report trade outcomes
2. **Active trades "Filled" modal**: When users mark trades as filled

See the bot's implementation files:
- `src/router/handlers/report.ts` - Report command handler
- `src/router/handlers/activeTrades.ts:284` - Filled modal handler

## Future Enhancements

- [ ] Aggregate outcome statistics API
- [ ] Per-user accuracy tracking (with hashed IDs)
- [ ] Data export/deletion endpoints (GDPR)
- [ ] Webhook notifications for model retraining
- [ ] Confidence score adjustments based on outcomes
