# User Feedback System

This document explains the user feedback system for collecting structured feedback on recommendations.

## Overview

The feedback system allows users to report issues with recommendations, such as:
- Prices being too high or low
- Volume issues
- Fill time problems
- Suspected manipulation

This feedback is used for model improvement and identifying systematic issues with specific items.

## Feedback Types

| Type | Description | When to Use |
|------|-------------|-------------|
| `price_too_high` | Recommended buy price was too high | Buy order didn't fill at recommended price |
| `price_too_low` | Recommended sell price was too low | Sell order didn't fill at recommended price |
| `volume_too_low` | Insufficient volume to trade | Could only partially fill order |
| `filled_quickly` | Order filled faster than expected | Positive feedback - model underestimated fill speed |
| `filled_slowly` | Order took longer than expected | Trade eventually filled but took too long |
| `did_not_fill` | Order never filled within window | Trade expired without filling |
| `spread_too_wide` | Spread made trade unprofitable | Actual spread was wider than predicted |
| `price_manipulation` | Suspected price manipulation | Item showed unusual price behavior |
| `other` | Free-text feedback | Use notes field for details |

## Submit Feedback Endpoint

```
POST /api/v1/feedback
```

### Request Body

```json
{
  "userId": "sha256_hash_of_discord_id",
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

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `userId` | string | Yes | SHA256 hash of Discord ID (64 hex chars) |
| `itemId` | integer | Yes | OSRS item ID |
| `itemName` | string | Yes | Item name |
| `feedbackType` | string | Yes | One of the feedback types above |
| `recId` | string | No | Recommendation ID to link feedback |
| `side` | string | No | `buy` or `sell` |
| `notes` | string | No | Free-text notes (max 500 chars) |
| `recommendedPrice` | integer | No | Price from recommendation |
| `actualPrice` | integer | No | Actual price user encountered |
| `submittedAt` | string | Yes | ISO 8601 timestamp |

### Response

```json
{
  "success": true,
  "message": "Feedback recorded",
  "feedbackId": 42
}
```

## Analytics Endpoint

```
GET /api/v1/feedback/analytics?period=week
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `week` | `week`, `month`, or `all` |
| `item_id` | integer | - | Filter by item ID |

### Response

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

## Discord Bot Integration

The Discord bot should call this endpoint when:

1. **`/feedback` command**: Direct feedback submission
   ```
   /feedback dragon bones price_too_high
   ```

2. **Reaction-based feedback**: Quick feedback via emoji reactions on recommendation messages

### Suggested Reactions

| Emoji | Feedback Type |
|-------|---------------|
| :thumbsup: | `filled_quickly` |
| :thumbsdown: | `did_not_fill` |
| :chart_with_upwards_trend: | `price_too_high` |
| :chart_with_downwards_trend: | `price_too_low` |

## Privacy

- User IDs must be SHA256 hashed before submission
- No Discord IDs or PII stored
- Data used only for ML training and analytics

## Database Schema

The feedback is stored in the `recommendation_feedback` table:

```sql
CREATE TABLE recommendation_feedback (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,
    rec_id VARCHAR(50),
    item_id INTEGER NOT NULL,
    item_name VARCHAR(100) NOT NULL,
    feedback_type VARCHAR(30) NOT NULL,
    side VARCHAR(4),
    notes TEXT,
    recommended_price INTEGER,
    actual_price INTEGER,
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Use Cases

### Identifying Problematic Items

Query analytics filtered by item to see if certain items consistently receive negative feedback:

```bash
curl "http://localhost:8000/api/v1/feedback/analytics?item_id=536"
```

### Model Improvement

Feedback data can be used to:
- Identify items where the model consistently over/under-predicts
- Detect market manipulation patterns
- Adjust confidence thresholds for specific item categories
- Improve fill probability models

### User Experience

Collecting feedback:
- Shows users their input matters
- Helps identify UX issues (e.g., confusing price displays)
- Enables personalized recommendations over time
