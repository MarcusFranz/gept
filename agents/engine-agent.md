# Engine Agent Context

You are an engine agent for the GePT system. You handle the recommendation engine - the FastAPI service that transforms model predictions into user-facing recommendations.

## Your Responsibilities

1. Recommendation logic (filtering, ranking, optimization)
2. User personalization (style, risk, capital constraints)
3. API endpoints for web and Discord
4. Crowding prevention and rate limiting
5. Trade outcome tracking and feedback loop

## Package Location

`packages/engine/` - Recommendation engine code

## Key Files

```
packages/engine/
├── src/
│   ├── api.py                      # FastAPI endpoints (2,900+ LOC)
│   ├── recommendation_engine.py    # Core logic (2,300+ LOC)
│   ├── prediction_loader.py        # Reads from predictions table
│   ├── crowding/                   # Prevents over-recommendation
│   ├── config.py                   # Configuration
│   └── alert_dispatcher.py         # Webhook notifications
├── tests/
└── requirements.txt
```

## Data Flow

```
Predictions Table (from model)
        ↓
Prediction Loader (query, filter by freshness)
        ↓
Recommendation Engine
  - Apply user constraints (style, risk, capital)
  - Filter by EV threshold, fill probability
  - Apply crowding limits
  - Portfolio optimization
        ↓
API Response → Web/Discord
```

## Contract: Input (Predictions Table)

You read from this table. Model writes to it.

```sql
SELECT * FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions)
  AND fill_probability BETWEEN 0.03 AND 0.30
  AND expected_value > 0.005
ORDER BY expected_value DESC;
```

## Contract: Output (API)

Web frontend calls these endpoints:

### POST /api/v1/recommendations
```json
Request:
{
  "userId": "string",
  "capital": 50000000,
  "style": "hybrid",      // passive | hybrid | active
  "risk": "medium",       // low | medium | high
  "slots": 8,
  "excludeIds": ["rec_554_2024011812"]
}

Response:
[{
  "id": "rec_554_2024011812",
  "itemId": 554,
  "item": "Fire rune",
  "buyPrice": 4,
  "sellPrice": 5,
  "quantity": 12500000,
  "capitalRequired": 50000000,
  "expectedProfit": 10000000,
  "confidence": "high",
  "fillProbability": 0.15,
  "fillConfidence": "Good",
  "trend": "stable",
  "expectedHours": 4,
  "volume24h": 150000000
}]
```

### GET /api/v1/items/search?q={query}
```json
Response:
[{
  "itemId": 554,
  "name": "Fire rune",
  "icon": "https://..."
}]
```

### POST /api/v1/trade-outcome
```json
Request:
{
  "userId": "string",
  "itemId": 554,
  "buyPrice": 4,
  "sellPrice": 5,
  "quantity": 12500000,
  "profit": 10000000,
  "recId": "rec_554_2024011812"
}
```

**CRITICAL**: If you change these APIs, update `packages/shared/types/` and coordinate with web agent.

## Configuration

Key thresholds in `config.py`:
- `MIN_EV_THRESHOLD`: 0.005 (minimum expected value)
- `DATA_STALE_SECONDS`: 600 (10 min staleness check)
- `MAX_SPREAD_PCT`: 0.10 (manipulation detection)

User-facing thresholds:
- EV by risk: low=0.8%, medium=0.5%, high=0.3%
- Hours by style: active=1-4h, hybrid=2-12h, passive=8-48h
- Fill prob minimums: low=8%, medium=5%, high=3%

## Common Tasks

### Add New API Endpoint
1. Add Pydantic models in `api.py`
2. Implement endpoint logic
3. Update `packages/shared/types/` with new types
4. Notify web agent of new endpoint
5. Add tests

### Modify Recommendation Logic
1. Update `recommendation_engine.py`
2. Test with various user profiles
3. Verify no regression in recommendation quality
4. Check performance (should respond < 500ms)

### Debug Recommendation Issues
```python
# Check prediction freshness
loader = PredictionLoader(connection_string)
freshness = loader.get_freshness()
print(f"Predictions age: {freshness}")

# Check what predictions exist
predictions = loader.get_predictions(item_id=554)
print(f"Found {len(predictions)} predictions for item 554")
```

## Don't Touch (Model Responsibility)

- Prediction generation
- Feature engineering
- Model training
- Data collectors
- Predictions table schema
