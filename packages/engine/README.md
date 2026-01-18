# GePT Recommendation Engine

Transforms raw ML predictions into optimized OSRS Grand Exchange trade recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ampere Server                            │
│                    (<SERVER_IP>)                           │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Cron Job   │────▶│  Inference   │────▶│  PostgreSQL  │   │
│  │  (5 min)     │     │   Engine     │     │  predictions │   │
│  └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                    │           │
└────────────────────────────────────────────────────┼───────────┘
                                                     │
                              SSH Tunnel (port 5432) │
                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Recommendation Engine                        │
│                        (This Service)                           │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │  Prediction  │────▶│   Filter &   │────▶│   Discord    │   │
│  │   Loader     │     │   Optimize   │     │  Bot Format  │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

The engine reads pre-computed fill probability predictions from the Ampere server's `predictions` table and applies user constraints (capital, trading style, risk tolerance, GE slots) to generate optimized trade recommendations.

## Quick Start

### Prerequisites

- Python 3.11+
- SSH access to Ampere server (for database tunnel)
- `oracle_key.pem` SSH key

### Installation

```bash
cd gept-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Database Connection

Create an SSH tunnel to the Ampere server:

```bash
ssh -i oracle_key.pem -L 5432:localhost:5432 ubuntu@<SERVER_IP>
```

Keep this running in a separate terminal.

### Running

```bash
# Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Or use the main entry point
python -m src.main
```

### Verify Connection

```bash
curl http://localhost:8000/api/v1/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/recommendations` | GET/POST | Get optimized recommendations |
| `/api/v1/recommendations/{rec_id}` | GET | Lookup recommendation by ID |
| `/api/v1/recommendations/item/{item_id}` | GET | Lookup by item ID |
| `/api/v1/predictions/{item_id}` | GET | Get all predictions for an item |
| `/api/v1/health` | GET | Health check |

### Get Recommendations

```bash
# Via POST (preferred)
curl -X POST "http://localhost:8000/api/v1/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "style": "active",
    "capital": 10000000,
    "risk": "medium",
    "slots": 4
  }'

# Via GET
curl "http://localhost:8000/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
```

### Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `style` | `passive`, `hybrid`, `active` | Trading style (affects time horizon) |
| `capital` | `>= 1000` | Available GP |
| `risk` | `low`, `medium`, `high` | Risk tolerance (affects EV threshold) |
| `slots` | `1-8` | Available GE slots |

### Style Effects

| Style | Hour Range | Description |
|-------|------------|-------------|
| `active` | 1-4 hours | Quick flips, requires monitoring |
| `hybrid` | 2-12 hours | Medium-term trades |
| `passive` | 8-48 hours | Overnight/passive trades |

### Risk Effects

| Risk | Min EV | Fill Probability |
|------|--------|------------------|
| `low` | 0.8% | 8-25% (more confident) |
| `medium` | 0.5% | 5-30% |
| `high` | 0.3% | 3-35% (accepts uncertainty) |

## Response Format

```json
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
  "trend": "Stable"
}
```

## Project Structure

```
gept-engine/
├── src/
│   ├── api.py                  # FastAPI server
│   ├── config.py               # Configuration
│   ├── prediction_loader.py    # Queries predictions table
│   └── recommendation_engine.py # Optimization logic
├── tests/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Configuration

Environment variables (see `.env.example`):

```bash
# Database (via SSH tunnel)
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/osrs_data

# Thresholds
MIN_EV_THRESHOLD=0.005
DATA_STALE_SECONDS=600

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## Database Schema

The engine reads from the `predictions` table on Ampere:

```sql
SELECT
    item_id,
    item_name,
    hour_offset,        -- 1-24 hours
    offset_pct,         -- Price offset (0.02 = 2%)
    fill_probability,   -- P(both orders fill)
    expected_value,     -- EV = prob × net_profit
    buy_price,
    sell_price,
    confidence
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions);
```

## Discord Bot Integration

The engine provides recommendations in the format expected by the Discord bot's `Recommendation` interface:

```typescript
interface Recommendation {
  id: string;           // Stable ID for rate limiting
  itemId: number;       // OSRS item ID
  item: string;         // Display name
  buyPrice: number;     // Suggested buy price
  sellPrice: number;    // Suggested sell price
  quantity: number;     // Quantity (respects buy limits)
  capitalRequired: number;
  expectedProfit: number;
  confidence: 'high' | 'medium' | 'low';
  volumeTier: string;   // High/Medium/Low
  trend: string;        // Rising/Stable/Falling
}
```

### Recommendation IDs

IDs are stable within the same hour: `rec_{itemId}_{YYYYMMDDHH}`

This ensures users aren't charged twice for viewing the same recommendation.
