# GePT Recommendation Engine

Transforms raw ML predictions into optimized OSRS Grand Exchange trade recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prediction Host (redacted)                   │
│                         (<HOST>)                                │
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

The engine reads pre-computed fill probability predictions from the `predictions` table and applies user constraints (capital, trading style, risk tolerance, GE slots) to generate optimized trade recommendations.

## Quick Start

### Prerequisites

- Python 3.11+
- SSH access to the prediction database host (for database tunnel)
- SSH key for the database host

### Installation

```bash
cd gept-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a local `.env` in `packages/engine` (loaded by `src/main.py`) with the minimum required values:

```bash
# Database (via SSH tunnel)
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/osrs_data

# API auth (required)
INTERNAL_API_KEY=replace-with-random-token
```

Generate a local API key if you need one:

```bash
openssl rand -hex 32
```

If you run `uvicorn` directly (instead of `python -m src.main`), make sure your shell loads the env file:

```bash
set -a
source .env
set +a
```

### Database Connection

Create an SSH tunnel to the prediction database host:

```bash
ssh -i <ssh_key>.pem -L 5432:localhost:5432 <user>@<host>
```

Keep this running in a separate terminal.

### Running

```bash
# From monorepo root (preferred for consistency)
npm run dev:engine

# Or directly with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Or use the main entry point
python -m src.main
```

### Running (Docker)

```bash
cd packages/engine
cp .env.docker.example .env.docker

docker compose -f docker-compose.local.yml --env-file .env.docker up --build
```

Then open `http://localhost:8000/docs` or hit `http://localhost:8000/healthz`.

### Verify Connection

```bash
curl http://localhost:8000/api/v1/health
```

## Tests & Linting

```bash
# From packages/engine
pytest
flake8 src tests
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
  -H "X-API-Key: $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "style": "active",
    "capital": 10000000,
    "risk": "medium",
    "slots": 4
  }'

# Via GET
curl -H "X-API-Key: $INTERNAL_API_KEY" \
  "http://localhost:8000/api/v1/recommendations?style=active&capital=10000000&risk=medium&slots=4"
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
  "trend": "Stable"
}
```

## Authentication

All API endpoints require the `X-API-Key` header. Example:

```bash
curl -H "X-API-Key: $INTERNAL_API_KEY" http://localhost:8000/api/v1/health
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

# API auth (required)
INTERNAL_API_KEY=replace-with-random-token

# Thresholds
MIN_EV_THRESHOLD=0.005
DATA_STALE_SECONDS=600

# API
API_HOST=0.0.0.0
API_PORT=8000
```

Generate a key if you don't have one yet:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Database Schema

The engine reads from the `predictions` table in the prediction database:

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
  trend: string;        // Rising/Stable/Falling
}
```

### Recommendation IDs

IDs are stable within the same hour: `rec_{itemId}_{YYYYMMDDHH}`

This ensures users aren't charged twice for viewing the same recommendation.
