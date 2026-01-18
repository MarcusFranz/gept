# Server Setup Guide - Trade Outcome Reporting Endpoint

This document describes the server-side requirements for deploying the trade outcome reporting feature (Issue #17).

## Overview

The trade outcome reporting endpoint allows the Discord bot to submit user-reported trade outcomes (profit/loss) back to the prediction engine. This data creates a feedback loop for ML model improvement.

**New Endpoint:** `POST /api/v1/recommendations/{recId}/outcome`

---

## Prerequisites

- PostgreSQL 12+ database (existing `gept` database)
- Python 3.11+ runtime
- Docker and docker-compose (if using containerized deployment)
- Network access from API server to PostgreSQL

---

## Step 1: Database Migration

### 1.1 Create the `trade_outcomes` Table

Connect to your PostgreSQL database and run:

```sql
-- Create trade_outcomes table for ML feedback loop
CREATE TABLE IF NOT EXISTS trade_outcomes (
    id SERIAL PRIMARY KEY,
    user_id_hash VARCHAR(64) NOT NULL,      -- SHA256 hash of Discord user ID
    rec_id VARCHAR(50) NOT NULL,             -- Recommendation ID from the engine
    item_id INTEGER NOT NULL,                -- OSRS item ID
    item_name VARCHAR(100) NOT NULL,         -- Item name at time of trade
    buy_price INTEGER NOT NULL,              -- Actual buy price (gp)
    sell_price INTEGER NOT NULL,             -- Actual sell price (gp)
    quantity INTEGER NOT NULL,               -- Quantity traded
    actual_profit BIGINT NOT NULL,           -- Actual profit/loss (gp)
    reported_at TIMESTAMP WITH TIME ZONE NOT NULL,  -- When user reported outcome
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for efficient querying during model training
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_item_id ON trade_outcomes(item_id);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_user_id_hash ON trade_outcomes(user_id_hash);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_rec_id ON trade_outcomes(rec_id);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_reported_at ON trade_outcomes(reported_at);
CREATE INDEX IF NOT EXISTS idx_trade_outcomes_created_at ON trade_outcomes(created_at);

-- Comment for documentation
COMMENT ON TABLE trade_outcomes IS 'Stores user-reported trade outcomes for ML model feedback loop';
```

### 1.2 Apply via Command Line

```bash
psql postgresql://[USER]:[PASS]@[HOST]:5432/gept -f migrations/001_create_trade_outcomes_table.sql
```

Or connect and paste:

```bash
psql postgresql://[USER]:[PASS]@[HOST]:5432/gept
```

### 1.3 Verify Migration

```sql
-- Check table exists
\dt trade_outcomes

-- Check indexes
\di | grep trade_outcomes

-- Should show:
--  trade_outcomes
--  idx_trade_outcomes_item_id
--  idx_trade_outcomes_user_id_hash
--  idx_trade_outcomes_rec_id
--  idx_trade_outcomes_reported_at
--  idx_trade_outcomes_created_at
```

---

## Step 2: Database Permissions

Ensure the API's database user has the necessary permissions:

```sql
-- Grant permissions to the API user
GRANT INSERT, SELECT ON trade_outcomes TO gept_api;
GRANT USAGE, SELECT ON SEQUENCE trade_outcomes_id_seq TO gept_api;

-- Verify permissions
\dp trade_outcomes
```

Replace `gept_api` with your actual API database username.

---

## Step 3: Environment Variables

The endpoint can use either the existing `DB_CONNECTION_STRING` or a separate outcome database connection.

### Current Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_CONNECTION_STRING` | PostgreSQL connection string | `postgresql://user:pass@host:5432/gept` |
| `API_HOST` | API bind address | `0.0.0.0` |
| `API_PORT` | API port | `8000` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_POOL_SIZE` | Database connection pool size | `5` |
| `OUTCOME_DB_CONNECTION_STRING` | Separate database for trade outcomes (if not set, uses `DB_CONNECTION_STRING`) | (empty) |
| `OUTCOME_DB_POOL_SIZE` | Outcome database connection pool size | `5` |
| `MIN_EV_THRESHOLD` | Minimum expected value threshold | `0.005` |
| `DATA_STALE_SECONDS` | Max age for price data | `600` |
| `DISCORD_WEBHOOK_URL` | Discord webhook for alerts | (empty) |

### Docker Networking Note

When running in Docker, `localhost` in `DB_CONNECTION_STRING` refers to the container itself, not the host machine. Use `host.docker.internal` instead:

```
# Wrong (inside Docker):
DB_CONNECTION_STRING=postgresql://user:pass@localhost:5432/gept

# Correct (inside Docker):
DB_CONNECTION_STRING=postgresql://user:pass@host.docker.internal:5432/gept
```

The CI/CD deploy workflow automatically adds `--add-host=host.docker.internal:host-gateway` to enable this hostname resolution.

---

## Step 4: Deployment

### Option A: Docker Deployment (Recommended)

```bash
cd gept-engine

# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d

# Check logs
docker-compose logs -f prediction-engine
```

### Option B: Direct Deployment

```bash
cd gept-engine

# Pull latest code
git pull origin main

# Install/update dependencies
pip install -r requirements.txt

# Restart API (using your process manager)
# Example with systemd:
sudo systemctl restart gept-api

# Or with uvicorn directly:
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

---

## Step 5: Verification

### 5.1 Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "ok",
  "checks": [...],
  "timestamp": "2026-01-10T00:00:00Z",
  ...
}
```

### 5.2 Test the New Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/recommendations/rec_test123/outcome \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
    "itemId": 5295,
    "itemName": "Ranarr seed",
    "recId": "rec_test123",
    "buyPrice": 43250,
    "sellPrice": 44890,
    "quantity": 1100,
    "actualProfit": 1602000,
    "reportedAt": "2026-01-09T12:00:00Z"
  }'
```

Expected response:
```json
{
  "success": true,
  "message": "Outcome recorded"
}
```

### 5.3 Verify Database Record

```sql
SELECT * FROM trade_outcomes ORDER BY created_at DESC LIMIT 5;
```

---

## Step 6: Rollback Plan

If issues arise after deployment:

### Rollback API

```bash
# Docker
docker-compose down
git checkout <previous-commit>
docker-compose up --build -d

# Direct
git checkout <previous-commit>
pip install -r requirements.txt
sudo systemctl restart gept-api
```

### Rollback Database (if needed)

```sql
-- Only if you need to completely remove the feature
DROP TABLE IF EXISTS trade_outcomes;
```

Note: Dropping the table will delete all collected outcome data. Only do this if absolutely necessary.

---

## Monitoring

### Check for Errors in Logs

```bash
# Docker
docker-compose logs prediction-engine | grep -i error

# Systemd
journalctl -u gept-api | grep -i error
```

### Database Storage

Monitor table growth:

```sql
SELECT
    pg_size_pretty(pg_total_relation_size('trade_outcomes')) as total_size,
    (SELECT COUNT(*) FROM trade_outcomes) as row_count;
```

Expected growth: ~200 bytes per outcome record.

---

## Security Notes

1. **User IDs are hashed**: The endpoint validates that user IDs are SHA256 hashes (64 hex characters). Raw Discord IDs will be rejected.

2. **No PII stored**: Only hashed identifiers are accepted.

3. **Validation**: The API validates all inputs:
   - `recId` must match URL parameter and body
   - `buyPrice`, `sellPrice`, `quantity` must be positive integers
   - `reportedAt` must be valid ISO 8601 timestamp

---

## Deployment Checklist

- [ ] Database migration applied (`trade_outcomes` table exists)
- [ ] Database user has INSERT/SELECT permissions
- [ ] API deployed with latest code
- [ ] Health check passes (`/api/v1/health`)
- [ ] Test POST to `/api/v1/recommendations/{recId}/outcome` succeeds
- [ ] Verify record appears in `trade_outcomes` table
- [ ] Discord bot updated to call new endpoint (separate deployment)

---

## Contact

For issues with this deployment, contact the development team.
