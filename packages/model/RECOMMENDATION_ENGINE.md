# GePT Prediction Engine - Recommendation Engine Integration Guide

This document describes how to interact with the GePT prediction system deployed on the internal production server.

## Overview

The prediction engine runs every 5 minutes, generating fill probability predictions for OSRS Grand Exchange trades. Predictions are stored in a PostgreSQL/TimescaleDB database and refreshed continuously.

---

## Server Access

### Production Server Details

Server configuration is centralized in `config/servers.env` (values redacted). Override via environment variables.

| Property | Variable | Default |
|----------|----------|---------|
| Host | `$AMPERE_IP` | `<set in environment>` |
| User@Host | `$AMPERE_HOST` | `<set in environment>` |
| SSH Key | `$AMPERE_SSH_KEY` | `<path to key>` |
| Database Port | - | `5432` |

### SSH Connection

```bash
# Using config variable (recommended)
ssh -i $AMPERE_SSH_KEY $AMPERE_HOST

# Or with defaults
ssh -i <ssh_key_path> <ssh_user>@<host>
```

### Database Connection (from production server)

```bash
psql -h localhost -U osrs_user -d osrs_data
# Password: $DB_PASS
```

### Database Connection (from remote machine via SSH tunnel)

```bash
# Terminal 1: Create tunnel (using config variable)
ssh -i $AMPERE_SSH_KEY -L 5432:localhost:5432 $AMPERE_HOST

# Terminal 2: Connect through tunnel
psql -h localhost -U osrs_user -d osrs_data
# Password: $DB_PASS
```

### Connection String

```
postgresql://osrs_user:$DB_PASS@localhost:5432/osrs_data
```

---

## Database Schema

### Predictions Table

The `predictions` table contains model outputs, refreshed every 5 minutes.

```sql
CREATE TABLE predictions (
    id BIGSERIAL,
    time TIMESTAMPTZ NOT NULL,              -- When prediction was generated
    item_id INTEGER NOT NULL,               -- OSRS item ID
    item_name TEXT NOT NULL,                -- Human-readable name
    hour_offset INTEGER NOT NULL,           -- Hours ahead (1-24)
    target_hour TIMESTAMPTZ NOT NULL,       -- When the trade should fill by
    offset_pct DECIMAL(5,4) NOT NULL,       -- Price offset (0.02 = 2%)
    fill_probability DECIMAL(7,6) NOT NULL, -- Model's fill probability [0,1]
    expected_value DECIMAL(8,6) NOT NULL,   -- EV = probability × net_profit
    buy_price DECIMAL(12,2) NOT NULL,       -- Suggested buy price
    sell_price DECIMAL(12,2) NOT NULL,      -- Suggested sell price
    current_high DECIMAL(12,2),             -- Current high price
    current_low DECIMAL(12,2),              -- Current low price
    confidence TEXT NOT NULL DEFAULT 'medium',
    model_version TEXT DEFAULT 'v1',
    PRIMARY KEY (time, item_id, hour_offset, offset_pct)
);
```

### Key Columns Explained

| Column | Description |
|--------|-------------|
| `time` | When this prediction was generated (refreshes every 5 min) |
| `item_id` | OSRS item ID (e.g., 565 = Blood rune) |
| `hour_offset` | Prediction horizon: 1-24 hours into the future |
| `target_hour` | `time + hour_offset` - when the trade should complete |
| `offset_pct` | Buy/sell offset from current price (typically 0.02 = 2%) |
| `fill_probability` | Probability that BOTH buy and sell orders fill within `hour_offset` hours |
| `expected_value` | `fill_probability × (2 × offset_pct - 0.02)` net profit expectation |
| `buy_price` | `current_low × (1 - offset_pct)` - price to set buy offer |
| `sell_price` | `current_high × (1 + offset_pct)` - price to set sell offer |
| `confidence` | `low`, `medium`, or `high` based on model validation metrics |

---

## Querying Predictions

### Get Latest Predictions (Most Recent Batch)

```sql
SELECT * FROM latest_predictions;
```

Or explicitly:

```sql
SELECT *
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions);
```

### Top Opportunities by Expected Value

```sql
SELECT * FROM top_opportunities;
```

Or with custom filters:

```sql
SELECT
    item_id,
    item_name,
    hour_offset,
    fill_probability,
    expected_value,
    buy_price,
    sell_price
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions)
  AND fill_probability >= 0.05      -- At least 5% chance
  AND fill_probability < 0.50       -- Filter overconfident predictions
  AND expected_value > 0.01         -- At least 1% expected profit
ORDER BY expected_value DESC
LIMIT 20;
```

### Predictions for a Specific Item

```sql
SELECT
    hour_offset,
    fill_probability,
    expected_value,
    buy_price,
    sell_price,
    confidence
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions)
  AND item_id = 565  -- Blood rune
ORDER BY hour_offset;
```

### Best Short-Term Opportunities (1-4 hours)

```sql
SELECT
    item_id,
    item_name,
    hour_offset,
    fill_probability,
    expected_value,
    buy_price,
    sell_price
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions)
  AND hour_offset <= 4
  AND fill_probability BETWEEN 0.05 AND 0.50
ORDER BY expected_value DESC
LIMIT 10;
```

---

## Handling 5-Minute Updates

### Update Pattern

The prediction engine runs on a cron schedule:

```
*/5 * * * *  →  Predictions refresh at :00, :05, :10, :15, ...
```

Each refresh:
1. Loads 72 hours of recent price data
2. Computes 113 features per item
3. Generates predictions for ~330 items × multiple hour offsets
4. Writes ~6,000+ predictions to the `predictions` table

### Recommended Polling Strategy

**Option A: Poll for Latest Timestamp**

```python
import psycopg2
from datetime import datetime, timedelta

def get_latest_predictions(conn):
    """Fetch predictions only if newer than our last fetch."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(time) FROM predictions")
        latest = cur.fetchone()[0]
    return latest

# Poll every 30 seconds, only process if timestamp changed
last_seen = None
while True:
    latest = get_latest_predictions(conn)
    if latest != last_seen:
        # New predictions available
        process_new_predictions(conn, latest)
        last_seen = latest
    time.sleep(30)
```

**Option B: Subscribe via LISTEN/NOTIFY (Advanced)**

Set up a trigger on the predictions table:

```sql
-- On production server, run once:
CREATE OR REPLACE FUNCTION notify_new_predictions()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('new_predictions', NEW.time::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER predictions_notify
AFTER INSERT ON predictions
FOR EACH STATEMENT
EXECUTE FUNCTION notify_new_predictions();
```

Then in your application:

```python
import psycopg2
import select

conn = psycopg2.connect(...)
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

cur = conn.cursor()
cur.execute("LISTEN new_predictions;")

while True:
    if select.select([conn], [], [], 60) != ([], [], []):
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            print(f"New predictions at: {notify.payload}")
            process_new_predictions(conn)
```

**Option C: Align to 5-Minute Boundaries**

```python
from datetime import datetime
import time

def wait_for_next_cycle():
    """Wait until next 5-minute boundary + 30 seconds buffer."""
    now = datetime.now()
    seconds_past = (now.minute % 5) * 60 + now.second
    wait_time = (5 * 60) - seconds_past + 30  # 30s buffer for inference to complete
    if wait_time > 300:
        wait_time -= 300
    time.sleep(wait_time)

while True:
    wait_for_next_cycle()
    predictions = fetch_latest_predictions(conn)
    process_predictions(predictions)
```

---

## Interpreting Predictions

### Fill Probability

The `fill_probability` represents:
> "What is the probability that if I place a buy order at `buy_price` and a sell order at `sell_price`, BOTH will fill within `hour_offset` hours?"

**Key thresholds:**
- `< 0.01`: Very unlikely to fill, not actionable
- `0.05 - 0.15`: Moderate opportunity, worth considering
- `0.15 - 0.30`: Strong signal, high priority
- `0.30 - 0.50`: Very strong signal (high-volume items, short windows)
- `> 0.50`: Model is likely overconfident, capped at 0.50

### Expected Value

```
expected_value = fill_probability × net_profit_margin
net_profit_margin = (2 × offset_pct) - 0.02  # 2% GE tax
```

For a 2% offset:
```
net_profit = (2 × 0.02) - 0.02 = 0.02 = 2%
```

If `fill_probability = 0.15`:
```
expected_value = 0.15 × 0.02 = 0.003 = 0.3% per trade
```

**Recommendation:** Target `expected_value > 0.005` (0.5%) for actionable trades.

### Hour Offset Selection

| Hour Range | Use Case |
|------------|----------|
| 1-4 hours | Quick flips, requires active monitoring |
| 5-12 hours | Medium-term, check a few times |
| 13-24 hours | Overnight/passive trades |

Short-term predictions (hours 1-4) are generally more accurate but require faster execution.

### Confidence Levels

| Level | Meaning |
|-------|---------|
| `high` | Model passed all validation checks, ROC-AUC > 0.75 |
| `medium` | Model is valid but has moderate uncertainty |
| `low` | Use with caution, model may be poorly calibrated |

---

## Example Integration Code

### Python: Fetch Top Opportunities

```python
import psycopg2
from psycopg2.extras import RealDictCursor

CONN_PARAMS = {
    'host': 'localhost',  # Use SSH tunnel
    'port': 5432,
    'database': 'osrs_data',
    'user': 'osrs_user',
    'password': '$DB_PASS'
}

def get_top_opportunities(min_ev=0.005, max_hour=12, limit=20):
    """Fetch top trading opportunities from latest predictions."""
    conn = psycopg2.connect(**CONN_PARAMS)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    item_id,
                    item_name,
                    hour_offset,
                    fill_probability,
                    expected_value,
                    buy_price,
                    sell_price,
                    current_high,
                    current_low,
                    confidence
                FROM predictions
                WHERE time = (SELECT MAX(time) FROM predictions)
                  AND fill_probability BETWEEN 0.03 AND 0.50
                  AND expected_value >= %s
                  AND hour_offset <= %s
                ORDER BY expected_value DESC
                LIMIT %s
            """, (min_ev, max_hour, limit))
            return cur.fetchall()
    finally:
        conn.close()

# Usage
opportunities = get_top_opportunities(min_ev=0.01, max_hour=6)
for opp in opportunities:
    print(f"{opp['item_name']}: "
          f"Buy@{opp['buy_price']:.0f}, Sell@{opp['sell_price']:.0f}, "
          f"P(fill)={opp['fill_probability']:.1%}, "
          f"EV={opp['expected_value']*100:.2f}%")
```

### Python: Continuous Monitoring

```python
import time
from datetime import datetime

def monitor_predictions(callback, poll_interval=30):
    """Continuously monitor for new predictions."""
    conn = psycopg2.connect(**CONN_PARAMS)
    last_time = None

    try:
        while True:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(time) FROM predictions")
                current_time = cur.fetchone()[0]

            if current_time and current_time != last_time:
                print(f"[{datetime.now()}] New predictions available: {current_time}")
                opportunities = get_top_opportunities()
                callback(opportunities)
                last_time = current_time

            time.sleep(poll_interval)
    finally:
        conn.close()

# Usage
def on_new_predictions(opportunities):
    print(f"Found {len(opportunities)} opportunities")
    for opp in opportunities[:5]:
        print(f"  {opp['item_name']}: EV={opp['expected_value']*100:.2f}%")

monitor_predictions(on_new_predictions)
```

### Node.js / TypeScript

```typescript
import { Pool } from 'pg';

const pool = new Pool({
  host: 'localhost',  // Use SSH tunnel
  port: 5432,
  database: 'osrs_data',
  user: 'osrs_user',
  password: '$DB_PASS',
});

interface Prediction {
  item_id: number;
  item_name: string;
  hour_offset: number;
  fill_probability: number;
  expected_value: number;
  buy_price: number;
  sell_price: number;
  confidence: string;
}

async function getTopOpportunities(
  minEv = 0.005,
  maxHour = 12,
  limit = 20
): Promise<Prediction[]> {
  const result = await pool.query(`
    SELECT
      item_id,
      item_name,
      hour_offset,
      fill_probability::float,
      expected_value::float,
      buy_price::float,
      sell_price::float,
      confidence
    FROM predictions
    WHERE time = (SELECT MAX(time) FROM predictions)
      AND fill_probability BETWEEN 0.03 AND 0.50
      AND expected_value >= $1
      AND hour_offset <= $2
    ORDER BY expected_value DESC
    LIMIT $3
  `, [minEv, maxHour, limit]);

  return result.rows;
}

// Usage
const opportunities = await getTopOpportunities(0.01, 6);
console.log(`Top ${opportunities.length} opportunities:`);
opportunities.forEach(opp => {
  console.log(`${opp.item_name}: Buy@${opp.buy_price}, EV=${(opp.expected_value * 100).toFixed(2)}%`);
});
```

---

## Supporting Tables

### Items Metadata

```sql
SELECT item_id, name, members, highalch, lowalch
FROM items
WHERE item_id = 565;
```

### Historical Prices

```sql
-- Recent 5-minute prices
SELECT timestamp, avg_high_price, avg_low_price, high_price_volume, low_price_volume
FROM price_data_5min
WHERE item_id = 565
ORDER BY timestamp DESC
LIMIT 100;

-- Hourly aggregates
SELECT timestamp, avg_high_price, avg_low_price
FROM prices_1h
WHERE item_id = 565
ORDER BY timestamp DESC
LIMIT 24;
```

### Actual Fill Tracking (Calibration)

```sql
-- Check model calibration over last 7 days
SELECT * FROM calibration_summary;
```

---

## Cron Schedule Details

The inference engine runs via cron on the production server. Deployment paths are redacted here; use your server's configured deploy root.

```bash
# View current cron jobs
crontab -l

# Expected entry:
*/5 * * * * <deploy_root>/scripts/run_inference_cron.sh >> <deploy_root>/logs/inference.log 2>&1
```

### Inference Cycle Performance

Typical cycle times (314 items, 5,596 predictions):
- Model loading: ~25s (first run only, cached after)
- Price data loading: ~1s
- Feature computation: ~21s
- Prediction generation: ~90s
- Database writes: <1s
- **Total wall clock: ~2 minutes**

Since the cycle completes well under 5 minutes, there's no overlap between runs.

### Log Monitoring

```bash
# SSH into server (using config variable)
ssh -i $AMPERE_SSH_KEY $AMPERE_HOST

# Watch live logs
tail -f <deploy_root>/logs/inference.log

# Check last run
tail -100 <deploy_root>/logs/inference.log
```

### Manual Inference Run

```bash
cd <deploy_root>
source venv/bin/activate
python run_inference.py --dry-run  # Test without DB writes
python run_inference.py            # Full run with DB writes
```

---

## Troubleshooting

### No Recent Predictions

```sql
-- Check when last predictions were generated
SELECT MAX(time) as last_update,
       NOW() - MAX(time) as age
FROM predictions;
```

If older than 10 minutes, check the cron job and logs on the production server.

### Database Connection Issues

1. Ensure SSH tunnel is active (if connecting remotely)
2. Check PostgreSQL is running: `sudo systemctl status postgresql`
3. Verify credentials in connection string

### Inference Errors

Check logs on production server:
```bash
tail -200 <deploy_root>/logs/inference.log | grep -i error
```

Common issues:
- Missing model files → Re-run deployment
- Database connection timeout → Check PostgreSQL status
- Memory issues → Monitor with `htop`

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  Production Server (internal)                   │
│                       (<host>)                                  │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Cron Job   │────▶│  Inference   │────▶│  PostgreSQL  │   │
│  │  (5 min)     │     │   Engine     │     │  predictions │   │
│  └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                    │           │
│  ┌──────────────┐     ┌──────────────┐            │           │
│  │    Models    │     │ Price Data   │            │           │
│  │  (331 items) │     │ (5min/1min)  │            │           │
│  └──────────────┘     └──────────────┘            │           │
│                                                    │           │
└────────────────────────────────────────────────────┼───────────┘
                                                     │
                              SSH Tunnel (port 5432) │
                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Recommendation Engine                        │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │    Poll /    │────▶│    Filter    │────▶│   Display    │   │
│  │   Subscribe  │     │   & Rank     │     │   to User    │   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

| Task | Command/Query |
|------|---------------|
| SSH to server | `ssh -i $AMPERE_SSH_KEY $AMPERE_HOST` |
| Create tunnel | `ssh -i $AMPERE_SSH_KEY -L 5432:localhost:5432 $AMPERE_HOST` |
| Connect to DB | `psql -h localhost -U osrs_user -d osrs_data` |
| Latest predictions | `SELECT * FROM latest_predictions;` |
| Top opportunities | `SELECT * FROM top_opportunities;` |
| Check last update | `SELECT MAX(time) FROM predictions;` |
| View cron logs | `tail -f <deploy_root>/logs/inference.log` |
| Manual inference | `cd <deploy_root> && source venv/bin/activate && python run_inference.py` |
