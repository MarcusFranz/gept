# Runbook: Restart Data Collectors

## Purpose
Restart the Docker-based data collectors that ingest price data.

## Risk Level
LOW - Collectors are stateless and auto-recover. Brief gap in data is acceptable.

## Collectors Overview

| Collector | Interval | Purpose |
|-----------|----------|---------|
| collector-5m | 5 min | Primary OHLC price data |
| collector-1m | 1 min | High-frequency ticks |
| collector-hourly | 1 hour | Historical aggregates |
| collector-news | 30 min | Market news events |
| collector-players | 1 min | Player count trends |
| item-updater | 24 hours | Item metadata sync |

## Pre-checks

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# Check current status
cd /home/ubuntu/gept/collectors
docker compose ps

# Check for errors
docker compose logs --tail=20
```

## Restart Single Collector

If only one collector is misbehaving:

```bash
cd /home/ubuntu/gept/collectors

# Restart specific collector
docker compose restart collector-5m

# Verify it's running
docker compose ps collector-5m

# Check logs
docker compose logs collector-5m --tail=30
```

## Restart All Collectors

```bash
cd /home/ubuntu/gept/collectors

# Graceful restart (preferred)
docker compose restart

# Verify all running
docker compose ps

# Check for startup errors
docker compose logs --tail=50
```

## Full Reset (If Issues Persist)

```bash
cd /home/ubuntu/gept/collectors

# Stop all
docker compose down

# Clear any stale state
docker system prune -f

# Start fresh
docker compose up -d

# Verify
docker compose ps
docker compose logs --tail=30
```

## Verification

### Check Collectors Are Writing Data

```bash
# Check recent price data
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  COUNT(*) as rows_last_10min
FROM price_data_5min
WHERE time > NOW() - INTERVAL '10 minutes';
"
```

**Expected:** Should see new rows appearing (varies by item count)

### Check Individual Collector Health

```bash
# 5-minute collector (most critical)
docker compose logs collector-5m --tail=10 | grep -i "success\|error\|saved"

# 1-minute collector
docker compose logs collector-1m --tail=10

# Check metrics endpoint (if enabled)
curl http://localhost:8080/metrics 2>/dev/null | grep collector
```

## Verification Checklist

- [ ] All containers showing "Up" status
- [ ] No error messages in recent logs
- [ ] New data appearing in price_data_5min
- [ ] Collector dashboard accessible (port 8080)

## Troubleshooting

**Collector won't start:**
```bash
# Check detailed logs
docker compose logs collector-5m

# Common issues:
# - Database connection: Check PostgreSQL is running
# - Port conflict: Check nothing else on required ports
# - Memory: Check available memory with `free -h`
```

**Collector starts but no data:**
```bash
# Check OSRS API is accessible
curl -s "https://prices.runescape.wiki/api/v1/osrs/latest" | head -100

# Check database write permissions
psql -h localhost -U osrs_user -d osrs_data -c "
INSERT INTO price_data_5min (time, item_id, high, low, volume)
VALUES (NOW(), 0, 0, 0, 0);
DELETE FROM price_data_5min WHERE item_id = 0;
"
```

**High memory usage:**
```bash
# Check container memory
docker stats --no-stream

# If a collector is using too much memory, restart it
docker compose restart collector-5m
```

**Gaps in data:**
```bash
# Check for gaps in last 24 hours
psql -h localhost -U osrs_user -d osrs_data -c "
WITH time_series AS (
  SELECT generate_series(
    NOW() - INTERVAL '24 hours',
    NOW(),
    INTERVAL '5 minutes'
  ) as expected_time
)
SELECT expected_time
FROM time_series
WHERE NOT EXISTS (
  SELECT 1 FROM price_data_5min
  WHERE time BETWEEN expected_time - INTERVAL '1 minute'
                 AND expected_time + INTERVAL '1 minute'
)
ORDER BY expected_time DESC
LIMIT 20;
"
```

## Monitoring Stack

The collectors run alongside monitoring:

```bash
# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3001/api/health

# View dashboards
# Open http://150.136.170.128:3001 (internal only)
```

## Emergency: Manual Data Collection

If collectors are completely broken and you need data urgently:

```bash
# Run single collection manually
cd /home/ubuntu/gept/collectors
python collect_5m_pg.py

# This will do one collection cycle and exit
```
