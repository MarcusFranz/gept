# Runbook: Restart PostgreSQL

## Purpose
Restart the PostgreSQL database server.

## Risk Level
**HIGH** - This affects ALL services:
- Recommendation engine will lose connections
- Collectors will fail writes temporarily
- Inference will fail if running

## When to Use
- PostgreSQL is unresponsive
- After configuration changes
- Memory issues requiring restart
- **NOT for routine maintenance**

## Pre-requisites
- SSH access to Ampere server
- Understanding of current system state
- Ideally: Low-traffic period

## Pre-checks

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# 1. Confirm PostgreSQL is actually the problem
sudo systemctl status postgresql
psql -h localhost -U osrs_user -d osrs_data -c "SELECT 1;"

# 2. Check what's connected
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT pid, usename, application_name, state, query_start
FROM pg_stat_activity
WHERE datname = 'osrs_data';
"

# 3. Check for long-running queries
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT pid, now() - query_start as duration, query
FROM pg_stat_activity
WHERE state = 'active' AND query_start < now() - interval '1 minute';
"

# 4. Note current prediction freshness
psql -h localhost -U osrs_user -d osrs_data -c "SELECT MAX(time) FROM predictions;"
```

## Notify Before Proceeding

**This will cause temporary service disruption.**

Services affected:
- Recommendation engine (will reconnect automatically)
- Data collectors (will retry automatically)
- Inference (will fail if running during restart)

## Procedure

### Step 1: Stop Dependent Services (Optional but Safer)
```bash
# Stop engine to prevent connection errors
sudo systemctl stop recommendation-engine

# Stop collectors
cd /home/ubuntu/gept/collectors
docker compose stop
```

### Step 2: Restart PostgreSQL
```bash
sudo systemctl restart postgresql

# Wait for it to be ready
sleep 10

# Verify it's accepting connections
psql -h localhost -U osrs_user -d osrs_data -c "SELECT 1;"
```

### Step 3: Restart Dependent Services
```bash
# Start collectors first (data ingestion)
cd /home/ubuntu/gept/collectors
docker compose start

# Verify collectors running
docker compose ps

# Start recommendation engine
sudo systemctl start recommendation-engine

# Verify engine
sleep 5
curl http://localhost:8000/api/v1/health
```

### Step 4: Verify System Health
```bash
# Check all services
sudo systemctl status postgresql
sudo systemctl status recommendation-engine
docker compose ps

# Check data is flowing
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  'predictions' as table_name, MAX(time) as latest
FROM predictions
UNION ALL
SELECT
  'price_data_5min', MAX(time)
FROM price_data_5min;
"

# Check engine health
curl http://localhost:8000/api/v1/health
```

## Verification Checklist

- [ ] PostgreSQL status: active
- [ ] Can connect with psql
- [ ] Recommendation engine: healthy
- [ ] Collectors: all running
- [ ] New data appearing in price_data_5min
- [ ] Predictions being generated

## Rollback

PostgreSQL restart is not easily "rolled back." If issues persist:

```bash
# Check PostgreSQL logs
sudo tail -100 /var/log/postgresql/postgresql-*-main.log

# Check disk space
df -h

# Check memory
free -h

# If configuration issue, restore previous config
sudo cp /etc/postgresql/*/main/postgresql.conf.bak /etc/postgresql/*/main/postgresql.conf
sudo systemctl restart postgresql
```

## Troubleshooting

**PostgreSQL won't start:**
```bash
# Check logs
sudo journalctl -u postgresql -n 100 --no-pager

# Common issues:
# - Disk full: df -h
# - Corrupt WAL: Check pg_wal directory
# - Config error: Check postgresql.conf syntax
```

**Services won't reconnect:**
```bash
# Engine: Force restart
sudo systemctl restart recommendation-engine

# Collectors: Full restart
cd /home/ubuntu/gept/collectors
docker compose down
docker compose up -d
```

## Prevention

To avoid needing PostgreSQL restarts:
- Monitor disk space (alert at 80%)
- Monitor connection count
- Schedule maintenance during low-traffic periods
- Use connection pooling (PgBouncer) if connection issues frequent
