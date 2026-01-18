# Runbook: Full System Health Check

## Purpose
Verify all GePT services are running correctly.

## Pre-requisites
- SSH access to Ampere server
- Read access to PostgreSQL

## Procedure

### 1. Check Engine Service
```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# Check service status
sudo systemctl status recommendation-engine

# Verify health endpoint
curl http://localhost:8000/api/v1/health
```

**Expected:** Service active, health returns 200 OK

### 2. Check PostgreSQL
```bash
# Check service
sudo systemctl status postgresql

# Verify connection
psql -h localhost -U osrs_user -d osrs_data -c "SELECT 1;"
```

**Expected:** Service active, query returns 1

### 3. Check Prediction Freshness
```bash
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  MAX(time) as latest_prediction,
  NOW() - MAX(time) as age,
  CASE
    WHEN NOW() - MAX(time) < INTERVAL '10 minutes' THEN 'HEALTHY'
    ELSE 'STALE'
  END as status
FROM predictions;
"
```

**Expected:** Age < 10 minutes, status = HEALTHY

### 4. Check Collectors
```bash
cd /home/ubuntu/gept/collectors
docker compose ps
```

**Expected:** All collectors running (Up status)

### 5. Check Recent Data Ingestion
```bash
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  COUNT(*) as rows_last_hour,
  MIN(time) as oldest,
  MAX(time) as newest
FROM price_data_5min
WHERE time > NOW() - INTERVAL '1 hour';
"
```

**Expected:** rows_last_hour > 0, timestamps within last hour

## Verification Checklist

- [ ] Engine service: ACTIVE
- [ ] Engine health endpoint: 200 OK
- [ ] PostgreSQL: ACTIVE
- [ ] Predictions: < 10 minutes old
- [ ] Collectors: All UP
- [ ] Data ingestion: Recent data present

## Troubleshooting

**Engine not responding:**
1. Check logs: `sudo journalctl -u recommendation-engine -n 50`
2. Try restart: `sudo systemctl restart recommendation-engine`

**Stale predictions:**
1. Check Hydra inference cron
2. Check inference logs: `tail -50 /home/ubuntu/gept/logs/inference.log`
3. Verify Ampere fallback is running

**Collector issues:**
1. Check specific collector: `docker compose logs collector-5m`
2. Restart: `docker compose restart collector-5m`
