# Runbook: Check Inference Pipeline

## Purpose
Verify the ML inference pipeline is running correctly and producing fresh predictions.

## Risk Level
NONE - Read-only checks.

## When to Use
- Recommendations seem stale or missing
- Scheduled health check
- After model deployment
- Debugging prediction issues

## Quick Check (30 seconds)

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# Check prediction freshness
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  MAX(time) as latest_prediction,
  NOW() - MAX(time) as age,
  CASE
    WHEN NOW() - MAX(time) < INTERVAL '10 minutes' THEN 'HEALTHY'
    WHEN NOW() - MAX(time) < INTERVAL '20 minutes' THEN 'WARNING'
    ELSE 'STALE'
  END as status
FROM predictions;
"
```

**Expected:** Age < 10 minutes, Status = HEALTHY

## Full Pipeline Check

### 1. Check Hydra Inference (Primary)

```bash
# SSH to Hydra
ssh hydra

# Check cron is running
crontab -l | grep inference

# Check recent inference logs
tail -50 ~/gept/logs/inference.log

# Check for errors
grep -i error ~/gept/logs/inference.log | tail -20

# Check last successful run
grep "Inference complete" ~/gept/logs/inference.log | tail -5
```

### 2. Check Ampere Fallback

```bash
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

# Check fallback cron
crontab -l | grep inference

# Check fallback logs
tail -50 /home/ubuntu/gept/logs/inference.log

# Fallback should show "Predictions fresh, skipping" if Hydra is working
grep "fresh\|skipping" /home/ubuntu/gept/logs/inference.log | tail -5
```

### 3. Check Data Freshness (Input to Inference)

```bash
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  'price_data_5min' as source,
  MAX(time) as latest,
  NOW() - MAX(time) as age
FROM price_data_5min
UNION ALL
SELECT
  'predictions',
  MAX(time),
  NOW() - MAX(time)
FROM predictions;
"
```

**Expected:**
- price_data_5min: < 10 minutes old
- predictions: < 10 minutes old

### 4. Check Prediction Quality

```bash
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  COUNT(*) as total_predictions,
  COUNT(DISTINCT item_id) as unique_items,
  AVG(fill_probability) as avg_prob,
  MIN(fill_probability) as min_prob,
  MAX(fill_probability) as max_prob,
  AVG(expected_value) as avg_ev
FROM predictions
WHERE time = (SELECT MAX(time) FROM predictions);
"
```

**Expected:**
- total_predictions: ~33,000+ (314 items × 108 targets)
- unique_items: 314
- avg_prob: 0.05 - 0.15 (varies)
- Reasonable EV distribution

### 5. Check Model Status

```bash
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT status, COUNT(*) as count
FROM model_registry
GROUP BY status;
"
```

**Expected:** Most models in ACTIVE status

## Diagnosis Tree

```
Predictions stale?
├── Yes → Check Hydra inference
│   ├── Cron not running → Fix cron
│   ├── Errors in log → Debug errors
│   └── Running but no output → Check DB connection
│
└── Predictions exist but look wrong?
    ├── All zeros → Model loading issue
    ├── All same value → Feature computation issue
    └── Missing items → Check model registry
```

## Common Issues & Fixes

**Inference not running (Hydra):**
```bash
# Check if process is stuck
ps aux | grep inference

# Kill and restart manually
pkill -f run_inference
cd ~/gept && ./scripts/run_inference_hydra.sh
```

**Database connection failed:**
```bash
# Test connection from Hydra
psql -h 150.136.170.128 -U osrs_user -d osrs_data -c "SELECT 1;"

# Check SSH tunnel (if used)
ps aux | grep "ssh.*5432"
```

**Predictions very old (>1 hour):**
```bash
# Emergency: Run Ampere fallback manually
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128
cd /home/ubuntu/gept
./scripts/run_inference_fallback.sh --force
```

**Price data not updating:**
```bash
# Check collectors
cd /home/ubuntu/gept/collectors
docker compose ps
docker compose logs collector-5m --tail=50
```

## Monitoring Queries

Save these for regular checks:

```sql
-- Prediction freshness by hour
SELECT
  date_trunc('hour', time) as hour,
  COUNT(*) as predictions,
  COUNT(DISTINCT item_id) as items
FROM predictions
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY 1
ORDER BY 1 DESC;

-- Model version distribution
SELECT
  model_version,
  COUNT(*) as predictions,
  MAX(time) as latest
FROM predictions
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY model_version;
```
