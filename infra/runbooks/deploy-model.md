# Runbook: Deploy New Trained Models

## Purpose
Deploy newly trained CatBoost models to production inference.

## Risk Level
MEDIUM - Affects prediction quality. Old models remain as fallback.

## Pre-requisites
- SSH access to Hydra and Ampere servers
- New models trained and validated
- Model metrics (AUC, calibration) reviewed

## Pre-checks

```bash
# On Hydra (where models are trained)
ssh hydra

# 1. Verify new models exist
ls -la ~/gept/models/
# Look for newest directory (format: YYYYMMDD_HHMMSS)

# 2. Check model count (should be 314 items)
ls ~/gept/models/YYYYMMDD_HHMMSS/ | wc -l

# 3. Verify registry.json exists
cat ~/gept/models/YYYYMMDD_HHMMSS/registry.json | head -20
```

## Procedure

### Step 1: Validate Models Locally
```bash
# On Hydra
cd ~/gept

# Run validation script
python -m src.training.model_validator \
  --models-dir models/YYYYMMDD_HHMMSS \
  --min-auc 0.60

# Check for any failed validations
```

### Step 2: Update Model Registry
```bash
# Update the symlink to point to new models
cd ~/gept/models
ln -sfn YYYYMMDD_HHMMSS current
ls -la current  # Verify symlink
```

### Step 3: Test Inference Locally
```bash
# Run a test inference cycle
cd ~/gept
python run_inference.py --dry-run --models-dir models/current

# Check output for errors
```

### Step 4: Deploy to Production

**Option A: Wait for next cron cycle (safest)**
```bash
# Models will be picked up automatically on next inference run
# Cron runs every 5 minutes
```

**Option B: Trigger immediate inference**
```bash
# Force an inference run now
cd ~/gept
./scripts/run_inference_hydra.sh

# Monitor logs
tail -f logs/inference.log
```

### Step 5: Verify Predictions
```bash
# SSH to Ampere and check predictions
ssh -i .secrets/oracle_key.pem ubuntu@150.136.170.128

psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  model_version,
  COUNT(*) as predictions,
  AVG(fill_probability) as avg_prob,
  MAX(time) as latest
FROM predictions
WHERE time > NOW() - INTERVAL '30 minutes'
GROUP BY model_version
ORDER BY latest DESC;
"
```

## Verification Checklist

- [ ] New model directory exists with 314 items
- [ ] registry.json is valid
- [ ] Validation script passes (AUC > 0.60)
- [ ] Dry-run inference completes without errors
- [ ] Predictions appear in database with new model_version
- [ ] Prediction quality looks reasonable (avg prob 0.03-0.30)

## Rollback

If new models produce bad predictions:

```bash
# On Hydra
cd ~/gept/models

# Point back to previous version
ln -sfn PREVIOUS_YYYYMMDD_HHMMSS current
ls -la current  # Verify

# Trigger new inference
cd ~/gept
./scripts/run_inference_hydra.sh

# Verify old predictions flowing
```

## Model Version Tracking

Always record in deployment log:
- Old model version
- New model version
- Training date
- Key metrics (avg AUC, calibration)
- Reason for update

## Troubleshooting

**Inference fails with new models:**
```bash
# Check for corrupt model files
python -c "from catboost import CatBoostClassifier; CatBoostClassifier().load_model('models/current/554/model.cbm')"

# Check feature compatibility
# New models must expect same 102 features as before
```

**Predictions look wrong:**
```bash
# Compare to historical
psql -h localhost -U osrs_user -d osrs_data -c "
SELECT
  DATE(time) as date,
  AVG(fill_probability) as avg_prob,
  COUNT(*) as count
FROM predictions
WHERE time > NOW() - INTERVAL '7 days'
GROUP BY DATE(time)
ORDER BY date;
"
```
