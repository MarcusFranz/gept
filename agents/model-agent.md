# Model Agent Context

You are a model agent for the GePT system. You handle ML pipeline work including data ingestion, feature engineering, training, and inference.

## Your Responsibilities

1. Data collectors (5min, 1min, hourly, news, player count, items)
2. Feature engineering (`feature_engine.py`)
3. Model training (CatBoost on Hydra)
4. Batch inference pipeline
5. Model registry and lifecycle management

## Package Location

`packages/model/` - ML pipeline code

## Key Files

```
packages/model/
├── src/
│   ├── batch_predictor_multitarget.py  # Production inference (108 targets)
│   ├── feature_engine.py                # 102 features from price data
│   ├── target_engine.py                 # Fill probability targets
│   ├── db_utils.py                      # Database connections
│   └── training/                        # Training orchestration
├── collectors/                          # Data ingestion (Docker)
├── cloud/                               # GPU training scripts
├── scripts/                             # Deployment, migrations
└── models/                              # Trained model artifacts
```

## Data Flow

```
Collectors → PostgreSQL (price_data_5min)
                ↓
Feature Engine (102 features from 24h of data)
                ↓
CatBoost Models (314 items × 108 targets)
                ↓
Predictions Table → Recommendation Engine
```

## Contract: Predictions Table

You write to this table. The recommendation engine reads from it.

```sql
CREATE TABLE predictions (
    time TIMESTAMPTZ NOT NULL,
    item_id INTEGER NOT NULL,
    hour_offset INTEGER NOT NULL,        -- 1-48 hours
    offset_pct DECIMAL(5,4) NOT NULL,    -- 0.0125-0.025 (1.25%-2.5%)
    fill_probability DECIMAL(7,6),       -- [0, 1] CLIPPED
    expected_value DECIMAL(8,6),
    buy_price DECIMAL(12,2),
    sell_price DECIMAL(12,2),
    confidence TEXT,                      -- low/medium/high
    model_version TEXT,
    PRIMARY KEY (time, item_id, hour_offset, offset_pct)
);
```

**CRITICAL**: If you change this schema, coordinate with engine agent first.

## Model Configuration

**Targets (108 per item):**
- 18 time windows: 1-12h hourly + 16h, 20h, 24h, 32h, 40h, 48h
- 6 offsets: 1.25%, 1.5%, 1.75%, 2%, 2.25%, 2.5%

**Features:**
- Price momentum (4h, 8h, 24h windows)
- Volatility metrics
- Volume-weighted metrics
- Lag features
- Time-of-day encoding

**Model Type:** CatBoost multi-target classifiers

## Inference Pipeline

**Primary (Hydra):**
```bash
# Cron every 5 minutes
*/5 * * * * /home/user/gept/scripts/run_inference_hydra.sh
```

**Fallback (Ampere):**
```bash
# Only runs if predictions > 10 minutes stale
*/5 * * * * /home/ubuntu/gept/scripts/run_inference_fallback.sh
```

## Common Tasks

### Check Inference Health
```bash
# Check prediction freshness
psql -h localhost -U osrs_user -d osrs_data \
  -c "SELECT MAX(time), NOW() - MAX(time) as age FROM predictions;"

# Check inference logs
tail -50 /home/ubuntu/gept/logs/inference.log
```

### Retrain Models
1. Prepare training data: `python cloud/prepare_runpod_data.py`
2. Run training on Hydra: `python cloud/train_runpod_multitarget.py`
3. Validate models: Check AUC, calibration metrics
4. Update model registry: `python scripts/update_model_registry.py`
5. Deploy: Coordinate with infra-agent

### Add New Collector
1. Create collector in `collectors/`
2. Add to `docker-compose.yml`
3. Test locally
4. Deploy via infra-agent

## Don't Touch (Engine Responsibility)

- Recommendation filtering logic
- User personalization
- API endpoints
- Crowding/rate limiting
