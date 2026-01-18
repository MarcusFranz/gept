# GePT Continuous Training Pipeline

**Date:** 2026-01-17
**Status:** Design Document

## Overview

A continuous training system that keeps the GPU fully utilized, prioritizing models based on drift detection rather than simple age-based scheduling.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AMPERE SERVER                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Inference    │───▶│ Drift        │───▶│ model_performance    │  │
│  │ (5 min)      │    │ Detection    │    │ table                │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                     │               │
│  ┌──────────────┐                                   │               │
│  │ Data         │◀──────── sync every 5 min ────────┘               │
│  │ Collectors   │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SSH Tunnel
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         HYDRA SERVER                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Local Data   │◀───│ Data Sync    │◀───│ PostgreSQL           │  │
│  │ Cache        │    │ Service      │    │ (via tunnel)         │  │
│  │ (Parquet)    │    └──────────────┘    └──────────────────────┘  │
│  └──────────────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    PRIORITY SCHEDULER                         │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐  │  │
│  │  │ CRITICAL   │ │ WARNING    │ │ PREDICTED  │ │ STABLE    │  │  │
│  │  │ DRIFT      │ │ DRIFT      │ │ DRIFT      │ │ (idle)    │  │  │
│  │  │ >10% drop  │ │ 5-10% drop │ │ trending ↓ │ │ quarterly │  │  │
│  │  │ OPT+TRAIN  │ │ TRAIN      │ │ TRAIN      │ │ OPT       │  │  │
│  │  └────────────┘ └────────────┘ └────────────┘ └───────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      GPU EXECUTOR                             │  │
│  │  ┌────────────────────┐    ┌────────────────────────────┐    │  │
│  │  │ Hyperparameter     │    │ Model Training             │    │  │
│  │  │ Optimization       │    │ (category-specific params) │    │  │
│  │  │ (Optuna + GPU)     │    │                            │    │  │
│  │  └────────────────────┘    └────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐                                                   │
│  │ Model Upload │──────────▶ Ampere (model_registry, S3/local)     │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Priority Queue Design

### Priority Levels (Highest to Lowest)

| Priority | Condition | Action | Max Age |
|----------|-----------|--------|---------|
| P0: CRITICAL_DRIFT | AUC dropped >10% from baseline | OPTIMIZE + TRAIN | Immediate |
| P1: WARNING_DRIFT | AUC dropped 5-10% from baseline | TRAIN | Within 24h |
| P2: PREDICTED_DRIFT | AUC trending toward threshold | TRAIN | Within 48h |
| P3: STALE | Model age > 7 days | TRAIN | Within 7 days |
| P4: STABLE | Model healthy, no optimization in 90 days | OPTIMIZE | Quarterly |

### Drift Detection Metrics

```python
class DriftMetrics:
    # Baseline is rolling 7-day average AUC from model_performance table
    baseline_auc: float
    current_auc: float  # Last 24h average
    auc_trend: float    # Slope of AUC over last 7 days

    @property
    def drift_severity(self) -> str:
        drop = (self.baseline_auc - self.current_auc) / self.baseline_auc
        if drop > 0.10:
            return "CRITICAL"
        elif drop > 0.05:
            return "WARNING"
        elif self.auc_trend < -0.01:  # Declining trend
            return "PREDICTED"
        return "STABLE"
```

### Queue Management

```python
@dataclass
class TrainingJob:
    item_id: int
    category: str
    priority: int  # 0-4 (lower = higher priority)
    action: str    # "OPTIMIZE", "TRAIN", or "OPTIMIZE_AND_TRAIN"
    created_at: datetime
    drift_severity: str
    current_auc: float
    model_age_days: int

class PriorityQueue:
    def get_next_job(self) -> TrainingJob:
        """
        Returns highest priority job, with tie-breaking:
        1. Priority level (P0 > P1 > P2 > P3 > P4)
        2. Drift severity (CRITICAL > WARNING > PREDICTED > STABLE)
        3. Model age (oldest first within same priority)
        """
        pass
```

## Local Data Cache

### Design Goals
- Eliminate DB query latency for training
- Keep data fresh (append every 5 minutes)
- Support parallel item training without DB contention

### Implementation

```
/home/ubuntu/gept/data_cache/
├── price_data/
│   ├── item_2.parquet       # Cannonball
│   ├── item_561.parquet     # Nature rune
│   └── ...
├── metadata/
│   ├── items.parquet
│   └── last_sync.json       # {"timestamp": "2026-01-17T12:00:00Z"}
└── sync.log
```

### Sync Service

```python
# data_sync_service.py
class DataSyncService:
    """
    Runs as systemd service on Hydra.
    - Every 5 minutes: append new price data
    - Every hour: full sync of metadata tables
    - On startup: verify cache integrity
    """

    def append_prices(self):
        """Append only new rows since last sync"""
        last_ts = self.get_last_sync_timestamp()
        for item_id in self.eligible_items:
            new_data = self.query_new_prices(item_id, since=last_ts)
            if not new_data.empty:
                self.append_to_parquet(item_id, new_data)

    def query_new_prices(self, item_id: int, since: datetime) -> pd.DataFrame:
        """Query via SSH tunnel to Ampere PostgreSQL"""
        query = f"""
            SELECT timestamp, high_price, low_price, high_volume, low_volume
            FROM price_data_5min
            WHERE item_id = {item_id} AND timestamp > '{since}'
            ORDER BY timestamp
        """
        return pd.read_sql(query, self.conn)
```

## Category-Specific Hyperparameters

Based on experiment results (see HYPERPARAMETER_EXPERIMENT_RESULTS.md):

```python
CATEGORY_HYPERPARAMS = {
    "high_volume_consumables": {
        "depth": 3,
        "learning_rate": 0.15,
        "l2_leaf_reg": 1.0,
        "iterations": 500,
        "od_wait": 50,
    },
    "mid_volume_resources": {
        "depth": 5,
        "learning_rate": 0.15,
        "l2_leaf_reg": 10.0,
        "iterations": 500,
        "od_wait": 50,
    },
    "potions_food": {
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "iterations": 500,
        "od_wait": 50,
    },
    "equipment_common": {
        "depth": 4,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "iterations": 500,
        "od_wait": 50,
        "days_history": 180,  # Longer window for rare items
    },
    "equipment_high_value": {
        "depth": 4,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "iterations": 500,
        "od_wait": 50,
        "days_history": 180,
    },
    "default": {
        "depth": 4,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
        "iterations": 500,
        "od_wait": 50,
    },
}
```

## Scheduler Logic

```python
class ContinuousTrainingScheduler:
    """
    Main scheduler loop - keeps GPU 100% utilized.
    """

    def run(self):
        while True:
            # 1. Refresh priority queue from model_performance table
            self.refresh_queue()

            # 2. Get next job
            job = self.queue.get_next_job()

            if job is None:
                # No urgent work - do proactive optimization
                job = self.get_proactive_optimization_job()

            if job is None:
                # All models are fresh and optimized - wait for drift
                logger.info("All models healthy, sleeping 5 minutes")
                time.sleep(300)
                continue

            # 3. Execute job
            logger.info(f"Starting {job.action} for item {job.item_id} "
                       f"(priority={job.priority}, drift={job.drift_severity})")

            try:
                if job.action == "OPTIMIZE_AND_TRAIN":
                    self.run_optimization(job)
                    self.run_training(job)
                elif job.action == "OPTIMIZE":
                    self.run_optimization(job)
                else:  # TRAIN
                    self.run_training(job)

                # 4. Upload model and update registry
                self.upload_model(job)
                self.update_registry(job)

            except Exception as e:
                logger.error(f"Job failed: {e}")
                self.mark_job_failed(job)

            # 5. Small delay to check for higher priority work
            time.sleep(1)

    def run_training(self, job: TrainingJob):
        """Train model with category-specific hyperparameters"""
        params = CATEGORY_HYPERPARAMS.get(job.category, CATEGORY_HYPERPARAMS["default"])

        # Load from local cache
        X, y = self.load_training_data(job.item_id, params.get("days_history", 60))

        # Train 36-target model
        model = train_multitarget_model(X, y, params)

        return model

    def run_optimization(self, job: TrainingJob):
        """Run Optuna optimization for item-specific hyperparameters"""
        # Use category defaults as starting point
        base_params = CATEGORY_HYPERPARAMS.get(job.category, CATEGORY_HYPERPARAMS["default"])

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.objective(trial, job.item_id, base_params),
            n_trials=20,  # ~15-20 minutes of optimization
            timeout=1200,  # Max 20 minutes
        )

        # Save optimized params to item_hyperparameters table
        self.save_item_hyperparameters(job.item_id, study.best_params)
```

## Database Schema Additions

```sql
-- Item-specific optimized hyperparameters
CREATE TABLE IF NOT EXISTS item_hyperparameters (
    item_id INTEGER PRIMARY KEY REFERENCES items(id),
    category VARCHAR(50) NOT NULL,
    depth INTEGER NOT NULL,
    learning_rate REAL NOT NULL,
    l2_leaf_reg REAL NOT NULL,
    days_history INTEGER NOT NULL DEFAULT 60,
    optimized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    optimization_auc REAL,
    notes TEXT
);

-- Training job history
CREATE TABLE IF NOT EXISTS training_jobs (
    id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES items(id),
    job_type VARCHAR(20) NOT NULL,  -- 'TRAIN', 'OPTIMIZE', 'OPTIMIZE_AND_TRAIN'
    priority INTEGER NOT NULL,
    drift_severity VARCHAR(20),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'RUNNING',  -- 'RUNNING', 'COMPLETED', 'FAILED'
    result_auc REAL,
    training_time_seconds REAL,
    error_message TEXT
);

-- Model performance tracking (extended)
ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS baseline_auc REAL;
ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS auc_trend REAL;
ALTER TABLE model_performance ADD COLUMN IF NOT EXISTS drift_severity VARCHAR(20);
```

## Systemd Services

### gept-scheduler.service
```ini
[Unit]
Description=GePT Continuous Training Scheduler
After=network.target db-tunnel.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/gept
Environment="PATH=/home/ubuntu/miniconda3/envs/gept/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ubuntu/miniconda3/envs/gept/bin/python -u src/continuous_scheduler.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

### gept-data-sync.service
```ini
[Unit]
Description=GePT Data Sync Service
After=network.target db-tunnel.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/gept
Environment="PATH=/home/ubuntu/miniconda3/envs/gept/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ubuntu/miniconda3/envs/gept/bin/python -u src/data_sync_service.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

## Implementation Plan

### Phase 1: Data Infrastructure
1. Create local data cache directory structure
2. Implement DataSyncService with 5-minute append
3. Create gept-data-sync.service
4. Verify cache integrity on startup

### Phase 2: Database Extensions
1. Add item_hyperparameters table
2. Extend model_performance with drift columns
3. Add training_jobs table for audit trail

### Phase 3: Priority Queue
1. Implement DriftMetrics calculation
2. Create PriorityQueue with multi-factor ordering
3. Add drift detection to inference pipeline

### Phase 4: Scheduler
1. Implement ContinuousTrainingScheduler
2. Integrate category-specific hyperparameters
3. Add Optuna optimization support
4. Create gept-scheduler.service

### Phase 5: Monitoring
1. Add Prometheus metrics for training jobs
2. Create Grafana dashboard for training status
3. Add alerting for critical drift

## Estimated Resource Usage

| Component | CPU | Memory | GPU | Disk I/O |
|-----------|-----|--------|-----|----------|
| Data Sync | 5% | 500MB | 0% | Low |
| Scheduler | 10% | 1GB | 0% | Low |
| Training | 20% | 4GB | 100% | Medium |
| Optimization | 20% | 4GB | 100% | Medium |

**Expected throughput:**
- Training: ~20-40 seconds per item (36 targets)
- Optimization: ~15-20 minutes per item (20 trials)
- Full 399-item training cycle: ~3-4 hours
- Quarterly optimization cycle: ~100-130 hours (spread across quarter)

## Next Steps

1. [ ] Implement DataSyncService
2. [ ] Create database schema extensions
3. [ ] Build priority queue with drift detection
4. [ ] Implement ContinuousTrainingScheduler
5. [ ] Add monitoring and alerting
