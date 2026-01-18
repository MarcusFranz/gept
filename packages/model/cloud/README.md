# GePT Training Pipeline

Train OSRS prediction models using GPU acceleration.

## GPU Training

Train models on local or cloud GPU instances. Supports RTX 3060 through A100.

### Performance by GPU

| GPU | VRAM | Time/Model | 400 Items | Notes |
|-----|------|------------|-----------|-------|
| RTX 6000 Ada | 48GB | 10-12s | ~70 min | Best throughput |
| RTX 4090 | 24GB | 15-20s | ~2 hours | Excellent value |
| RTX 3060 | 12GB | 40-50s | ~5-6 hours | Good for daily runs |
| CPU (16 cores) | N/A | 60-90s | ~8-10 hours | Fallback option |

### Quick Start

```bash
# 1. Prepare data (requires DB access)
export DB_PASS="your_password"
python prepare_runpod_data.py --months 6 --output-dir /path/to/data

# Note the run_id from output (e.g., 20260111_113226)

# 2. Train models
python train_runpod_multitarget.py \
    --run-id 20260111_113226 \
    --all \
    --local \
    --cache-dir /path/to/data \
    --output-dir /path/to/models

# 3. Transfer to production
scp -r /path/to/models/20260111_113226 ubuntu@ampere:/home/ubuntu/gept/models/
```

### GPU-Specific Settings

```bash
# High-end GPU (Ada/A100) - maximize throughput
python train_runpod_multitarget.py --run-id <id> --all --local \
    --threads 32 --numba-threads 16 --prefetch 4

# Mid-range GPU (3060/4070) - balanced
python train_runpod_multitarget.py --run-id <id> --all --local \
    --threads 8 --numba-threads 8 --prefetch 2

# CPU fallback
python train_runpod_multitarget.py --run-id <id> --all --local \
    --cpu --threads 16
```

### Output Structure

```
models/<run_id>/
├── <item_id>/
│   └── model.cbm              # CatBoost model (108 targets)
├── training_summary.json      # Success/error counts, AUC metrics
└── config.json                # Training configuration
```

### Automation Guide

For automated daily training (see Issue #28), implement these steps:

#### 1. Cron Job Setup (05:00 EST daily)

```bash
# /etc/cron.d/gept-training
0 5 * * * ubuntu /home/ubuntu/gept/scripts/daily_training.sh >> /var/log/gept/training.log 2>&1
```

#### 2. Daily Training Script Template

```bash
#!/bin/bash
# daily_training.sh

set -e
export DB_PASS="${DB_PASS}"
export RUN_ID=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/home/ubuntu/gept/data"
MODEL_DIR="/home/ubuntu/gept/models"
PROD_DIR="/home/ubuntu/gept/production_models"

# Step 1: Prepare data
python prepare_runpod_data.py \
    --run-id $RUN_ID \
    --months 6 \
    --output-dir $DATA_DIR

# Step 2: Train models
python train_runpod_multitarget.py \
    --run-id $RUN_ID \
    --all \
    --local \
    --cache-dir $DATA_DIR \
    --output-dir $MODEL_DIR \
    --threads 8 \
    --prefetch 2

# Step 3: Validate training results
SUMMARY="$MODEL_DIR/$RUN_ID/training_summary.json"
ERRORS=$(jq '.errors' $SUMMARY)
AVG_AUC=$(jq '.average_auc' $SUMMARY)

if [ "$ERRORS" -gt 10 ]; then
    echo "ERROR: Too many training errors ($ERRORS)"
    exit 1
fi

if (( $(echo "$AVG_AUC < 0.65" | bc -l) )); then
    echo "WARNING: Low average AUC ($AVG_AUC)"
fi

# Step 4: Deploy to production (with lifecycle management)
# See Issue #28 for ACTIVE -> DEPRECATED -> SUNSET -> ARCHIVED flow
ln -sfn "$MODEL_DIR/$RUN_ID" "$PROD_DIR/latest"
echo "Training complete: $RUN_ID (AUC: $AVG_AUC)"
```

#### 3. Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DB_PASS` | Database password | (required, no default) |
| `OMP_NUM_THREADS` | OpenMP threads | `8` |
| `NUMBA_NUM_THREADS` | Numba threads | `8` |

#### 4. Monitoring

Check training success via:
- `training_summary.json` - Error counts and AUC metrics
- System logs - `/var/log/gept/training.log`
- Prometheus metrics - Model age, prediction latency

#### 5. Model Lifecycle (Issue #28)

```
ACTIVE (new model, accepting trades)
   ↓ (next day's training completes)
DEPRECATED (no new trades, existing trades continue)
   ↓ (48-hour grace period)
SUNSET (readonly, waiting for trades to close)
   ↓ (all trades closed)
ARCHIVED (removed from production)
```

### Requirements

```bash
pip install catboost numba pandas pyarrow scikit-learn tqdm google-cloud-storage
```

### GPU Setup (CUDA 12.4)

```bash
# Ubuntu/WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-4
```
