#!/usr/bin/env bash
set -uo pipefail
cd /Users/marcus/code/gept

LOG=/Users/marcus/code/gept/packages/inference/logs/gept3_hybrid_m4_full_s128_t48_n3k_e12_b2_ga4.log
mkdir -p /Users/marcus/code/gept/packages/inference/logs

rc=0
run_id="$(date +%Y%m%d_%H%M%S)"
{
  echo "[$(date)] starting hybrid full run run_id=${run_id}"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.70
  export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.60
  if ! /Users/marcus/code/gept/packages/inference/.venv312/bin/python -u /Users/marcus/code/gept/packages/inference/scripts/gept3/train_gept3_hybrid.py \
    --dataset /Users/marcus/code/gept/packages/inference/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n3000_min24_multitask_h1-24.npz \
    --out-dir /Users/marcus/code/gept/packages/inference/models/gept3 \
    --run-name gept3_hybrid_m4_full_s128_t48_n3k_e12_b2_ga4 \
    --epochs 12 \
    --batch-size 2 \
    --eval-batch-size 2 \
    --eval-max-batches 200 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 4 \
    --temporal-n-heads 4 \
    --temporal-n-layers 2 \
    --ff-mult 4.0 \
    --dropout 0.10 \
    --n-price-quantiles 5 \
    --val-days 7 \
    --test-days 7 \
    --device mps \
    --precision fp32 \
    --enable-all-heads \
    --loss-w-sell 1.0 \
    --loss-w-buy 0.35 \
    --loss-w-price 0.20 \
    --loss-w-regime 0.10 \
    --grad-accum-steps 4 \
    --gc-every-steps 20 \
    --mps-empty-cache-every-steps 20 \
    --log-every-batches 10; then
    rc=$?
  fi
  echo "[$(date)] finished run_id=${run_id} rc=$rc"
} >> "$LOG" 2>&1

exit "$rc"
