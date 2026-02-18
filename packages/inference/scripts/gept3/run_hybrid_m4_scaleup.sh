#!/usr/bin/env bash
set -uo pipefail
cd /Users/marcus/code/gept

RUN_NAME="${RUN_NAME:-gept3_hybrid_m4_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b1_ga8}"
DATASET="${DATASET:-/Users/marcus/code/gept/packages/inference/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz}"
LOG="${LOG:-/Users/marcus/code/gept/packages/inference/logs/${RUN_NAME}.log}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-250}"
LR="${LR:-2.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
D_MODEL="${D_MODEL:-384}"
N_HEADS="${N_HEADS:-12}"
N_LAYERS="${N_LAYERS:-6}"
TEMPORAL_N_HEADS="${TEMPORAL_N_HEADS:-6}"
TEMPORAL_N_LAYERS="${TEMPORAL_N_LAYERS:-3}"
FF_MULT="${FF_MULT:-4.0}"
DROPOUT="${DROPOUT:-0.10}"
N_PRICE_QUANTILES="${N_PRICE_QUANTILES:-5}"
VAL_DAYS="${VAL_DAYS:-7}"
TEST_DAYS="${TEST_DAYS:-7}"
DEVICE="${DEVICE:-mps}"
PRECISION="${PRECISION:-fp32}"
LOSS_W_SELL="${LOSS_W_SELL:-1.0}"
LOSS_W_BUY="${LOSS_W_BUY:-0.35}"
LOSS_W_PRICE="${LOSS_W_PRICE:-0.20}"
LOSS_W_REGIME="${LOSS_W_REGIME:-0.10}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
GC_EVERY_STEPS="${GC_EVERY_STEPS:-12}"
MPS_EMPTY_CACHE_EVERY_STEPS="${MPS_EMPTY_CACHE_EVERY_STEPS:-12}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-10}"

mkdir -p /Users/marcus/code/gept/packages/inference/logs

rc=0
run_id="$(date +%Y%m%d_%H%M%S)"
{
  echo "[$(date)] starting hybrid full run run_id=${run_id}"
  echo "[$(date)] config run_name=${RUN_NAME} dataset=${DATASET} epochs=${EPOCHS} d_model=${D_MODEL} n_layers=${N_LAYERS} temporal_n_layers=${TEMPORAL_N_LAYERS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS}"
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.68
  export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.58
  if ! /Users/marcus/code/gept/packages/inference/.venv312/bin/python -u /Users/marcus/code/gept/packages/inference/scripts/gept3/train_gept3_hybrid.py \
    --dataset "$DATASET" \
    --out-dir /Users/marcus/code/gept/packages/inference/models/gept3 \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --eval-max-batches "$EVAL_MAX_BATCHES" \
    --lr "$LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --d-model "$D_MODEL" \
    --n-heads "$N_HEADS" \
    --n-layers "$N_LAYERS" \
    --temporal-n-heads "$TEMPORAL_N_HEADS" \
    --temporal-n-layers "$TEMPORAL_N_LAYERS" \
    --ff-mult "$FF_MULT" \
    --dropout "$DROPOUT" \
    --n-price-quantiles "$N_PRICE_QUANTILES" \
    --val-days "$VAL_DAYS" \
    --test-days "$TEST_DAYS" \
    --device "$DEVICE" \
    --precision "$PRECISION" \
    --enable-all-heads \
    --loss-w-sell "$LOSS_W_SELL" \
    --loss-w-buy "$LOSS_W_BUY" \
    --loss-w-price "$LOSS_W_PRICE" \
    --loss-w-regime "$LOSS_W_REGIME" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --gc-every-steps "$GC_EVERY_STEPS" \
    --mps-empty-cache-every-steps "$MPS_EMPTY_CACHE_EVERY_STEPS" \
    --log-every-batches "$LOG_EVERY_BATCHES"; then
    rc=$?
  fi
  echo "[$(date)] finished run_id=${run_id} rc=$rc"
} >> "$LOG" 2>&1

exit "$rc"
