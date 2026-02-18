#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
INFER_DIR="$ROOT/packages/inference"

PYTHON_BIN="${PYTHON_BIN:-$INFER_DIR/.venv312/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

RUN_NAME="${RUN_NAME:-gept3_hybrid_cuda_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b2_ga4}"
DATASET="${DATASET:-$INFER_DIR/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz}"
LOG_DIR="${LOG_DIR:-$INFER_DIR/logs}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"
OUT_DIR="${OUT_DIR:-$INFER_DIR/models/gept3}"

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
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
SEED="${SEED:-1337}"
DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-fp16}"
LOSS_W_SELL="${LOSS_W_SELL:-1.0}"
LOSS_W_BUY="${LOSS_W_BUY:-0.35}"
LOSS_W_PRICE="${LOSS_W_PRICE:-0.20}"
LOSS_W_REGIME="${LOSS_W_REGIME:-0.10}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-20}"
GC_EVERY_STEPS="${GC_EVERY_STEPS:-0}"
MPS_EMPTY_CACHE_EVERY_STEPS="${MPS_EMPTY_CACHE_EVERY_STEPS:-0}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ ! -f "$DATASET" ]]; then
  echo "dataset_missing path=$DATASET" >&2
  exit 1
fi

export PYTHONPATH="$INFER_DIR:$INFER_DIR/src:${PYTHONPATH:-}"

run_id="$(date +%Y%m%d_%H%M%S)"

cmd=(
  "$PYTHON_BIN" -u "$INFER_DIR/scripts/gept3/train_gept3_hybrid.py"
  --dataset "$DATASET"
  --out-dir "$OUT_DIR"
  --run-name "$RUN_NAME"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --eval-max-batches "$EVAL_MAX_BATCHES"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --d-model "$D_MODEL"
  --n-heads "$N_HEADS"
  --n-layers "$N_LAYERS"
  --temporal-n-heads "$TEMPORAL_N_HEADS"
  --temporal-n-layers "$TEMPORAL_N_LAYERS"
  --ff-mult "$FF_MULT"
  --dropout "$DROPOUT"
  --n-price-quantiles "$N_PRICE_QUANTILES"
  --val-days "$VAL_DAYS"
  --test-days "$TEST_DAYS"
  --seed "$SEED"
  --device "$DEVICE"
  --precision "$PRECISION"
  --enable-all-heads
  --loss-w-sell "$LOSS_W_SELL"
  --loss-w-buy "$LOSS_W_BUY"
  --loss-w-price "$LOSS_W_PRICE"
  --loss-w-regime "$LOSS_W_REGIME"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --gc-every-steps "$GC_EVERY_STEPS"
  --mps-empty-cache-every-steps "$MPS_EMPTY_CACHE_EVERY_STEPS"
  --checkpoint-every-epochs "$CHECKPOINT_EVERY_EPOCHS"
  --log-every-batches "$LOG_EVERY_BATCHES"
)

if [[ "$AUTO_RESUME" == "1" || "$AUTO_RESUME" == "true" || "$AUTO_RESUME" == "TRUE" ]]; then
  cmd+=(--auto-resume)
fi
if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$RESUME_FROM")
fi

rc=0
{
  echo "[$(date)] starting cuda hybrid run run_id=${run_id}"
  echo "[$(date)] root=${ROOT}"
  echo "[$(date)] run_name=${RUN_NAME} dataset=${DATASET} out_dir=${OUT_DIR} device=${DEVICE} precision=${PRECISION}"
  echo "[$(date)] batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} epochs=${EPOCHS} auto_resume=${AUTO_RESUME} checkpoint_every_epochs=${CHECKPOINT_EVERY_EPOCHS}"
  printf '[%s] cmd=' "$(date)"
  printf '%q ' "${cmd[@]}"
  echo

  if ! "${cmd[@]}"; then
    rc=$?
  fi

  echo "[$(date)] finished cuda hybrid run run_id=${run_id} rc=${rc}"
} >>"$LOG" 2>&1

exit "$rc"
