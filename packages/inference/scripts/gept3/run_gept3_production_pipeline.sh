#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
INFER_DIR="$ROOT/packages/inference"

PYTHON_BIN="${PYTHON_BIN:-$INFER_DIR/.venv312/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

STAGE="${STAGE:-all}" # all | row | hybrid | train
DEVICE="${DEVICE:-cuda}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

# Preflight guards (production defaults).
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-250}"
MIN_RAM_GB="${MIN_RAM_GB:-96}"

# Row dataset (full-history intent).
LOOKBACK_DAYS="${LOOKBACK_DAYS:-365}"
ITEM_SOURCE="${ITEM_SOURCE:-predictions}"
PREDICTIONS_MODEL_ID="${PREDICTIONS_MODEL_ID:-}"
MAX_ITEMS="${MAX_ITEMS:-0}"
MIN_VOLUME="${MIN_VOLUME:-500}"
SAMPLE_EVERY="${SAMPLE_EVERY:-1}"
MAX_ROWS_PER_ITEM="${MAX_ROWS_PER_ITEM:-0}" # 0 = uncapped
ROW_SEED="${ROW_SEED:-1337}"
ROW_LOG_EVERY="${ROW_LOG_EVERY:-25}"
REGIME_HOURS="${REGIME_HOURS:-6}"
REGIME_RET_THRESHOLD="${REGIME_RET_THRESHOLD:-0.010}"
REGIME_DRAWDOWN_THRESHOLD="${REGIME_DRAWDOWN_THRESHOLD:-0.015}"
PRICE_DELTA_CLIP_ABS="${PRICE_DELTA_CLIP_ABS:-0.30}"

ROW_DATASET="${ROW_DATASET:-$INFER_DIR/data/gept3_row_365d_preditems_allmodels_every1_uncapped_multitask_h1-24.npz}"

# Hybrid snapshot dataset.
SEQ_LEN="${SEQ_LEN:-48}"
SNAPSHOT_SIZE="${SNAPSHOT_SIZE:-128}"
MIN_ITEMS_PER_SNAPSHOT="${MIN_ITEMS_PER_SNAPSHOT:-24}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-18000}" # 0 = uncapped; can exceed host memory.
SELECT_MODE="${SELECT_MODE:-uniform}"
ROW_PICK_MODE="${ROW_PICK_MODE:-random}"
HYBRID_SEED="${HYBRID_SEED:-1337}"

HYBRID_DATASET="${HYBRID_DATASET:-$INFER_DIR/data/gept3_hybrid_snap_365d_preditems_allmodels_s${SNAPSHOT_SIZE}_t${SEQ_LEN}_n${MAX_SNAPSHOTS}_min${MIN_ITEMS_PER_SNAPSHOT}_multitask_h1-24.npz}"

# Training params.
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

RUN_NAME="${RUN_NAME:-gept3_hybrid_prod_s${SNAPSHOT_SIZE}_t${SEQ_LEN}_n${MAX_SNAPSHOTS}_d${D_MODEL}_l${N_LAYERS}_tl${TEMPORAL_N_LAYERS}_e${EPOCHS}_b${BATCH_SIZE}_ga${GRAD_ACCUM_STEPS}}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
    log "dry_run: $*"
    return 0
  fi
  log "run: $*"
  "$@"
}

check_stage() {
  case "$STAGE" in
    all|row|hybrid|train) ;;
    *)
      echo "invalid STAGE=$STAGE (expected: all|row|hybrid|train)" >&2
      exit 1
      ;;
  esac
}

preflight() {
  if [[ "$SKIP_PREFLIGHT" == "1" || "$SKIP_PREFLIGHT" == "true" || "$SKIP_PREFLIGHT" == "TRUE" ]]; then
    log "preflight skipped"
    return 0
  fi

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "python_missing path=$PYTHON_BIN" >&2
    exit 1
  fi

  mkdir -p "$INFER_DIR/data" "$INFER_DIR/models" "$INFER_DIR/logs"

  # Disk check (filesystem hosting inference dir).
  avail_kb="$(df -Pk "$INFER_DIR" | awk 'NR==2 {print $4}')"
  need_kb="$((MIN_FREE_DISK_GB * 1024 * 1024))"
  if [[ "$avail_kb" -lt "$need_kb" ]]; then
    echo "insufficient_disk_gb available=$((avail_kb/1024/1024)) required=$MIN_FREE_DISK_GB" >&2
    exit 1
  fi
  log "disk_ok available_gb=$((avail_kb/1024/1024)) required_gb=$MIN_FREE_DISK_GB"

  # Linux RAM check.
  if [[ -f /proc/meminfo ]]; then
    mem_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo | head -n1)"
    need_mem_kb="$((MIN_RAM_GB * 1024 * 1024))"
    if [[ -n "$mem_kb" && "$mem_kb" -lt "$need_mem_kb" ]]; then
      echo "insufficient_ram_gb available=$((mem_kb/1024/1024)) required=$MIN_RAM_GB" >&2
      exit 1
    fi
    log "ram_ok available_gb=$((mem_kb/1024/1024)) required_gb=$MIN_RAM_GB"
  else
    log "ram_check_skipped_non_linux"
  fi

  # GPU check for CUDA runs.
  if [[ "$DEVICE" == "cuda" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "cuda_requested_but_nvidia_smi_missing" >&2
      exit 1
    fi
    log "gpu_ok"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  fi

  if [[ -z "${DB_PASS:-}" ]]; then
    echo "missing DB_PASS for dataset build" >&2
    exit 1
  fi

  # DB reachability + item universe size.
  PYTHONPATH="$INFER_DIR:$INFER_DIR/src:${PYTHONPATH:-}" "$PYTHON_BIN" - <<PY
import os
from src.db_utils import get_simple_connection
from src.fillprob.dataset import get_items_from_latest_predictions, get_tradeable_items

item_source = os.getenv('ITEM_SOURCE', 'predictions')
pred_model_id = os.getenv('PREDICTIONS_MODEL_ID', '').strip() or None
max_items = int(os.getenv('MAX_ITEMS', '0') or '0')
min_volume = int(os.getenv('MIN_VOLUME', '500') or '500')

conn = get_simple_connection()
try:
    if item_source == 'predictions':
        ids = get_items_from_latest_predictions(conn, model_id=pred_model_id, limit=(max_items or None))
    else:
        ids = get_tradeable_items(conn, min_volume=min_volume, lookback_days=7, limit=(max_items or None))
    print(f"db_ok item_source={item_source} n_items={len(ids)}")
finally:
    conn.close()
PY
}

inspect_npz() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    log "npz_missing path=$path"
    return 0
  fi
  "$PYTHON_BIN" - <<PY
import numpy as np
import zipfile
from numpy.lib import format as npfmt
from pathlib import Path
p = Path(r"$path")
arr = np.load(p, allow_pickle=True)
print(f"npz_summary path={p} size_gb={p.stat().st_size/1024/1024/1024:.2f} keys={list(arr.files)}")

def _read_npy_header(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as fh:
        version = npfmt.read_magic(fh)
        if version == (1, 0):
            shape, _, dtype = npfmt.read_array_header_1_0(fh)
        elif version == (2, 0):
            shape, _, dtype = npfmt.read_array_header_2_0(fh)
        elif version == (3, 0):
            shape, _, dtype = npfmt.read_array_header_3_0(fh)
        else:
            raise RuntimeError(f"unsupported_npy_version={version}")
    return shape, np.dtype(dtype)

meta = {}
try:
    with zipfile.ZipFile(p, "r") as zf:
        for key in arr.files:
            member = f"{key}.npy"
            if member in zf.namelist():
                try:
                    shape, dtype = _read_npy_header(zf, member)
                    meta[key] = (shape, dtype)
                except Exception as exc:
                    meta[key] = (None, f"unknown ({exc})")
except Exception:
    # Fallback to keys-only summary when zip metadata isn't available.
    meta = {}

for key in ('x','x_seq','item_id','ts','y_sell','y_buy','y_price_high','y_price_low','y_regime'):
    if key in arr.files:
        if key in meta:
            shape, dtype = meta[key]
            print(f"shape[{key}]={shape} dtype={dtype}")
        else:
            print(f"shape[{key}]=unknown dtype=unknown")
arr.close()
PY
}

build_row() {
  log "stage=row out=$ROW_DATASET lookback_days=$LOOKBACK_DAYS sample_every=$SAMPLE_EVERY max_rows_per_item=$MAX_ROWS_PER_ITEM"
  run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/build_row_dataset.py" \
    --out "$ROW_DATASET" \
    --lookback-days "$LOOKBACK_DAYS" \
    --item-source "$ITEM_SOURCE" \
    --predictions-model-id "$PREDICTIONS_MODEL_ID" \
    --max-items "$MAX_ITEMS" \
    --min-volume "$MIN_VOLUME" \
    --sample-every "$SAMPLE_EVERY" \
    --max-rows-per-item "$MAX_ROWS_PER_ITEM" \
    --seed "$ROW_SEED" \
    --log-every "$ROW_LOG_EVERY" \
    --regime-hours "$REGIME_HOURS" \
    --regime-ret-threshold "$REGIME_RET_THRESHOLD" \
    --regime-drawdown-threshold "$REGIME_DRAWDOWN_THRESHOLD" \
    --price-delta-clip-abs "$PRICE_DELTA_CLIP_ABS"
  inspect_npz "$ROW_DATASET"
}

build_hybrid() {
  log "stage=hybrid out=$HYBRID_DATASET seq_len=$SEQ_LEN snapshot_size=$SNAPSHOT_SIZE max_snapshots=$MAX_SNAPSHOTS"
  if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" && ! -f "$ROW_DATASET" ]]; then
    echo "row_dataset_missing path=$ROW_DATASET" >&2
    exit 1
  fi
  run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/build_hybrid_snapshot_dataset.py" \
    --row-dataset "$ROW_DATASET" \
    --out "$HYBRID_DATASET" \
    --seq-len "$SEQ_LEN" \
    --snapshot-size "$SNAPSHOT_SIZE" \
    --min-items-per-snapshot "$MIN_ITEMS_PER_SNAPSHOT" \
    --max-snapshots "$MAX_SNAPSHOTS" \
    --select-mode "$SELECT_MODE" \
    --row-pick-mode "$ROW_PICK_MODE" \
    --seed "$HYBRID_SEED"
  inspect_npz "$HYBRID_DATASET"
}

train_model() {
  log "stage=train run_name=$RUN_NAME dataset=$HYBRID_DATASET"
  if [[ "$DRY_RUN" != "1" && "$DRY_RUN" != "true" && "$DRY_RUN" != "TRUE" && ! -f "$HYBRID_DATASET" ]]; then
    echo "hybrid_dataset_missing path=$HYBRID_DATASET" >&2
    exit 1
  fi
  run_cmd env \
    ROOT="$ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_NAME="$RUN_NAME" \
    DATASET="$HYBRID_DATASET" \
    DEVICE="$DEVICE" \
    PRECISION="$PRECISION" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" \
    LR="$LR" \
    WEIGHT_DECAY="$WEIGHT_DECAY" \
    D_MODEL="$D_MODEL" \
    N_HEADS="$N_HEADS" \
    N_LAYERS="$N_LAYERS" \
    TEMPORAL_N_HEADS="$TEMPORAL_N_HEADS" \
    TEMPORAL_N_LAYERS="$TEMPORAL_N_LAYERS" \
    FF_MULT="$FF_MULT" \
    DROPOUT="$DROPOUT" \
    N_PRICE_QUANTILES="$N_PRICE_QUANTILES" \
    VAL_DAYS="$VAL_DAYS" \
    TEST_DAYS="$TEST_DAYS" \
    SEED="$SEED" \
    LOSS_W_SELL="$LOSS_W_SELL" \
    LOSS_W_BUY="$LOSS_W_BUY" \
    LOSS_W_PRICE="$LOSS_W_PRICE" \
    LOSS_W_REGIME="$LOSS_W_REGIME" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    LOG_EVERY_BATCHES="$LOG_EVERY_BATCHES" \
    GC_EVERY_STEPS="$GC_EVERY_STEPS" \
    MPS_EMPTY_CACHE_EVERY_STEPS="$MPS_EMPTY_CACHE_EVERY_STEPS" \
    CHECKPOINT_EVERY_EPOCHS="$CHECKPOINT_EVERY_EPOCHS" \
    AUTO_RESUME="$AUTO_RESUME" \
    RESUME_FROM="$RESUME_FROM" \
    "$SCRIPT_DIR/run_hybrid_cuda_scaleup.sh"
}

main() {
  check_stage
  log "pipeline_start stage=$STAGE root=$ROOT"
  preflight

  case "$STAGE" in
    row)
      build_row
      ;;
    hybrid)
      build_hybrid
      ;;
    train)
      train_model
      ;;
    all)
      build_row
      build_hybrid
      train_model
      ;;
  esac

  log "pipeline_done stage=$STAGE"
}

main "$@"
