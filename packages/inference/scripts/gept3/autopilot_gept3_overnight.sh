#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/marcus/code/gept"
SCRIPTS_DIR="$ROOT/packages/inference/scripts/gept3"
DATA_DIR="$ROOT/packages/inference/data"
LOG_DIR="$ROOT/packages/inference/logs"
MODELS_DIR="$ROOT/packages/inference/models/gept3"
ROW_DATASET="$DATA_DIR/gept3_row_365d_preditems_allmodels_every1_cap12000_multitask_h1-24.npz"

AUTOPILOT_LOG="${AUTOPILOT_LOG:-$LOG_DIR/gept3_autopilot_overnight.log}"
STATE_FILE="${STATE_FILE:-$LOG_DIR/gept3_autopilot_overnight.state}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"

mkdir -p "$LOG_DIR"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$AUTOPILOT_LOG"
}

# Ordered rung pipeline: run_name|snapshot_size|seq_len|max_snapshots|d_model|n_heads|n_layers|temporal_n_heads|temporal_n_layers|grad_accum|epochs|lr
RUNS=(
  "gept3_hybrid_m4_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b1_ga8|128|48|4500|384|12|6|6|3|8|12|2.5e-4"
  "gept3_hybrid_m4_scaleup_s160_t48_n3200_d384_l6_tl3_e10_b1_ga8|160|48|3200|384|12|6|6|3|8|10|2.5e-4"
  "gept3_hybrid_m4_scaleup_s160_t64_n2400_d384_l6_tl3_e8_b1_ga10|160|64|2400|384|12|6|6|3|10|8|2.0e-4"
)

load_state() {
  if [[ -f "$STATE_FILE" ]]; then
    source "$STATE_FILE"
  fi
  CURRENT_INDEX="${CURRENT_INDEX:-0}"
}

save_state() {
  cat >"$STATE_FILE" <<EOF
CURRENT_INDEX=$CURRENT_INDEX
EOF
}

dataset_path_for() {
  local snapshot_size="$1"
  local seq_len="$2"
  local max_snapshots="$3"
  echo "$DATA_DIR/gept3_hybrid_snap_365d_preditems_allmodels_s${snapshot_size}_t${seq_len}_n${max_snapshots}_min24_multitask_h1-24.npz"
}

ensure_dataset() {
  local snapshot_size="$1"
  local seq_len="$2"
  local max_snapshots="$3"
  local ds_path
  ds_path="$(dataset_path_for "$snapshot_size" "$seq_len" "$max_snapshots")"
  if [[ -f "$ds_path" ]]; then
    log "dataset_ready path=$ds_path"
    echo "$ds_path"
    return 0
  fi
  log "dataset_build_start path=$ds_path"
  /Users/marcus/code/gept/packages/inference/.venv312/bin/python \
    "$SCRIPTS_DIR/build_hybrid_snapshot_dataset.py" \
    --row-dataset "$ROW_DATASET" \
    --out "$ds_path" \
    --seq-len "$seq_len" \
    --snapshot-size "$snapshot_size" \
    --min-items-per-snapshot 24 \
    --max-snapshots "$max_snapshots" \
    --select-mode uniform \
    --row-pick-mode random \
    --seed 1337 >>"$AUTOPILOT_LOG" 2>&1
  log "dataset_build_done path=$ds_path"
  echo "$ds_path"
}

run_active() {
  local run_name="$1"
  pgrep -f "train_gept3_hybrid.py.*${run_name}" >/dev/null 2>&1
}

launch_rung() {
  local run_name="$1"
  local snapshot_size="$2"
  local seq_len="$3"
  local max_snapshots="$4"
  local d_model="$5"
  local n_heads="$6"
  local n_layers="$7"
  local temporal_n_heads="$8"
  local temporal_n_layers="$9"
  local grad_accum="${10}"
  local epochs="${11}"
  local lr="${12}"

  local ds_path
  ds_path="$(ensure_dataset "$snapshot_size" "$seq_len" "$max_snapshots")"

  local train_session="gept3_train_${run_name}"
  local watchdog_session="gept3_watchdog_${run_name}"
  local train_log="$LOG_DIR/${run_name}.log"
  local watchdog_log="$LOG_DIR/${run_name}_watchdog.log"

  log "launch_start run=$run_name train_session=$train_session watchdog_session=$watchdog_session"
  RUN_NAME="$run_name" \
  TRAIN_SESSION_NAME="$train_session" \
  RUN_SCRIPT="$SCRIPTS_DIR/run_hybrid_m4_scaleup.sh" \
  TRAIN_LOG="$train_log" \
  WATCHDOG_LOG="$watchdog_log" \
  WATCH_SESSION="$watchdog_session" \
  WATCH_SCRIPT="$SCRIPTS_DIR/watch_hybrid_m4.sh" \
  DATASET="$ds_path" \
  D_MODEL="$d_model" \
  N_HEADS="$n_heads" \
  N_LAYERS="$n_layers" \
  TEMPORAL_N_HEADS="$temporal_n_heads" \
  TEMPORAL_N_LAYERS="$temporal_n_layers" \
  BATCH_SIZE=1 \
  EVAL_BATCH_SIZE=1 \
  EVAL_MAX_BATCHES=250 \
  GRAD_ACCUM_STEPS="$grad_accum" \
  EPOCHS="$epochs" \
  LR="$lr" \
  GC_EVERY_STEPS=12 \
  MPS_EMPTY_CACHE_EVERY_STEPS=12 \
  LOG_EVERY_BATCHES=10 \
  "$SCRIPTS_DIR/start_hybrid_watchdog.sh" >>"$AUTOPILOT_LOG" 2>&1
  log "launch_done run=$run_name"
}

run_succeeded() {
  local run_name="$1"
  local log_path="$LOG_DIR/${run_name}.log"
  [[ -f "$log_path" ]] || return 1
  rg -n "finished run_id=.* rc=0" "$log_path" >/dev/null 2>&1
}

report_exists() {
  local run_name="$1"
  [[ -f "$MODELS_DIR/$run_name/train_report.json" ]]
}

summarize_report() {
  local run_name="$1"
  local report="$MODELS_DIR/$run_name/train_report.json"
  if [[ ! -f "$report" ]]; then
    log "report_missing run=$run_name"
    return 0
  fi
  /Users/marcus/code/gept/packages/inference/.venv312/bin/python - <<PY >>"$AUTOPILOT_LOG" 2>&1
import json
from pathlib import Path
p=Path("$report")
r=json.loads(p.read_text())
sell=r.get("test_metrics",{}).get("sell",{})
print(f"[report] run={p.parent.name} test_sell_brier={sell.get('brier')} test_sell_logloss={sell.get('logloss')} test_emp_pos_at_080={sell.get('empirical_pos_given_pred_ge_080')}")
PY
}

load_state
log "autopilot_started current_index=$CURRENT_INDEX total_rungs=${#RUNS[@]}"

while true; do
  if (( CURRENT_INDEX >= ${#RUNS[@]} )); then
    log "all_rungs_complete"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  IFS='|' read -r run_name snapshot_size seq_len max_snapshots d_model n_heads n_layers temporal_n_heads temporal_n_layers grad_accum epochs lr <<<"${RUNS[$CURRENT_INDEX]}"

  if run_active "$run_name"; then
    log "run_active run=$run_name"
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  if run_succeeded "$run_name" && report_exists "$run_name"; then
    log "run_complete run=$run_name"
    summarize_report "$run_name"
    CURRENT_INDEX=$((CURRENT_INDEX + 1))
    save_state
    sleep 2
    continue
  fi

  # If not active and not completed, launch/relaunch current rung.
  launch_rung "$run_name" "$snapshot_size" "$seq_len" "$max_snapshots" "$d_model" "$n_heads" "$n_layers" "$temporal_n_heads" "$temporal_n_layers" "$grad_accum" "$epochs" "$lr"
  sleep "$CHECK_INTERVAL_SECONDS"
done
