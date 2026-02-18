#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
INFER_DIR="$ROOT/packages/inference"

RUN_NAME="${RUN_NAME:-gept3_hybrid_cuda_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b2_ga4}"
TRAIN_SESSION_NAME="${TRAIN_SESSION_NAME:-gept3_train_cuda}"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_hybrid_cuda_scaleup.sh}"
TRAIN_LOG="${TRAIN_LOG:-$INFER_DIR/logs/${RUN_NAME}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-$INFER_DIR/logs/${RUN_NAME}_watchdog.log}"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
STALL_SECONDS="${STALL_SECONDS:-900}"
RESTART_COOLDOWN_SECONDS="${RESTART_COOLDOWN_SECONDS:-30}"

mkdir -p "$(dirname "$TRAIN_LOG")"

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$WATCHDOG_LOG"
}

get_train_pid() {
  pgrep -f "train_gept3_hybrid.py.*${RUN_NAME}" | head -n1 || true
}

file_mtime_epoch() {
  local path="$1"
  if stat -c %Y "$path" >/dev/null 2>&1; then
    stat -c %Y "$path"
    return
  fi
  if stat -f %m "$path" >/dev/null 2>&1; then
    stat -f %m "$path"
    return
  fi
  date +%s
}

stop_training() {
  local reason="$1"
  local pid
  pid="$(get_train_pid)"
  if [[ -n "$pid" ]]; then
    log "stop_training reason=${reason} pid=${pid}"
    kill -TERM "$pid" >/dev/null 2>&1 || true
    sleep 8
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
  fi
  screen -S "$TRAIN_SESSION_NAME" -X quit >/dev/null 2>&1 || true
}

capture_clues() {
  if [[ -f "$TRAIN_LOG" ]]; then
    {
      echo "[clue] last_train_log_lines"
      tail -n 80 "$TRAIN_LOG"
    } >>"$WATCHDOG_LOG"
  fi

  if command -v dmesg >/dev/null 2>&1; then
    {
      echo "[clue] dmesg_tail"
      dmesg | tail -n 80
    } >>"$WATCHDOG_LOG" 2>/dev/null || true
  fi

  if command -v journalctl >/dev/null 2>&1; then
    {
      echo "[clue] journalctl_kernel_tail"
      journalctl -k -n 80 --no-pager
    } >>"$WATCHDOG_LOG" 2>/dev/null || true
  fi
}

start_training() {
  log "start_training session=${TRAIN_SESSION_NAME} script=${RUN_SCRIPT}"
  screen -S "$TRAIN_SESSION_NAME" -X quit >/dev/null 2>&1 || true
  screen -dmS "$TRAIN_SESSION_NAME" bash -lc "$RUN_SCRIPT"
  sleep 2

  local pid
  pid="$(get_train_pid)"
  if [[ -n "$pid" ]]; then
    log "training_started pid=${pid}"
  else
    log "training_start_pending_or_failed pid_not_found"
  fi
}

if ! command -v screen >/dev/null 2>&1; then
  echo "screen_missing install_screen_required" >&2
  exit 1
fi

log "watchdog_started run_name=${RUN_NAME} check_interval_s=${CHECK_INTERVAL_SECONDS} stall_s=${STALL_SECONDS}"

last_restart_ts=0
loop_count=0

while true; do
  loop_count="$((loop_count + 1))"
  now="$(date +%s)"
  pid="$(get_train_pid)"

  if [[ -z "$pid" ]]; then
    if (( now - last_restart_ts >= RESTART_COOLDOWN_SECONDS )); then
      log "detected_missing_process"
      capture_clues
      start_training
      last_restart_ts="$now"
    fi
    sleep "$CHECK_INTERVAL_SECONDS"
    continue
  fi

  if [[ -f "$TRAIN_LOG" ]]; then
    log_mtime="$(file_mtime_epoch "$TRAIN_LOG")"
    log_age="$((now - log_mtime))"

    if (( loop_count % 2 == 0 )); then
      log "heartbeat pid=${pid} log_age_s=${log_age}"
    fi

    if (( log_age > STALL_SECONDS )); then
      log "detected_stall pid=${pid} log_age_s=${log_age}"
      capture_clues
      stop_training "stall_log_age_${log_age}s"
      start_training
      last_restart_ts="$now"
      sleep "$CHECK_INTERVAL_SECONDS"
      continue
    fi
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
