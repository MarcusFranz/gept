#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-gept3_hybrid_m4_full_s128_t48_n3k_e12_b2_ga4}"
TRAIN_SESSION_NAME="${TRAIN_SESSION_NAME:-gept3_train_m4}"
RUN_SCRIPT="${RUN_SCRIPT:-/Users/marcus/code/gept/packages/inference/scripts/gept3/run_hybrid_m4_full.sh}"
TRAIN_LOG="${TRAIN_LOG:-/Users/marcus/code/gept/packages/inference/logs/${RUN_NAME}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-/Users/marcus/code/gept/packages/inference/logs/${RUN_NAME}_watchdog.log}"

CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-30}"
STALL_SECONDS="${STALL_SECONDS:-900}"
RESTART_COOLDOWN_SECONDS="${RESTART_COOLDOWN_SECONDS:-20}"

mkdir -p /Users/marcus/code/gept/packages/inference/logs

log() {
  local msg="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$WATCHDOG_LOG"
}

get_train_pid() {
  pgrep -f "train_gept3_hybrid.py.*${RUN_NAME}" | head -n1 || true
}

stop_training() {
  local reason="$1"
  local pid
  pid="$(get_train_pid)"
  if [[ -n "$pid" ]]; then
    log "stop_training reason=${reason} pid=${pid}"
    kill -TERM "$pid" || true
    sleep 8
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" || true
    fi
  fi
  screen -S "$TRAIN_SESSION_NAME" -X quit >/dev/null 2>&1 || true
}

capture_clues() {
  if [[ -f "$TRAIN_LOG" ]]; then
    {
      echo "[clue] last_train_log_lines"
      tail -n 60 "$TRAIN_LOG"
    } >> "$WATCHDOG_LOG"
  fi
  {
    echo "[clue] recent_system_log"
    /usr/bin/log show --style compact --last 3m \
      --predicate '(process == "kernel") || (process == "launchd") || (process == "runningboardd") || (process == "Python")' \
      | rg -i 'jetsam|memorystatus|killed|python' | tail -n 80 || true
  } >> "$WATCHDOG_LOG"
}

start_training() {
  log "start_training session=${TRAIN_SESSION_NAME}"
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

log "watchdog_started check_interval_s=${CHECK_INTERVAL_SECONDS} stall_s=${STALL_SECONDS}"

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
    log_mtime="$(stat -f %m "$TRAIN_LOG" 2>/dev/null || echo "$now")"
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
