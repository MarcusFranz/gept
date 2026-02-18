#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
INFER_DIR="$ROOT/packages/inference"

RUN_NAME="${RUN_NAME:-gept3_hybrid_cuda_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b2_ga4}"
TRAIN_SESSION_NAME="${TRAIN_SESSION_NAME:-gept3_train_cuda}"
WATCH_SESSION="${WATCH_SESSION:-gept3_watchdog_cuda}"
TRAIN_LOG="${TRAIN_LOG:-$INFER_DIR/logs/${RUN_NAME}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-$INFER_DIR/logs/${RUN_NAME}_watchdog.log}"

WATCH_SCRIPT="$SCRIPT_DIR/watch_hybrid_train.sh"
RUN_SCRIPT="${RUN_SCRIPT:-$SCRIPT_DIR/run_hybrid_cuda_scaleup.sh}"

RUN_NAME="$RUN_NAME" \
TRAIN_SESSION_NAME="$TRAIN_SESSION_NAME" \
WATCH_SESSION="$WATCH_SESSION" \
TRAIN_LOG="$TRAIN_LOG" \
WATCHDOG_LOG="$WATCHDOG_LOG" \
RUN_SCRIPT="$RUN_SCRIPT" \
WATCH_SCRIPT="$WATCH_SCRIPT" \
"$SCRIPT_DIR/start_hybrid_watchdog.sh"
