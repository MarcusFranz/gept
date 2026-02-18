#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

SESSION="${SESSION:-gept3_prod_pipeline}"
PROFILE="${PROFILE:-$SCRIPT_DIR/production_profile_full.env}"
STAGE="${STAGE:-all}"
REPLACE_SESSION="${REPLACE_SESSION:-1}"
LOG_FILE="${LOG_FILE:-$ROOT/packages/inference/logs/${SESSION}.log}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux_not_found" >&2
  exit 1
fi

if [[ ! -f "$PROFILE" ]]; then
  echo "profile_missing path=$PROFILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE_SESSION" == "1" || "$REPLACE_SESSION" == "true" || "$REPLACE_SESSION" == "TRUE" ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "session_exists session=$SESSION" >&2
    exit 1
  fi
fi

CMD="cd '$ROOT' && source '$PROFILE' && STAGE='$STAGE' '$ROOT/packages/inference/scripts/gept3/run_gept3_production_pipeline.sh' 2>&1 | tee -a '$LOG_FILE'"

tmux new-session -d -s "$SESSION" "bash -lc \"$CMD\""

echo "started_session=$SESSION"
echo "log_file=$LOG_FILE"
echo "attach: tmux attach -t $SESSION"
echo "tail: tail -f $LOG_FILE"
