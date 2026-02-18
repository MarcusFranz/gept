#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-gept3_autopilot_overnight}"
SCRIPT="${SCRIPT:-$SCRIPT_DIR/autopilot_gept3_overnight.sh}"

pkill -TERM -f "$SCRIPT" >/dev/null 2>&1 || true
sleep 1
pgrep -f "$SCRIPT" | xargs -I{} kill -KILL {} 2>/dev/null || true
screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
screen -dmS "$SESSION" bash -lc "$SCRIPT"
sleep 1

echo "started autopilot session: ${SESSION}"
screen -ls | rg -n "${SESSION}" || true
