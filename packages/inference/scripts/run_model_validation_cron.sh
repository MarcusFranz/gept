#!/bin/bash
# GePT model validation cron wrapper
#
# Example crontab (daily at 04:30 UTC):
#   30 4 * * * /path/to/gept/packages/inference/scripts/run_model_validation_cron.sh >> /path/to/gept/packages/inference/logs/model_validation.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCK_FILE="/tmp/gept_model_validation.lock"
MAX_RUNTIME_SECONDS=1800

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] model validation already running, skipping"
    exit 0
fi
echo $$ >&9
trap "flock -u 9" EXIT

cd "$INFER_DIR" || exit 1

if [ -f "$INFER_DIR/.env" ]; then
    set -a
    source "$INFER_DIR/.env"
    set +a
fi

if [ -f "$INFER_DIR/venv/bin/activate" ]; then
    source "$INFER_DIR/venv/bin/activate"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting model validation"
timeout "$MAX_RUNTIME_SECONDS" python src/validation/run_model_validation.py \
    --window-hours 168 \
    --max-rows 200000
exit_code=$?

if [ "$exit_code" -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: model validation timed out after ${MAX_RUNTIME_SECONDS}s"
elif [ "$exit_code" -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: model validation failed with exit code ${exit_code}"
    exit "$exit_code"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] model validation complete"
