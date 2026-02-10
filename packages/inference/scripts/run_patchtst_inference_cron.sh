#!/bin/bash
# GePT PatchTST Inference Cron Script
# Runs every 5 minutes via crontab
#
# Crontab entry:
#   */5 * * * * /home/ubuntu/gept/scripts/run_patchtst_inference_cron.sh >> /home/ubuntu/gept/logs/patchtst_inference.log 2>&1

set -euo pipefail

GEPT_DIR="/home/ubuntu/gept"
LOCK_FILE="/tmp/gept_patchtst_inference.lock"
MAX_RUNTIME=270  # 4.5 minutes max before next cycle

# Atomic lock acquisition
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another PatchTST instance is running, skipping"
    exit 0
fi
echo $$ >&9
trap "flock -u 9" EXIT

cd "$GEPT_DIR" || exit 1

# Load environment variables
set -a && source "$GEPT_DIR/.env" && set +a

# Activate virtual environment
source "$GEPT_DIR/venv/bin/activate"

# Set Python path for pipeline imports
export PYTHONPATH="$GEPT_DIR/src"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PatchTST inference cycle"
timeout $MAX_RUNTIME python src/pipeline/run_patchtst_inference.py \
    --model-path models/patchtst/best_model.pt \
    --min-volume 2000

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: PatchTST inference timed out after ${MAX_RUNTIME}s"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: PatchTST inference failed with exit code $EXIT_CODE"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PatchTST inference cycle complete"
