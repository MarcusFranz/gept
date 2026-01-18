#!/bin/bash
# GePT Inference Cron Script
# Runs every 5 minutes via crontab
#
# Crontab entry:
#   */5 * * * * /home/ubuntu/gept/current/scripts/run_inference_cron.sh >> /home/ubuntu/gept/logs/inference.log 2>&1
#
# Note: Uses 'current' symlink for versioned deployments with rollback support.
# The symlink points to the active release directory (e.g., releases/20260113_150000/).

set -euo pipefail

# Error handler for debugging
error_handler() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Script failed on line $1" >&2
    exit 1
}
trap 'error_handler $LINENO' ERR

GEPT_DIR="/home/ubuntu/gept"
CURRENT_DIR="$GEPT_DIR/current"
LOCK_FILE="/tmp/gept_inference.lock"
MAX_RUNTIME=240  # 4 minutes max before next cycle

# Atomic lock acquisition using flock
# Opens lock file on file descriptor 9, attempts non-blocking exclusive lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another instance is running, skipping"
    exit 0
fi

# Lock acquired - write PID for debugging/monitoring
echo $$ >&9

# Cleanup: release lock on exit (flock auto-releases on fd close)
trap "flock -u 9" EXIT

# Verify current symlink exists
if [ ! -L "$CURRENT_DIR" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Current symlink not found at $CURRENT_DIR"
    exit 1
fi

cd "$CURRENT_DIR" || exit 1

# Load environment variables from .env (DB_PASS, etc.)
set -a && source "$GEPT_DIR/.env" && set +a

# Activate virtual environment (shared across releases)
source "$GEPT_DIR/venv/bin/activate"

# Run inference with timeout
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting inference cycle (version: $(basename "$(readlink "$CURRENT_DIR")"))"
# Let predictor auto-discover latest model run in models/ directory
timeout $MAX_RUNTIME python run_inference.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Inference timed out after ${MAX_RUNTIME}s"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Inference failed with exit code $EXIT_CODE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Inference cycle complete"
