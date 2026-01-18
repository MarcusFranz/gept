#!/bin/bash
# =============================================================================
# GePT Inference - Ampere (Fallback)
# =============================================================================
# Fallback inference on Ampere - only runs if Hydra hasn't updated predictions.
# Checks prediction freshness before running to avoid duplicate work.
#
# Crontab entry (on Ampere):
#   */5 * * * * /home/ubuntu/gept/scripts/run_inference_fallback.sh >> /home/ubuntu/gept/logs/inference.log 2>&1
# =============================================================================

set -euo pipefail

GEPT_DIR="/home/ubuntu/gept"
CURRENT_DIR="$GEPT_DIR/current"
LOCK_FILE="/tmp/gept_inference.lock"
MAX_RUNTIME=240
STALE_THRESHOLD_MINUTES=10  # Run fallback if predictions older than this

# Logging helper
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handler
error_handler() {
    log "ERROR: Script failed on line $1"
    exit 1
}
trap 'error_handler $LINENO' ERR

# Atomic lock acquisition
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "Another inference instance is running, skipping"
    exit 0
fi
echo $$ >&9
trap "flock -u 9" EXIT

# Verify current symlink exists
if [ ! -L "$CURRENT_DIR" ]; then
    log "ERROR: Current symlink not found at $CURRENT_DIR"
    exit 1
fi

cd "$CURRENT_DIR" || exit 1

# Load environment variables
set -a && source "$GEPT_DIR/.env" && set +a

# Activate virtual environment
source "$GEPT_DIR/venv/bin/activate"

# Check if predictions are fresh (Hydra is handling it)
FRESHNESS_CHECK=$(PGPASSWORD="$DB_PASS" psql -h localhost -U osrs_user -d osrs_data -t -c "
    SELECT CASE
        WHEN MAX(time) > NOW() - INTERVAL '${STALE_THRESHOLD_MINUTES} minutes'
        THEN 'FRESH'
        ELSE 'STALE'
    END
    FROM predictions;
" 2>/dev/null | tr -d ' \n')

if [ "$FRESHNESS_CHECK" = "FRESH" ]; then
    log "Predictions fresh (Hydra is active), skipping fallback"
    exit 0
fi

log "WARNING: Predictions stale (>${STALE_THRESHOLD_MINUTES}min), running fallback inference"
log "Starting inference cycle (Ampere fallback, version: $(basename "$(readlink "$CURRENT_DIR")"))"

# Run inference with timeout (uses default multi-target predictor for Ampere's models)
timeout $MAX_RUNTIME python run_inference.py

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    log "WARNING: Inference timed out after ${MAX_RUNTIME}s"
elif [ $EXIT_CODE -ne 0 ]; then
    log "ERROR: Inference failed with exit code $EXIT_CODE"
else
    log "Fallback inference cycle complete"
fi

exit $EXIT_CODE
