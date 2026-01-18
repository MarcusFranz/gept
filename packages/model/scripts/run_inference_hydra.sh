#!/bin/bash
# =============================================================================
# GePT Inference - Hydra (Primary)
# =============================================================================
# Runs inference on Hydra, writes predictions to Ampere DB via SSH tunnel.
# This is the PRIMARY inference runner - Ampere only runs if Hydra is down.
#
# Crontab entry (on Hydra):
#   */5 * * * * /home/ubuntu/gept/scripts/run_inference_hydra.sh >> /home/ubuntu/gept/logs/inference.log 2>&1
# =============================================================================

set -euo pipefail

GEPT_DIR="/home/ubuntu/gept"
LOCK_FILE="/tmp/gept_inference.lock"
MAX_RUNTIME=240  # 4 minutes max

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

cd "$GEPT_DIR" || exit 1

# Load environment variables
set -a && source "$GEPT_DIR/.env" && set +a

# Ensure SSH tunnel to Ampere DB is up
if ! nc -z localhost 5432 2>/dev/null; then
    log "WARNING: DB tunnel not available, attempting to establish..."
    # Try to establish tunnel (assumes key is configured)
    ssh -f -N -L 5432:localhost:5432 ubuntu@150.136.170.128 2>/dev/null || {
        log "ERROR: Could not establish DB tunnel"
        exit 1
    }
    sleep 2
fi

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gept

log "Starting inference cycle (Hydra primary)"

# Run inference with timeout
# Uses CPU for inference to leave GPU free for training
# Uses multi-target format (models/<run_id>/<item_id>/model.cbm)
# Loads data from local Parquet cache (no DB queries over network)
export CUDA_VISIBLE_DEVICES=""  # Force CPU inference
timeout $MAX_RUNTIME python run_inference.py --cache-dir /home/ubuntu/gept/data_cache

EXIT_CODE=$?

if [ $EXIT_CODE -eq 124 ]; then
    log "WARNING: Inference timed out after ${MAX_RUNTIME}s"
elif [ $EXIT_CODE -ne 0 ]; then
    log "ERROR: Inference failed with exit code $EXIT_CODE"
else
    log "Inference cycle complete"
fi

exit $EXIT_CODE
