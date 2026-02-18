#!/bin/bash
# Sell-fill beta inference cron wrapper.

set -euo pipefail

GEPT_DIR="${GEPT_DIR:-/home/ubuntu/gept}"
LOCK_FILE="/tmp/gept_sellfill_beta_inference.lock"
MAX_RUNTIME=600

# Atomic lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another sell-fill beta instance is running, skipping"
    exit 0
fi
echo $$ >&9
trap "flock -u 9" EXIT

cd "$GEPT_DIR" || exit 1

set -a && source "$GEPT_DIR/.env" && set +a
source "$GEPT_DIR/venv/bin/activate"

ARTIFACT_DIR="${SELLFILL_ARTIFACT_DIR:-/data/gept/models/sell_fill_v1/180d_preditems_all_e1_cap6000_h1-24_e8_b8192}"
BASE_MODEL_ID="${SELLFILL_BASE_MODEL_ID:-gept_v2}"
OUT_MODEL_ID="${SELLFILL_BETA_MODEL_ID:-sellfill_beta_v1}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting sell-fill beta inference cycle"
timeout $MAX_RUNTIME python src/pipeline/run_sellfill_beta_inference.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --base-model-id "$BASE_MODEL_ID" \
    --out-model-id "$OUT_MODEL_ID"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Sell-fill beta inference timed out after ${MAX_RUNTIME}s"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Sell-fill beta inference failed with exit code $EXIT_CODE"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sell-fill beta inference cycle complete"
