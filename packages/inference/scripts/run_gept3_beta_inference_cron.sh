#!/bin/bash
# GePT3 hybrid beta inference cron wrapper.

set -euo pipefail

GEPT_DIR="${GEPT_DIR:-/home/ubuntu/gept}"
LOCK_FILE="/tmp/gept3_beta_inference.lock"
MAX_RUNTIME="${MAX_RUNTIME:-1800}"  # 30m default; hybrid beta can be heavier.

# Atomic lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another gept3 beta instance is running, skipping"
    exit 0
fi
echo $$ >&9
trap "flock -u 9" EXIT

cd "$GEPT_DIR" || exit 1

set -a && source "$GEPT_DIR/.env" && set +a
source "$GEPT_DIR/venv/bin/activate"

ARTIFACT_DIR="${GEPT3_BETA_ARTIFACT_DIR:-/data/gept/models/gept3/gept3_hybrid_e4_beta}"
BASE_MODEL_ID="${GEPT3_BETA_BASE_MODEL_ID:-gept_v2}"
OUT_MODEL_ID="${GEPT3_BETA_MODEL_ID:-gept3_hybrid_e4_beta}"
DEVICE="${GEPT3_BETA_DEVICE:-auto}"
CHUNK_SIZE="${GEPT3_BETA_CHUNK_SIZE:-256}"
FEATURE_LOOKBACK_DAYS="${GEPT3_BETA_FEATURE_LOOKBACK_DAYS:-10}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting gept3 beta inference cycle"
timeout "$MAX_RUNTIME" python src/pipeline/run_gept3_beta_inference.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --base-model-id "$BASE_MODEL_ID" \
    --out-model-id "$OUT_MODEL_ID" \
    --device "$DEVICE" \
    --chunk-size "$CHUNK_SIZE" \
    --feature-lookback-days "$FEATURE_LOOKBACK_DAYS"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: GePT3 beta inference timed out after ${MAX_RUNTIME}s"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: GePT3 beta inference failed with exit code $EXIT_CODE"
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GePT3 beta inference cycle complete"

