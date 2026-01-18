#!/bin/bash
# Daily GePT Training Data Export
#
# Exports training data from PostgreSQL to GCS and triggers Cloud Run training.
# Runs daily via systemd timer (gept-data-export.timer).
#
# Environment variables (set in systemd service or .env):
#   DB_PASS               - PostgreSQL password
#   GCS_BUCKET            - GCS bucket name (default: osrs-models-mof)
#   GOOGLE_APPLICATION_CREDENTIALS - Path to GCP service account key
#   TIER_MODE             - Enable tiered training (default: true)
#   FORCE_DISCOVERY       - Force full discovery run (default: false)
#
# Usage:
#   ./daily_export.sh                    # Normal tiered run
#   ./daily_export.sh --discovery        # Force discovery (all items)
#   ./daily_export.sh --dry-run          # Test without uploading

set -euo pipefail

# Configuration
GEPT_DIR="${GEPT_DIR:-/home/ubuntu/gept}"
GCS_BUCKET="${GCS_BUCKET:-osrs-models-mof}"
REGION="${REGION:-us-central1}"
JOB_NAME="${JOB_NAME:-gept-daily-train}"
LOG_DIR="${LOG_DIR:-/var/log/gept}"
TIER_MODE="${TIER_MODE:-true}"
FORCE_DISCOVERY="${FORCE_DISCOVERY:-false}"

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --discovery)
            FORCE_DISCOVERY="true"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --bucket)
            GCS_BUCKET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate run ID (date-based for easy identification)
RUN_ID=$(date +%Y-%m-%d)
LOG_FILE="${LOG_DIR}/export_${RUN_ID}.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# Error handler
error_exit() {
    log "ERROR: $1"
    # TODO: Add alerting (Discord webhook, PagerDuty, etc.)
    exit 1
}

# Check if it's discovery day (1st of month)
is_discovery_day() {
    [[ $(date +%d) == "01" ]] && return 0 || return 1
}

log "=========================================="
log "GePT Daily Export Starting"
log "=========================================="
log "Run ID:          $RUN_ID"
log "GCS Bucket:      $GCS_BUCKET"
log "Tier Mode:       $TIER_MODE"
log "Force Discovery: $FORCE_DISCOVERY"
log "Dry Run:         $DRY_RUN"

# Check for discovery day
if is_discovery_day; then
    log "Discovery day detected (1st of month) - will train all items"
    FORCE_DISCOVERY="true"
fi

# Verify environment
if [[ -z "${DB_PASS:-}" ]]; then
    error_exit "DB_PASS environment variable not set"
fi

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    error_exit "GOOGLE_APPLICATION_CREDENTIALS not set"
fi

if [[ ! -f "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
    error_exit "GCP credentials file not found: ${GOOGLE_APPLICATION_CREDENTIALS}"
fi

# Change to project directory
cd "$GEPT_DIR" || error_exit "Cannot change to $GEPT_DIR"

# Activate virtual environment if it exists
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    log "Activated virtual environment"
fi

# Run data export
log ""
log "Starting data preparation..."
EXPORT_START=$(date +%s)

EXPORT_CMD="python cloud/prepare_training_data.py --bucket $GCS_BUCKET --run-id $RUN_ID --workers 4"

if $DRY_RUN; then
    EXPORT_CMD="$EXPORT_CMD --dry-run"
fi

log "Command: $EXPORT_CMD"

if ! $EXPORT_CMD 2>&1 | tee -a "$LOG_FILE"; then
    error_exit "Data export failed"
fi

EXPORT_END=$(date +%s)
EXPORT_DURATION=$((EXPORT_END - EXPORT_START))
log "Data export completed in ${EXPORT_DURATION}s"

if $DRY_RUN; then
    log ""
    log "[DRY RUN] Skipping Cloud Run job trigger"
    log "=========================================="
    log "Dry run complete"
    exit 0
fi

# Get item count from config
log ""
log "Reading run config..."
ITEM_COUNT=$(gsutil cat "gs://${GCS_BUCKET}/runs/${RUN_ID}/config.json" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['items']))")

if [[ -z "$ITEM_COUNT" ]] || [[ "$ITEM_COUNT" == "0" ]]; then
    error_exit "No items found in config"
fi

log "Items to train: $ITEM_COUNT"

# Trigger Cloud Run Job
log ""
log "Triggering Cloud Run Job..."
log "  Job:    $JOB_NAME"
log "  Region: $REGION"
log "  Tasks:  $ITEM_COUNT"

TRIGGER_CMD="gcloud run jobs execute $JOB_NAME \
    --region $REGION \
    --tasks $ITEM_COUNT \
    --set-env-vars RUN_ID=$RUN_ID,GCS_BUCKET=$GCS_BUCKET,TIER_MODE=$TIER_MODE,FORCE_DISCOVERY=$FORCE_DISCOVERY"

log "Command: $TRIGGER_CMD"

if ! $TRIGGER_CMD 2>&1 | tee -a "$LOG_FILE"; then
    error_exit "Failed to trigger Cloud Run job"
fi

log ""
log "=========================================="
log "Export complete!"
log "=========================================="
log "Run ID:      $RUN_ID"
log "Items:       $ITEM_COUNT"
log "Duration:    ${EXPORT_DURATION}s"
log ""
log "Monitor training:"
log "  gcloud run jobs executions list --job=$JOB_NAME --region=$REGION"
log ""
log "View logs:"
log "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=$JOB_NAME\" --limit=100"

exit 0
