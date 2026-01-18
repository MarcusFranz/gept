#!/bin/bash
# Daily Training Pipeline Orchestrator
# =====================================
#
# Main entry point for the daily training pipeline.
# Runs at 0500 EST via systemd timer.
#
# Pipeline Steps:
# 1. Initialize - Generate run ID, create training job record
# 2. Select Items - Choose items for training based on criteria
# 3. Train - Run remote training on WSL GPU machine
# 4. Validate - Validate trained models
# 5. Deploy - Activate validated models
# 6. Lifecycle - Process model lifecycle (sunset expired models)
# 7. Cleanup - Clean up temporary files
#
# Usage:
#   ./scripts/daily_training_pipeline.sh [options]
#
# Options:
#   --discovery         Discovery mode (train all items, 1st of month)
#   --max-items N       Override max items to train
#   --force-items IDs   Comma-separated item IDs to force-train
#   --skip-training     Skip training (validate/deploy existing run)
#   --run-id ID         Use existing run ID instead of generating new
#   --dry-run           Show what would be done without executing
#
# Environment variables (from .secrets/training.env):
#   DB_PASS             Database password (required)
#   WSL_HOST            WSL machine IP (required for training)
#   WSL_PORT            SSH port (default: 2222)
#   WSL_USER            SSH user (default: ubuntu)
#   WSL_KEY             SSH key path

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEPT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env files
# Use set -a to auto-export all sourced variables to child processes
if [ -f "$GEPT_DIR/.env" ]; then
    set -a
    source "$GEPT_DIR/.env"
    set +a
fi

# Also load training-specific secrets if available
if [ -f "$GEPT_DIR/.secrets/training.env" ]; then
    set -a
    source "$GEPT_DIR/.secrets/training.env"
    set +a
fi

# Validate required environment variables
validate_env_vars() {
    local missing=()

    if [ -z "${DB_PASS:-}" ]; then
        missing+=("DB_PASS")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: Missing required environment variables: ${missing[*]}" >&2
        echo "Set them in .env or .secrets/training.env" >&2
        exit 1
    fi
}

# Validate before any DB operations
validate_env_vars

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
LOG_DIR="/var/log/gept"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_DIR="/tmp/gept_logs"
mkdir -p "$LOG_DIR"

log() {
    local level=$1
    shift
    local msg="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local color=""

    case $level in
        INFO)  color=$GREEN ;;
        WARN)  color=$YELLOW ;;
        ERROR) color=$RED ;;
        STEP)  color=$BLUE ;;
        *)     color=$NC ;;
    esac

    echo -e "${color}[$level]${NC} $timestamp $msg" | tee -a "$LOG_FILE"
}

log_info()  { log INFO "$@"; }
log_warn()  { log WARN "$@"; }
log_error() { log ERROR "$@"; }
log_step()  { log STEP "$@"; }

# Error handler with traceback capture
error_exit() {
    local error_msg="$1"
    local traceback_file="/tmp/training_error_${RUN_ID}.txt"

    log_error "$error_msg"

    # Capture any Python traceback from recent log output
    if [ -f "$LOG_FILE" ]; then
        # Extract last traceback from log file (look for Traceback...Error pattern)
        local traceback
        traceback=$(grep -A 50 "Traceback" "$LOG_FILE" 2>/dev/null | tail -50 || echo "")
        if [ -n "$traceback" ]; then
            echo "$traceback" > "$traceback_file"
        fi
    fi

    # Update job status with error and traceback
    if [ "$DRY_RUN" = false ]; then
        python3 -c "
import sys
sys.path.insert(0, '.')
from src.db_utils import get_db_cursor

error_msg = '''$error_msg'''
traceback = ''
try:
    with open('$traceback_file', 'r') as f:
        traceback = f.read()[:4000]  # Limit to 4000 chars
except:
    pass

try:
    with get_db_cursor() as cur:
        cur.execute('''
            UPDATE training_jobs
            SET status = 'FAILED',
                status_changed_at = NOW(),
                error_message = %s,
                error_traceback = %s
            WHERE run_id = %s
        ''', (error_msg, traceback, '$RUN_ID'))
except Exception as e:
    print(f'Warning: Could not update job status: {e}', file=sys.stderr)
" 2>/dev/null || true
    fi

    exit 1
}

# Parse arguments
DISCOVERY=false
MAX_ITEMS=""
FORCE_ITEMS=""
SKIP_TRAINING=false
RUN_ID=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --discovery)
            DISCOVERY=true
            shift
            ;;
        --max-items)
            MAX_ITEMS="$2"
            shift 2
            ;;
        --force-items)
            FORCE_ITEMS="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate run ID if not provided
if [ -z "$RUN_ID" ]; then
    RUN_ID=$(date +%Y%m%d_%H%M%S)
fi

# Setup log file
LOG_FILE="$LOG_DIR/training_${RUN_ID}.log"

# Check for discovery mode (1st of month)
DAY_OF_MONTH=$(date +%d)
if [ "$DAY_OF_MONTH" = "01" ] && [ "$DISCOVERY" = false ]; then
    log_info "First day of month - enabling discovery mode"
    DISCOVERY=true
fi

log_info "=========================================="
log_info "GePT Daily Training Pipeline"
log_info "=========================================="
log_info "Run ID: $RUN_ID"
log_info "Log file: $LOG_FILE"
log_info "Discovery mode: $DISCOVERY"
if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No actual changes"
fi
log_info "=========================================="

cd "$GEPT_DIR"

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Helper function to update training job status
update_job_status() {
    local status=$1
    local message=${2:-""}

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would update job status to: $status"
        return
    fi

    python3 -c "
import sys
sys.path.insert(0, '.')
from src.db_utils import get_db_cursor

try:
    with get_db_cursor() as cur:
        cur.execute('''
            UPDATE training_jobs
            SET status = %s,
                status_changed_at = NOW(),
                error_message = CASE WHEN %s != '' THEN %s ELSE error_message END
            WHERE run_id = %s
        ''', ('$status', '$message', '$message', '$RUN_ID'))
except Exception as e:
    print(f'Warning: Could not update job status: {e}', file=sys.stderr)
" 2>/dev/null || true
}

# Step 1: Initialize training job
log_step "Step 1: Initializing training job..."

if [ "$DRY_RUN" = false ]; then
    python3 -c "
import sys
sys.path.insert(0, '.')
from src.db_utils import get_db_cursor
from datetime import datetime

try:
    with get_db_cursor() as cur:
        cur.execute('''
            INSERT INTO training_jobs (run_id, status, trigger_type, triggered_by)
            VALUES (%s, 'SELECTING', %s, 'cron')
            ON CONFLICT (run_id) DO UPDATE SET
                status = 'SELECTING',
                status_changed_at = NOW()
        ''', ('$RUN_ID', 'discovery' if '$DISCOVERY' == 'true' else 'scheduled'))
    print('Training job initialized')
except Exception as e:
    print(f'Warning: Could not create training job: {e}', file=sys.stderr)
" || log_warn "Could not initialize training job in database"
fi

# Step 2: Select items for training
log_step "Step 2: Selecting items for training..."

ITEMS_FILE="/tmp/training_items_${RUN_ID}.json"
SELECT_ARGS="--run-id $RUN_ID --output $ITEMS_FILE"

if [ "$DISCOVERY" = true ]; then
    SELECT_ARGS="$SELECT_ARGS --discovery"
fi
if [ -n "$MAX_ITEMS" ]; then
    SELECT_ARGS="$SELECT_ARGS --max-items $MAX_ITEMS"
fi
if [ -n "$FORCE_ITEMS" ]; then
    SELECT_ARGS="$SELECT_ARGS --force-items $FORCE_ITEMS"
fi

if [ "$DRY_RUN" = false ]; then
    python -m src.training.item_selector $SELECT_ARGS
else
    log_info "[DRY RUN] Would run: python -m src.training.item_selector $SELECT_ARGS"
fi

# Check item count
if [ "$DRY_RUN" = false ] && [ -f "$ITEMS_FILE" ]; then
    ITEM_COUNT=$(python3 -c "import json; print(len(json.load(open('$ITEMS_FILE'))['items']))")
    log_info "Selected $ITEM_COUNT items for training"

    if [ "$ITEM_COUNT" -eq 0 ]; then
        log_info "No items need training. Pipeline complete."
        update_job_status "COMPLETED"
        exit 0
    fi
else
    ITEM_COUNT=0
fi

# Step 3: Train models (remote)
if [ "$SKIP_TRAINING" = false ]; then
    log_step "Step 3: Training models on WSL..."

    update_job_status "TRAINING"

    TRAIN_ARGS="$RUN_ID"
    if [ -f "$ITEMS_FILE" ]; then
        TRAIN_ARGS="$TRAIN_ARGS --items $ITEMS_FILE"
    fi
    if [ "$DRY_RUN" = true ]; then
        TRAIN_ARGS="$TRAIN_ARGS --dry-run"
    fi

    if [ "$DRY_RUN" = false ]; then
        ./scripts/train_remote.sh $TRAIN_ARGS || error_exit "Training failed"
    else
        log_info "[DRY RUN] Would run: ./scripts/train_remote.sh $TRAIN_ARGS"
    fi

    log_info "Training complete"
else
    log_info "Skipping training (--skip-training)"
fi

# Step 4: Validate trained models
log_step "Step 4: Validating trained models..."

update_job_status "VALIDATING"

VALIDATION_FILE="/tmp/validation_${RUN_ID}.json"

if [ "$DRY_RUN" = false ]; then
    python -m src.training.model_validator \
        --run-id "$RUN_ID" \
        --register \
        --output "$VALIDATION_FILE" \
        || error_exit "Validation failed"

    # Parse validation results
    PASSED=$(python3 -c "import json; print(json.load(open('$VALIDATION_FILE'))['passed'])")
    FAILED=$(python3 -c "import json; print(json.load(open('$VALIDATION_FILE'))['failed'])")
    log_info "Validation: $PASSED passed, $FAILED failed"

    if [ "$PASSED" -eq 0 ]; then
        log_warn "No models passed validation. Skipping deployment."
        update_job_status "COMPLETED" "No models passed validation"
        exit 0
    fi
else
    log_info "[DRY RUN] Would run: python -m src.training.model_validator --run-id $RUN_ID --register"
fi

# Step 5: Deploy validated models
log_step "Step 5: Deploying validated models..."

update_job_status "DEPLOYING"

if [ "$DRY_RUN" = false ]; then
    python -m src.training.lifecycle_manager deploy --run-id "$RUN_ID" \
        || error_exit "Deployment failed"
    log_info "Deployment complete"
else
    log_info "[DRY RUN] Would run: python -m src.training.lifecycle_manager deploy --run-id $RUN_ID"
fi

# Step 6: Process lifecycle (sunset expired models)
log_step "Step 6: Processing model lifecycle..."

if [ "$DRY_RUN" = false ]; then
    python -m src.training.lifecycle_manager process-lifecycle \
        || log_warn "Lifecycle processing had issues"
else
    log_info "[DRY RUN] Would run: python -m src.training.lifecycle_manager process-lifecycle"
fi

# Step 7: Cleanup
log_step "Step 7: Cleanup..."

if [ "$DRY_RUN" = false ]; then
    # Remove temporary files
    rm -f "$ITEMS_FILE" 2>/dev/null || true
    rm -f "$VALIDATION_FILE" 2>/dev/null || true

    # Guard rm -rf to prevent accidental deletion if RUN_ID is empty
    if [ -n "${RUN_ID:-}" ] && [ -d "/tmp/gept_training_data/$RUN_ID" ]; then
        rm -rf "/tmp/gept_training_data/$RUN_ID" || log_warn "Failed to cleanup training data"
    fi

    log_info "Cleanup complete"
fi

# Update final status
if [ "$DRY_RUN" = false ]; then
    python3 -c "
import sys
sys.path.insert(0, '.')
from src.db_utils import get_db_cursor

try:
    with get_db_cursor() as cur:
        cur.execute('''
            UPDATE training_jobs
            SET status = 'COMPLETED',
                status_changed_at = NOW(),
                completed_at = NOW()
            WHERE run_id = %s
        ''', ('$RUN_ID',))
except Exception as e:
    print(f'Warning: Could not update job status: {e}', file=sys.stderr)
" || true
fi

log_info "=========================================="
log_info "Daily Training Pipeline Complete"
log_info "=========================================="
log_info "Run ID: $RUN_ID"
log_info "Log file: $LOG_FILE"
log_info "=========================================="

# Summary
if [ "$DRY_RUN" = false ]; then
    python -m src.training.lifecycle_manager summary || true
fi
