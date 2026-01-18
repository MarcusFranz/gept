#!/bin/bash
# Remote Training Wrapper Script
# ==============================
#
# Orchestrates training on a remote WSL machine:
# 1. Prepares data locally (on Ampere)
# 2. Transfers data to WSL via rsync
# 3. Triggers GPU training on WSL
# 4. Retrieves trained models back to Ampere
#
# Usage:
#   ./scripts/train_remote.sh <run_id> [options]
#
# Options:
#   --items <file>      JSON file with items to train (from item_selector)
#   --skip-prepare      Skip data preparation (data already exists)
#   --skip-transfer     Skip data transfer (data already on WSL)
#   --dry-run           Show what would be done without executing
#
# Environment variables:
#   WSL_HOST            WSL machine IP address (required)
#   WSL_PORT            SSH port (default: 2222)
#   WSL_USER            SSH user (default: ubuntu)
#   WSL_KEY             SSH key path (default: .secrets/wsl_key.pem)
#   WSL_DIR             Remote gept directory (default: /home/ubuntu/gept)
#   DB_PASS             Database password (required for data prep)
#
# Example:
#   export WSL_HOST=192.168.1.100
#   export DB_PASS=mypassword
#   ./scripts/train_remote.sh 20260111_050000

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEPT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env files
if [ -f "$GEPT_DIR/.env" ]; then
    set -a
    source "$GEPT_DIR/.env"
    set +a
fi

if [ -f "$GEPT_DIR/.secrets/training.env" ]; then
    set -a
    source "$GEPT_DIR/.secrets/training.env"
    set +a
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"; }

# Parse arguments
RUN_ID=""
ITEMS_FILE=""
SKIP_PREPARE=false
SKIP_TRANSFER=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --items)
            ITEMS_FILE="$2"
            shift 2
            ;;
        --skip-prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --skip-transfer)
            SKIP_TRANSFER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            log_error "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$RUN_ID" ]; then
                RUN_ID="$1"
            else
                log_error "Unexpected argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate run_id
if [ -z "$RUN_ID" ]; then
    log_error "Usage: $0 <run_id> [options]"
    exit 1
fi

# Configuration from environment
WSL_HOST="${WSL_HOST:-}"
WSL_PORT="${WSL_PORT:-2222}"
WSL_USER="${WSL_USER:-ubuntu}"
WSL_KEY="${WSL_KEY:-$GEPT_DIR/.secrets/wsl_key.pem}"
WSL_DIR="${WSL_DIR:-/home/ubuntu/gept}"

# Validate required variables
if [ -z "$WSL_HOST" ]; then
    log_error "WSL_HOST environment variable is required"
    exit 1
fi

if [ -z "${DB_PASS:-}" ]; then
    log_error "DB_PASS environment variable is required"
    log_error "Set it in .env or .secrets/training.env"
    exit 1
fi

if [ ! -f "$WSL_KEY" ]; then
    log_error "SSH key not found: $WSL_KEY"
    exit 1
fi

# Local paths
LOCAL_DATA_DIR="/tmp/gept_training_data"
LOCAL_MODELS_DIR="$GEPT_DIR/models"

# Remote paths
REMOTE_DATA_DIR="$WSL_DIR/data/prepared"
REMOTE_MODELS_DIR="$WSL_DIR/models"

# SSH options
SSH_OPTS="-i $WSL_KEY -p $WSL_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
SSH_CMD="ssh $SSH_OPTS $WSL_USER@$WSL_HOST"
SCP_CMD="scp $SSH_OPTS"
RSYNC_CMD="rsync -avz -e \"ssh $SSH_OPTS\""

log_info "=========================================="
log_info "Remote Training Pipeline"
log_info "=========================================="
log_info "Run ID: $RUN_ID"
log_info "WSL Host: $WSL_USER@$WSL_HOST:$WSL_PORT"
log_info "Remote Dir: $WSL_DIR"
if [ -n "$ITEMS_FILE" ]; then
    log_info "Items File: $ITEMS_FILE"
fi
if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No actual execution"
fi
log_info "=========================================="

# Step 1: Test SSH connection
log_step "Step 1: Testing SSH connection..."
if [ "$DRY_RUN" = false ]; then
    if ! $SSH_CMD "echo 'Connection successful'" 2>/dev/null; then
        log_error "Cannot connect to WSL machine"
        log_error "Ensure:"
        log_error "  1. WSL SSH server is running (sudo service ssh start)"
        log_error "  2. Port forwarding is configured on Windows host"
        log_error "  3. SSH key has correct permissions (chmod 600)"
        exit 1
    fi
    log_info "SSH connection successful"
else
    log_info "[DRY RUN] Would test SSH connection"
fi

# Step 2: Prepare training data (on WSL to avoid filling Ampere disk)
if [ "$SKIP_PREPARE" = false ]; then
    log_step "Step 2: Preparing training data on WSL..."

    PREPARE_ARGS="--output-dir $REMOTE_DATA_DIR --run-id $RUN_ID"

    # If items file provided, extract item IDs
    if [ -n "$ITEMS_FILE" ] && [ -f "$ITEMS_FILE" ]; then
        ITEM_IDS=$(python3 -c "
import json
with open('$ITEMS_FILE') as f:
    data = json.load(f)
items = data.get('items', data)
print(','.join(str(i.get('item_id', i)) for i in items))
")
        if [ -n "$ITEM_IDS" ]; then
            PREPARE_ARGS="$PREPARE_ARGS --items $ITEM_IDS"
        fi
    fi

    if [ "$DRY_RUN" = false ]; then
        # Sync prepare script and dependencies to WSL
        eval $RSYNC_CMD "$GEPT_DIR/cloud/prepare_runpod_data.py" "$WSL_USER@$WSL_HOST:$WSL_DIR/cloud/"
        eval $RSYNC_CMD "$GEPT_DIR/src/feature_engine.py" "$WSL_USER@$WSL_HOST:$WSL_DIR/src/"

        # Run data prep on WSL (via SSH tunnel to Ampere DB)
        # Note: WSL needs DB_PASS env var and SSH tunnel: ssh -L 5432:localhost:5432 ampere
        log_info "Running data prep on WSL with args: $PREPARE_ARGS"
        $SSH_CMD "cd $WSL_DIR && source venv/bin/activate && DB_PASS='$DB_PASS' python cloud/prepare_runpod_data.py $PREPARE_ARGS"
        log_info "Data preparation complete"
    else
        log_info "[DRY RUN] Would run on WSL: python cloud/prepare_runpod_data.py $PREPARE_ARGS"
    fi
else
    log_info "Skipping data preparation (--skip-prepare)"
fi

# Step 3: Skip transfer (data already on WSL)
log_info "Data prepared directly on WSL, no transfer needed"

# Step 4: Run training on WSL
log_step "Step 4: Starting training on WSL..."

TRAIN_CMD="cd $WSL_DIR && \
    source venv/bin/activate && \
    python cloud/train_runpod_multitarget.py \
        --run-id $RUN_ID \
        --all \
        --local \
        --cache-dir $REMOTE_DATA_DIR \
        --output-dir $REMOTE_MODELS_DIR \
        --threads 8 \
        --numba-threads 8 \
        --prefetch 2"

if [ "$DRY_RUN" = false ]; then
    log_info "Training command: $TRAIN_CMD"

    # Run training (this may take hours)
    START_TIME=$(date +%s)

    $SSH_CMD "$TRAIN_CMD"

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))

    log_info "Training complete in $DURATION_MIN minutes"
else
    log_info "[DRY RUN] Would run on WSL: $TRAIN_CMD"
fi

# Step 5: Retrieve trained models
log_step "Step 5: Retrieving trained models..."

if [ "$DRY_RUN" = false ]; then
    # Create local directory
    mkdir -p "$LOCAL_MODELS_DIR/$RUN_ID"

    # Retrieve models
    eval $RSYNC_CMD \
        "$WSL_USER@$WSL_HOST:$REMOTE_MODELS_DIR/$RUN_ID/" \
        "$LOCAL_MODELS_DIR/$RUN_ID/"

    log_info "Model retrieval complete"

    # Verify models
    MODEL_COUNT=$(find "$LOCAL_MODELS_DIR/$RUN_ID" -name "model.cbm" | wc -l)
    log_info "Retrieved $MODEL_COUNT models"

    if [ "$MODEL_COUNT" -eq 0 ]; then
        log_error "No models found! Training may have failed."
        exit 1
    fi
else
    log_info "[DRY RUN] Would retrieve $WSL_HOST:$REMOTE_MODELS_DIR/$RUN_ID/ to $LOCAL_MODELS_DIR/$RUN_ID/"
fi

# Step 6: Cleanup prepared data on WSL
log_step "Step 6: Cleaning up prepared data on WSL..."

if [ "$DRY_RUN" = false ]; then
    $SSH_CMD "rm -rf $REMOTE_DATA_DIR/$RUN_ID && echo 'Cleaned up $REMOTE_DATA_DIR/$RUN_ID'"
    log_info "Cleanup complete"
else
    log_info "[DRY RUN] Would remove $REMOTE_DATA_DIR/$RUN_ID on WSL"
fi

# Step 7: Summary
log_info "=========================================="
log_info "Training Pipeline Complete"
log_info "=========================================="
log_info "Run ID: $RUN_ID"
log_info "Models directory: $LOCAL_MODELS_DIR/$RUN_ID"

if [ "$DRY_RUN" = false ]; then
    # Show training summary if available
    SUMMARY_FILE="$LOCAL_MODELS_DIR/$RUN_ID/training_summary.json"
    if [ -f "$SUMMARY_FILE" ]; then
        log_info "Training summary:"
        python3 -c "
import json
with open('$SUMMARY_FILE') as f:
    s = json.load(f)
print(f\"  Items trained: {s.get('items_trained', 'N/A')}\")
print(f\"  Items failed: {s.get('items_failed', 'N/A')}\")
print(f\"  Duration: {s.get('total_duration_seconds', 0) / 60:.1f} minutes\")
"
    fi
fi

log_info "=========================================="
log_info "Next steps:"
log_info "  1. Validate models: python -m src.training.model_validator --run-id $RUN_ID"
log_info "  2. Deploy models: python -m src.training.lifecycle_manager deploy --run-id $RUN_ID"
log_info "=========================================="
