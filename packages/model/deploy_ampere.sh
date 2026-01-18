#!/bin/bash
# =============================================================================
# Deploy GePT Prediction Engine to Ampere Server
# =============================================================================
# Supports versioned deployments with atomic symlink switching and rollback.
#
# Usage:
#   ./deploy_ampere.sh              # Deploy new version
#   ./deploy_ampere.sh --rollback   # Rollback to previous version
#   ./deploy_ampere.sh --rollback VERSION  # Rollback to specific version
#   ./deploy_ampere.sh --list       # List available versions
#   ./deploy_ampere.sh --dry-run    # Preview deployment without executing
#   ./deploy_ampere.sh --keep N     # Keep N releases (default: 5)
#
# Directory Structure (after deployment):
#   /home/ubuntu/gept/
#   ├── releases/
#   │   └── YYYYMMDD_HHMMSS/
#   │       ├── src/
#   │       ├── models/
#   │       ├── scripts/
#   │       └── deploy_info.json
#   ├── current -> releases/YYYYMMDD_HHMMSS  (atomic symlink)
#   ├── logs/
#   └── venv/
# =============================================================================

set -euo pipefail

# Error handler for debugging
error_handler() {
    echo "ERROR: Script failed on line $1" >&2
    exit 1
}
trap 'error_handler $LINENO' ERR

# Determine script directory
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source central server configuration
if [ -f "$LOCAL_DIR/config/servers.env" ]; then
    # shellcheck source=config/servers.env
    source "$LOCAL_DIR/config/servers.env"
fi

# Configuration (uses values from servers.env with fallback defaults)
REMOTE_DIR="${AMPERE_GEPT_DIR:-/home/ubuntu/gept}"
AMPERE_KEY="${AMPERE_SSH_KEY:-$LOCAL_DIR/.secrets/oracle_key.pem}"
# Resolve relative path for SSH key
if [[ ! "$AMPERE_KEY" = /* ]]; then
    AMPERE_KEY="$LOCAL_DIR/$AMPERE_KEY"
fi
KEEP_RELEASES=5
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    echo "GePT Prediction Engine - Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --rollback [VERSION]  Rollback to previous or specific version"
    echo "  --list                List available versions"
    echo "  --dry-run             Preview deployment without executing"
    echo "  --keep N              Keep N releases (default: 5)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Deploy new version"
    echo "  $0 --rollback         # Rollback to previous version"
    echo "  $0 --rollback 20260113_140000  # Rollback to specific version"
    echo "  $0 --list             # Show available versions"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "$AMPERE_KEY" ]; then
        log_error "SSH key not found: $AMPERE_KEY"
        exit 1
    fi

    chmod 600 "$AMPERE_KEY"

    # Test SSH connection
    if ! ssh -i "$AMPERE_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$AMPERE_HOST" "echo ok" > /dev/null 2>&1; then
        log_error "Cannot connect to $AMPERE_HOST"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

list_versions() {
    log_info "Available versions on $AMPERE_HOST:"
    echo ""

    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        if [ -d '$REMOTE_DIR/releases' ]; then
            CURRENT_VERSION=\$(readlink '$REMOTE_DIR/current' 2>/dev/null | xargs basename 2>/dev/null || echo 'none')
            for version in \$(ls -1t '$REMOTE_DIR/releases' 2>/dev/null); do
                if [ \"\$version\" = \"\$CURRENT_VERSION\" ]; then
                    echo \"  * \$version (current)\"
                else
                    echo \"    \$version\"
                fi
            done
        else
            echo '  No releases found. Run a deployment first.'
        fi
    "
    echo ""
}

get_previous_version() {
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        ls -1t '$REMOTE_DIR/releases' 2>/dev/null | head -2 | tail -1
    "
}

get_current_version() {
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        readlink '$REMOTE_DIR/current' 2>/dev/null | xargs basename 2>/dev/null || echo ''
    "
}

do_rollback() {
    local target_version=$1

    if [ -z "$target_version" ]; then
        target_version=$(get_previous_version)
    fi

    if [ -z "$target_version" ]; then
        log_error "No previous version available for rollback"
        exit 1
    fi

    local current_version
    current_version=$(get_current_version)

    if [ "$target_version" = "$current_version" ]; then
        log_warning "Version $target_version is already current"
        exit 0
    fi

    log_info "Rolling back from $current_version to $target_version..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would switch symlink to $target_version"
        log_info "[DRY RUN] Would restart cron job"
        return
    fi

    # Verify target version exists
    if ! ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "[ -d '$REMOTE_DIR/releases/$target_version' ]"; then
        log_error "Version $target_version does not exist"
        echo ""
        list_versions
        exit 1
    fi

    # Atomic symlink switch
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        ln -sfn '$REMOTE_DIR/releases/$target_version' '$REMOTE_DIR/current.new'
        mv -Tf '$REMOTE_DIR/current.new' '$REMOTE_DIR/current'
    "

    log_success "Rolled back to version $target_version"

    # Verify the rollback
    verify_deployment
}

pre_deployment_check() {
    log_info "Running pre-deployment health check..."

    # Check if there's an existing deployment
    local has_current
    has_current=$(ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "[ -L '$REMOTE_DIR/current' ] && echo 'yes' || echo 'no'")

    if [ "$has_current" = "yes" ]; then
        # Check if inference is running and healthy
        local inference_running
        inference_running=$(ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "pgrep -f 'run_inference.py' > /dev/null 2>&1 && echo 'yes' || echo 'no'")

        if [ "$inference_running" = "yes" ]; then
            log_info "Inference process is running"
        else
            log_warning "No active inference process detected"
        fi
    else
        log_info "No existing deployment detected (first deployment)"
    fi

    log_success "Pre-deployment check passed"
}

migrate_existing_deployment() {
    log_info "Checking for existing non-versioned deployment..."

    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        # Create releases directory if it doesn't exist
        mkdir -p '$REMOTE_DIR/releases'

        # If there's an existing deployment without versioning, migrate it
        if [ -d '$REMOTE_DIR/src' ] && [ ! -L '$REMOTE_DIR/current' ]; then
            echo 'Migrating existing deployment to versioned structure...'
            MIGRATE_VERSION=\$(date +%Y%m%d_%H%M%S)_migrated
            mkdir -p '$REMOTE_DIR/releases/'\$MIGRATE_VERSION

            # Move existing directories to the migration release
            for dir in src models scripts; do
                if [ -d '$REMOTE_DIR/'\$dir ]; then
                    mv '$REMOTE_DIR/'\$dir '$REMOTE_DIR/releases/'\$MIGRATE_VERSION/
                fi
            done

            # Move run_inference.py if it exists at root
            if [ -f '$REMOTE_DIR/run_inference.py' ]; then
                mv '$REMOTE_DIR/run_inference.py' '$REMOTE_DIR/releases/'\$MIGRATE_VERSION/
            fi

            # Create symlink to migrated version
            ln -sfn '$REMOTE_DIR/releases/'\$MIGRATE_VERSION '$REMOTE_DIR/current'
            echo 'Migration complete: '\$MIGRATE_VERSION
        fi
    "
}

deploy_version() {
    local deploy_version
    deploy_version=$(date +%Y%m%d_%H%M%S)
    local deploy_dir="$REMOTE_DIR/releases/$deploy_version"
    local git_commit
    git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    log_info "Deploying version: $deploy_version"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create release directory: $deploy_dir"
        log_info "[DRY RUN] Would sync source code, models, and scripts"
        log_info "[DRY RUN] Would create symlink: current -> releases/$deploy_version"
        return
    fi

    echo "=========================================="
    echo "GePT Prediction Engine Deployment"
    echo "=========================================="
    echo "Local directory: $LOCAL_DIR"
    echo "Deploy version:  $deploy_version"
    echo "Git commit:      $git_commit"
    echo ""

    # Create release directory structure
    log_info "[1/7] Creating release directory structure..."
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "mkdir -p $deploy_dir/{src,models,scripts}"

    # Sync source code
    log_info "[2/7] Syncing source code..."
    scp -i "$AMPERE_KEY" -r "$LOCAL_DIR/src/"* "$AMPERE_HOST:$deploy_dir/src/"

    # Sync inference scripts
    log_info "[3/7] Syncing inference scripts..."
    scp -i "$AMPERE_KEY" "$LOCAL_DIR/run_inference.py" "$AMPERE_HOST:$deploy_dir/"
    scp -i "$AMPERE_KEY" -r "$LOCAL_DIR/scripts/"* "$AMPERE_HOST:$deploy_dir/scripts/"

    # Sync requirements.txt
    log_info "[3.5/7] Syncing requirements.txt..."
    scp -i "$AMPERE_KEY" "$LOCAL_DIR/requirements.txt" "$AMPERE_HOST:$deploy_dir/"

    # Sync models
    log_info "[4/7] Syncing trained models (this may take a while)..."
    if [ -f "$LOCAL_DIR/trained_models.tar.gz" ]; then
        echo "Using pre-compressed models archive..."
        scp -i "$AMPERE_KEY" "$LOCAL_DIR/trained_models.tar.gz" "$AMPERE_HOST:$deploy_dir/"
        ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "cd $deploy_dir && tar -xzf trained_models.tar.gz && rm trained_models.tar.gz"
    else
        echo "Creating models archive (multi-target format)..."
        cd "$LOCAL_DIR"
        # Find latest model directory (handles non-alphanumeric filenames safely)
        LATEST_MODEL_DIR=$(find models -maxdepth 1 -type d -name '20*' 2>/dev/null | sort | tail -1)
        if [ -z "$LATEST_MODEL_DIR" ]; then
            log_warning "No multi-target models found in models/. Trying models_downloaded..."
            LATEST_MODEL_DIR="models_downloaded"
        fi
        echo "Using model directory: $LATEST_MODEL_DIR"
        tar -czf /tmp/gept_models.tar.gz "$LATEST_MODEL_DIR"
        scp -i "$AMPERE_KEY" /tmp/gept_models.tar.gz "$AMPERE_HOST:$deploy_dir/"
        ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "cd $deploy_dir && tar -xzf gept_models.tar.gz && rm gept_models.tar.gz"
        rm /tmp/gept_models.tar.gz
    fi

    # Create deployment info
    log_info "[5/7] Recording deployment metadata..."
    local previous_version
    previous_version=$(get_current_version)
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "cat > $deploy_dir/deploy_info.json << EOF
{
    \"version\": \"$deploy_version\",
    \"deployed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"deployed_by\": \"deploy_ampere.sh\",
    \"git_commit\": \"$git_commit\",
    \"previous_version\": \"$previous_version\"
}
EOF"

    # Atomic symlink switch
    log_info "[6/7] Activating new version (atomic switch)..."
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        ln -sfn '$deploy_dir' '$REMOTE_DIR/current.new'
        mv -Tf '$REMOTE_DIR/current.new' '$REMOTE_DIR/current'
    "

    # Setup Python environment and cron
    log_info "[7/7] Setting up Python environment and scheduled jobs..."
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" << 'REMOTE_SETUP'
cd /home/ubuntu/gept

# Create shared directories if not exist
mkdir -p logs

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies from repo requirements.txt
pip install --upgrade pip -q
pip install -r current/requirements.txt -q

# Update inference cron job to use current symlink
chmod +x /home/ubuntu/gept/current/scripts/run_inference_cron.sh

CRON_LINE="*/5 * * * * /home/ubuntu/gept/current/scripts/run_inference_cron.sh >> /home/ubuntu/gept/logs/inference.log 2>&1"
(crontab -l 2>/dev/null | grep -v "run_inference_cron.sh"; echo "$CRON_LINE") | crontab -

# Install training scheduler (systemd timer)
# Only install if systemd files exist and we have permission
if [ -f "/home/ubuntu/gept/current/scripts/gept-training.service" ] && [ -f "/home/ubuntu/gept/current/scripts/gept-training.timer" ]; then
    echo "Installing training scheduler..."

    # Copy service files to systemd directory (requires sudo)
    sudo cp /home/ubuntu/gept/current/scripts/gept-training.service /etc/systemd/system/
    sudo cp /home/ubuntu/gept/current/scripts/gept-training.timer /etc/systemd/system/

    # Reload systemd and enable timer
    sudo systemctl daemon-reload
    sudo systemctl enable gept-training.timer
    sudo systemctl start gept-training.timer

    echo "Training scheduler installed (runs daily at 10:00 UTC)"
    systemctl status gept-training.timer --no-pager || true
else
    echo "WARNING: Training scheduler files not found, skipping installation"
fi

echo "Environment and scheduled jobs configured"
REMOTE_SETUP

    log_success "Deployed version $deploy_version"

    # Prune old releases
    prune_old_releases
}

prune_old_releases() {
    log_info "Pruning old releases (keeping last $KEEP_RELEASES)..."

    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        cd '$REMOTE_DIR/releases' 2>/dev/null || exit 0
        CURRENT_VERSION=\$(readlink '$REMOTE_DIR/current' | xargs basename)

        # Get all versions sorted by date (newest first), skip current and keep N
        count=0
        for version in \$(ls -1t); do
            count=\$((count + 1))
            if [ \$count -gt $KEEP_RELEASES ] && [ \"\$version\" != \"\$CURRENT_VERSION\" ]; then
                echo \"Removing old release: \$version\"
                rm -rf \"\$version\"
            fi
        done
    "

    log_success "Old releases pruned"
}

verify_deployment() {
    log_info "Verifying deployment..."

    local verification_failed=false

    echo ""
    echo "=== Current Version ==="
    local symlink_ok
    symlink_ok=$(ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        if [ -L '$REMOTE_DIR/current' ]; then
            VERSION=\$(readlink '$REMOTE_DIR/current' | xargs basename)
            echo \"Active version: \$VERSION\"
            if [ -f '$REMOTE_DIR/current/deploy_info.json' ]; then
                cat '$REMOTE_DIR/current/deploy_info.json'
            fi
            echo 'SYMLINK_OK'
        else
            echo 'ERROR: No current version symlink found'
            echo 'SYMLINK_FAILED'
        fi
    ")
    echo "$symlink_ok"
    if [[ "$symlink_ok" == *"SYMLINK_FAILED"* ]]; then
        verification_failed=true
    fi

    echo ""
    echo "=== Model Directory ==="
    local models_ok
    models_ok=$(ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        if [ -d '$REMOTE_DIR/current/models' ]; then
            # Check for timestamped model directories (multi-target format)
            MODEL_RUNS=\$(find '$REMOTE_DIR/current/models' -maxdepth 1 -type d -name '20*' 2>/dev/null | wc -l)
            if [ \$MODEL_RUNS -gt 0 ]; then
                LATEST=\$(find '$REMOTE_DIR/current/models' -maxdepth 1 -type d -name '20*' | sort | tail -1)
                ITEM_COUNT=\$(find \"\$LATEST\" -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | wc -l)
                echo \"Multi-target model runs: \$MODEL_RUNS\"
                echo \"Latest run: \$(basename \$LATEST)\"
                echo \"Items in latest run: \$ITEM_COUNT\"
                echo 'MODELS_OK'
            else
                echo 'WARNING: No timestamped model runs found'
                # Check for legacy item-based models
                LEGACY_ITEMS=\$(find '$REMOTE_DIR/current/models' -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | wc -l)
                if [ \$LEGACY_ITEMS -gt 0 ]; then
                    echo \"Legacy model format detected: \$LEGACY_ITEMS items\"
                    echo 'MODELS_OK'
                else
                    echo 'ERROR: No models found'
                    echo 'MODELS_FAILED'
                fi
            fi
        else
            echo 'ERROR: Models directory not found'
            echo 'MODELS_FAILED'
        fi
    ")
    echo "$models_ok"
    if [[ "$models_ok" == *"MODELS_FAILED"* ]]; then
        verification_failed=true
    fi

    echo ""
    echo "=== Release Directory ==="
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "ls -la '$REMOTE_DIR/releases' 2>/dev/null | head -10 || echo 'No releases directory'"

    echo ""
    echo "=== Cron Job ==="
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "crontab -l 2>/dev/null | grep gept || echo 'No cron job found'"

    echo ""
    echo "=== Training Scheduler ==="
    ssh -i "$AMPERE_KEY" "$AMPERE_HOST" "
        if systemctl is-enabled gept-training.timer 2>/dev/null | grep -q 'enabled'; then
            echo 'Training timer: ENABLED'
            systemctl status gept-training.timer --no-pager 2>/dev/null | head -5 || true
        else
            echo 'WARNING: Training timer not enabled'
            echo 'Run: sudo systemctl enable --now gept-training.timer'
        fi
    "

    if [ "$verification_failed" = true ]; then
        log_error "Deployment verification FAILED - manual intervention required"
        exit 1
    fi

    log_success "Verification complete"
}

# Main script
main() {
    local action="deploy"
    local rollback_version=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --rollback)
                action="rollback"
                if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                    rollback_version="$2"
                    shift
                fi
                shift
                ;;
            --list)
                action="list"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --keep)
                KEEP_RELEASES="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    check_prerequisites

    case $action in
        deploy)
            pre_deployment_check
            migrate_existing_deployment
            deploy_version
            verify_deployment
            echo ""
            echo "=========================================="
            echo "Deployment Complete!"
            echo "=========================================="
            echo ""
            echo "Next steps:"
            echo "  1. SSH into server: ssh -i .secrets/oracle_key.pem $AMPERE_HOST"
            echo "  2. Test inference: cd $REMOTE_DIR && source venv/bin/activate && python current/run_inference.py --dry-run"
            echo "  3. Monitor logs: tail -f $REMOTE_DIR/logs/inference.log"
            echo ""
            echo "Rollback command:"
            echo "  ./deploy_ampere.sh --rollback"
            echo ""
            ;;
        rollback)
            do_rollback "$rollback_version"
            ;;
        list)
            list_versions
            ;;
    esac
}

main "$@"
