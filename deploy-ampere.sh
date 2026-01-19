#!/bin/bash
# =============================================================================
# GePT Unified Deployment - Ampere Server
# =============================================================================
# Single command to deploy engine + model + collectors to Ampere.
# Idempotent: running twice produces same result.
# Atomic: symlink swap for instant rollback.
#
# Usage:
#   ./deploy-ampere.sh              # Full deployment
#   ./deploy-ampere.sh --quick      # Skip venv rebuild
#   ./deploy-ampere.sh --rollback   # Rollback to previous version
#   ./deploy-ampere.sh --status     # Check current status
#   ./deploy-ampere.sh --help       # Show help
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source server config
if [ -f "$SCRIPT_DIR/packages/model/config/servers.env" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/packages/model/config/servers.env"
fi

# Configuration
SERVER="${AMPERE_HOST:-ubuntu@150.136.170.128}"
SSH_KEY="${AMPERE_SSH_KEY:-.secrets/oracle_key.pem}"
if [[ ! "$SSH_KEY" = /* ]]; then
    SSH_KEY="$SCRIPT_DIR/$SSH_KEY"
fi
REMOTE_DIR="/home/ubuntu/gept"
KEEP_RELEASES=3
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ssh_cmd() { ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$SERVER" "$@"; }

show_help() {
    cat << 'EOF'
GePT Unified Deployment - Ampere Server

Usage: ./deploy-ampere.sh [OPTIONS]

Options:
  --quick         Skip venv/dependency rebuild
  --rollback      Rollback to previous version
  --status        Show current deployment status
  --dry-run       Preview deployment without executing
  --help          Show this help

Examples:
  ./deploy-ampere.sh              # Full deployment
  ./deploy-ampere.sh --quick      # Quick deploy (deps unchanged)
  ./deploy-ampere.sh --rollback   # Rollback to previous
  ./deploy-ampere.sh --status     # Check what's running
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "$SSH_KEY" ]; then
        log_error "SSH key not found: $SSH_KEY"
        exit 1
    fi

    if ! ssh_cmd "echo ok" > /dev/null 2>&1; then
        log_error "Cannot connect to $SERVER"
        exit 1
    fi

    log_success "Prerequisites OK"
}

show_status() {
    log_info "Current deployment status on Ampere..."

    ssh_cmd << 'EOF'
echo "=== Current Version ==="
if [ -L /home/ubuntu/gept/current ]; then
    VERSION=$(readlink /home/ubuntu/gept/current | xargs basename)
    echo "Active version: $VERSION"
    if [ -f /home/ubuntu/gept/current/deploy_info.json ]; then
        cat /home/ubuntu/gept/current/deploy_info.json
    fi
else
    echo "No current symlink found"
fi

echo ""
echo "=== Systemd Services ==="
systemctl list-units --type=service --state=running 2>/dev/null | grep -E 'gept|recommendation' || echo "No GePT services running"

echo ""
echo "=== Systemd Timers ==="
systemctl list-timers 2>/dev/null | grep gept || echo "No GePT timers"

echo ""
echo "=== Data Freshness ==="
PGPASSWORD=$DB_PASS psql -h localhost -U osrs_user -d osrs_data -t -c "
    SELECT 'price_data_5min' as tbl, NOW() - MAX(timestamp) as age FROM price_data_5min
    UNION ALL
    SELECT 'predictions', NOW() - MAX(time) FROM predictions
    UNION ALL
    SELECT 'player_counts', NOW() - MAX(timestamp) FROM player_counts;
" 2>/dev/null || echo "DB query failed"

echo ""
echo "=== Engine Health ==="
curl -s http://localhost:8000/api/v1/health 2>/dev/null || echo "Engine not responding"
EOF
}

deploy() {
    local quick=$1
    local version
    version=$(date +%Y%m%d_%H%M%S)
    local deploy_dir="$REMOTE_DIR/releases/$version"
    local git_commit
    git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    log_info "Deploying version: $version (commit: $git_commit)"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would deploy to $deploy_dir"
        return
    fi

    # Step 1: Create release directory structure
    log_info "Step 1/7: Creating release directory..."
    ssh_cmd "mkdir -p $deploy_dir/packages $deploy_dir/systemd"

    # Step 2: Sync packages
    log_info "Step 2/7: Syncing packages..."

    # Engine package
    rsync -avz --delete -e "ssh -i $SSH_KEY" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='*.egg-info' \
        "$SCRIPT_DIR/packages/engine/" "$SERVER:$deploy_dir/packages/engine/"

    # Model package (exclude cloud training and Docker collectors)
    rsync -avz --delete -e "ssh -i $SSH_KEY" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='*.egg-info' \
        --exclude='cloud' \
        "$SCRIPT_DIR/packages/model/" "$SERVER:$deploy_dir/packages/model/"

    # Shared package
    rsync -avz --delete -e "ssh -i $SSH_KEY" \
        --exclude='__pycache__' \
        "$SCRIPT_DIR/packages/shared/" "$SERVER:$deploy_dir/packages/shared/"

    # Step 3: Sync systemd files
    log_info "Step 3/7: Syncing systemd service files..."
    rsync -avz -e "ssh -i $SSH_KEY" \
        "$SCRIPT_DIR/infra/systemd/" "$SERVER:$deploy_dir/systemd/"

    # Step 4: Create deployment info
    log_info "Step 4/7: Recording deployment metadata..."
    local prev_version
    prev_version=$(ssh_cmd "readlink $REMOTE_DIR/current 2>/dev/null | xargs basename 2>/dev/null || echo 'none'")
    ssh_cmd "cat > $deploy_dir/deploy_info.json << DEPLOYJSON
{
    \"version\": \"$version\",
    \"git_commit\": \"$git_commit\",
    \"deployed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"previous_version\": \"$prev_version\",
    \"deployed_by\": \"deploy-ampere.sh\"
}
DEPLOYJSON"

    # Step 5: Install dependencies (if not quick)
    if [ "$quick" = false ]; then
        log_info "Step 5/7: Installing dependencies..."
        ssh_cmd << DEPSEOF
cd $deploy_dir
# Create venv if it doesn't exist
if [ ! -d $REMOTE_DIR/venv ]; then
    python3.11 -m venv $REMOTE_DIR/venv
fi
source $REMOTE_DIR/venv/bin/activate
pip install -q --upgrade pip

# Install from both packages
if [ -f packages/engine/requirements.txt ]; then
    pip install -q -r packages/engine/requirements.txt
fi
if [ -f packages/model/requirements.txt ]; then
    pip install -q -r packages/model/requirements.txt
fi
DEPSEOF
    else
        log_info "Step 5/7: Skipping dependencies (--quick)"
    fi

    # Step 6: Atomic symlink switch
    log_info "Step 6/7: Activating new version (atomic switch)..."
    ssh_cmd "ln -sfn $deploy_dir $REMOTE_DIR/current.new && mv -Tf $REMOTE_DIR/current.new $REMOTE_DIR/current"

    # Step 7: Restart services
    log_info "Step 7/7: Installing and restarting services..."
    ssh_cmd << 'RESTARTEOF'
# Install systemd files
sudo cp /home/ubuntu/gept/current/systemd/*.service /etc/systemd/system/ 2>/dev/null || true
sudo cp /home/ubuntu/gept/current/systemd/*.timer /etc/systemd/system/ 2>/dev/null || true
sudo systemctl daemon-reload

# Enable and restart engine service
sudo systemctl enable gept-engine.service 2>/dev/null || true
sudo systemctl restart gept-engine.service

# Enable and start timers
for timer in gept-inference gept-collect-5min gept-collect-1min gept-collect-hourly gept-validation gept-ml-validation; do
    if [ -f /etc/systemd/system/${timer}.timer ]; then
        sudo systemctl enable ${timer}.timer 2>/dev/null || true
        sudo systemctl start ${timer}.timer 2>/dev/null || true
    fi
done

echo "Services installed and started"
RESTARTEOF

    # Prune old releases
    log_info "Pruning old releases (keeping last $KEEP_RELEASES)..."
    ssh_cmd << PRUNEEOF
cd $REMOTE_DIR/releases 2>/dev/null || exit 0
current_version=\$(readlink $REMOTE_DIR/current | xargs basename)
count=0
for release_version in \$(ls -1t); do
    count=\$((count + 1))
    if [ \$count -gt $KEEP_RELEASES ] && [ "\$release_version" != "\$current_version" ]; then
        echo "Removing old release: \$release_version"
        rm -rf "\$release_version"
    fi
done
PRUNEEOF

    log_success "Deployed version $version"
}

rollback() {
    log_info "Rolling back to previous version..."

    local current
    current=$(ssh_cmd "readlink $REMOTE_DIR/current 2>/dev/null | xargs basename 2>/dev/null || echo ''")
    local previous
    previous=$(ssh_cmd "ls -1t $REMOTE_DIR/releases 2>/dev/null | head -2 | tail -1")

    if [ -z "$previous" ] || [ "$previous" = "$current" ]; then
        log_error "No previous version available for rollback"
        exit 1
    fi

    log_info "Rolling back: $current -> $previous"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would rollback to $previous"
        return
    fi

    ssh_cmd "ln -sfn $REMOTE_DIR/releases/$previous $REMOTE_DIR/current.new && mv -Tf $REMOTE_DIR/current.new $REMOTE_DIR/current"
    ssh_cmd "sudo systemctl restart gept-engine.service"

    log_success "Rolled back to $previous"
}

# Main
main() {
    local action="deploy"
    local quick=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick) quick=true; shift ;;
            --rollback) action="rollback"; shift ;;
            --status) action="status"; shift ;;
            --dry-run) DRY_RUN=true; shift ;;
            --help) show_help; exit 0 ;;
            *) log_error "Unknown option: $1"; show_help; exit 1 ;;
        esac
    done

    echo "=============================================="
    echo "  GePT Unified Deployment - Ampere"
    echo "=============================================="
    echo ""

    check_prerequisites

    case $action in
        deploy) deploy "$quick"; show_status ;;
        rollback) rollback; show_status ;;
        status) show_status ;;
    esac

    echo ""
    log_success "Done!"
}

main "$@"
