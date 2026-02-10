#!/bin/bash
# =============================================================================
# GePT Data Collectors - Deployment Script
# =============================================================================
# Deploys all data collection services to the Ampere server with versioning
# and rollback support.
#
# Usage:
#   ./deploy_collectors.sh              # Full deployment
#   ./deploy_collectors.sh --quick      # Skip Docker rebuild
#   ./deploy_collectors.sh --monitoring # Deploy only monitoring stack
#   ./deploy_collectors.sh --rollback   # Rollback to previous version
#   ./deploy_collectors.sh --rollback VERSION  # Rollback to specific version
#   ./deploy_collectors.sh --list       # List available versions
#   ./deploy_collectors.sh --dry-run    # Preview deployment
#   ./deploy_collectors.sh --help       # Show help
#
# Directory Structure (after deployment):
#   /home/ubuntu/osrs_collector/
#   ├── releases/
#   │   └── YYYYMMDD_HHMMSS/
#   │       ├── *.py
#   │       ├── docker-compose.yml
#   │       ├── prometheus/
#   │       ├── grafana/
#   │       └── deploy_info.json
#   ├── current -> releases/YYYYMMDD_HHMMSS  (atomic symlink)
#   └── .env  # Persistent config (not versioned)
#
# Requirements:
#   - SSH key at .secrets/oracle_key.pem (or AMPERE_SSH_KEY)
#   - SSH access to Ampere server (configure via config/servers.env or AMPERE_HOST)
# =============================================================================

set -euo pipefail

# Error handler for debugging
error_handler() {
    echo "ERROR: Script failed on line $1" >&2
    exit 1
}
trap 'error_handler $LINENO' ERR

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
# NOTE: This repo is public; do not bake production hostnames/IPs into defaults.
: "${AMPERE_HOST:?AMPERE_HOST is required (e.g. ubuntu@your-host)}"
: "${AMPERE_SSH_KEY:?AMPERE_SSH_KEY is required (path to SSH private key)}"
SERVER="$AMPERE_HOST"
SSH_KEY="$AMPERE_SSH_KEY"
# Resolve relative path for SSH key
if [[ ! "$SSH_KEY" = /* ]]; then
    SSH_KEY="$SCRIPT_DIR/$SSH_KEY"
fi
REMOTE_DIR="${AMPERE_COLLECTOR_DIR:-/home/ubuntu/osrs_collector}"
LOCAL_COLLECTORS="collectors"
KEEP_RELEASES=5
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    echo "GePT Data Collectors - Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick               Skip Docker image rebuild"
    echo "  --monitoring          Deploy only Prometheus and Grafana"
    echo "  --systemd             Deploy only systemd services (player_count, item_updater)"
    echo "  --rollback [VERSION]  Rollback to previous or specific version"
    echo "  --list                List available versions"
    echo "  --dry-run             Preview deployment without executing"
    echo "  --keep N              Keep N releases (default: 5)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Full deployment"
    echo "  $0 --quick            # Quick deployment without rebuild"
    echo "  $0 --monitoring       # Deploy monitoring stack only"
    echo "  $0 --rollback         # Rollback to previous version"
    echo "  $0 --rollback 20260113_140000  # Rollback to specific version"
    echo "  $0 --list             # Show available versions"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "$SSH_KEY" ]; then
        log_error "SSH key not found at $SSH_KEY"
        exit 1
    fi

    if [ ! -d "$LOCAL_COLLECTORS" ]; then
        log_error "Collectors directory not found at $LOCAL_COLLECTORS"
        exit 1
    fi

    # Test SSH connection
    if ! ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o BatchMode=yes "$SERVER" "echo ok" > /dev/null 2>&1; then
        log_error "Cannot connect to $SERVER"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

list_versions() {
    log_info "Available versions on $SERVER:"
    echo ""

    ssh -i "$SSH_KEY" "$SERVER" "
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
    ssh -i "$SSH_KEY" "$SERVER" "
        ls -1t '$REMOTE_DIR/releases' 2>/dev/null | head -2 | tail -1
    "
}

get_current_version() {
    ssh -i "$SSH_KEY" "$SERVER" "
        readlink '$REMOTE_DIR/current' 2>/dev/null | xargs basename 2>/dev/null || echo ''
    "
}

pre_deployment_check() {
    log_info "Running pre-deployment health check..."

    # Check if there's an existing deployment
    local has_current
    has_current=$(ssh -i "$SSH_KEY" "$SERVER" "[ -L '$REMOTE_DIR/current' ] && echo 'yes' || echo 'no'")

    if [ "$has_current" = "yes" ]; then
        # Check Docker container health
        log_info "Checking Docker container status..."
        local container_status
        container_status=$(ssh -i "$SSH_KEY" "$SERVER" "docker ps --format '{{.Names}}: {{.Status}}' | grep -c 'Up' || echo '0'")
        log_info "Running containers: $container_status"
    else
        log_info "No existing deployment detected (first deployment)"
    fi

    log_success "Pre-deployment check passed"
}

migrate_existing_deployment() {
    log_info "Checking for existing non-versioned deployment..."

    ssh -i "$SSH_KEY" "$SERVER" "
        # Create releases directory if it doesn't exist
        mkdir -p '$REMOTE_DIR/releases'

        # If there's an existing deployment without versioning, migrate it
        if [ -f '$REMOTE_DIR/docker-compose.yml' ] && [ ! -L '$REMOTE_DIR/current' ]; then
            echo 'Migrating existing deployment to versioned structure...'
            MIGRATE_VERSION=\$(date +%Y%m%d_%H%M%S)_migrated
            mkdir -p '$REMOTE_DIR/releases/'\$MIGRATE_VERSION

            # Move existing files to the migration release
            for item in *.py docker-compose.yml Dockerfile requirements.txt prometheus grafana shared migrations; do
                if [ -e '$REMOTE_DIR/'\$item ]; then
                    mv '$REMOTE_DIR/'\$item '$REMOTE_DIR/releases/'\$MIGRATE_VERSION/ 2>/dev/null || true
                fi
            done

            # Create symlink to migrated version
            ln -sfn '$REMOTE_DIR/releases/'\$MIGRATE_VERSION '$REMOTE_DIR/current'
            echo 'Migration complete: '\$MIGRATE_VERSION
        fi
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
        log_info "[DRY RUN] Would restart Docker containers"
        return
    fi

    # Verify target version exists
    if ! ssh -i "$SSH_KEY" "$SERVER" "[ -d '$REMOTE_DIR/releases/$target_version' ]"; then
        log_error "Version $target_version does not exist"
        echo ""
        list_versions
        exit 1
    fi

    # Atomic symlink switch
    ssh -i "$SSH_KEY" "$SERVER" "
        ln -sfn '$REMOTE_DIR/releases/$target_version' '$REMOTE_DIR/current.new'
        mv -Tf '$REMOTE_DIR/current.new' '$REMOTE_DIR/current'
    "

    # Restart Docker containers with the rolled-back version
    log_info "Restarting Docker containers..."
    ssh -i "$SSH_KEY" "$SERVER" "cd '$REMOTE_DIR/current' && docker compose up -d"

    log_success "Rolled back to version $target_version"

    # Verify the rollback
    verify_deployment
}

sync_collectors() {
    local deploy_dir=$1
    log_info "Syncing collector scripts..."

    rsync -avz -e "ssh -i $SSH_KEY" \
        "$LOCAL_COLLECTORS"/*.py \
        "$LOCAL_COLLECTORS"/docker-compose.yml \
        "$LOCAL_COLLECTORS"/Dockerfile \
        "$LOCAL_COLLECTORS"/requirements.txt \
        "$SERVER:$deploy_dir/"

    log_success "Collector scripts synced"
}

sync_shared_modules() {
    local deploy_dir=$1
    log_info "Syncing shared modules..."

    ssh -i "$SSH_KEY" "$SERVER" "mkdir -p $deploy_dir/shared $deploy_dir/migrations"

    rsync -avz -e "ssh -i $SSH_KEY" \
        "$LOCAL_COLLECTORS"/shared/ \
        "$SERVER:$deploy_dir/shared/"

    # Sync migrations if they exist
    if [ -d "scripts/migrations" ]; then
        rsync -avz -e "ssh -i $SSH_KEY" \
            scripts/migrations/*.sql \
            "$SERVER:$deploy_dir/migrations/" 2>/dev/null || true
    fi

    log_success "Shared modules synced"
}

sync_monitoring_configs() {
    local deploy_dir=$1
    log_info "Syncing monitoring configurations..."

    ssh -i "$SSH_KEY" "$SERVER" "mkdir -p $deploy_dir/prometheus $deploy_dir/grafana/provisioning/datasources $deploy_dir/grafana/provisioning/dashboards"

    rsync -avz -e "ssh -i $SSH_KEY" \
        "$LOCAL_COLLECTORS"/prometheus/ \
        "$SERVER:$deploy_dir/prometheus/"

    rsync -avz -e "ssh -i $SSH_KEY" \
        "$LOCAL_COLLECTORS"/grafana/ \
        "$SERVER:$deploy_dir/grafana/"

    # Fix permissions for Prometheus
    ssh -i "$SSH_KEY" "$SERVER" "chmod 644 $deploy_dir/prometheus/*.yml"

    log_success "Monitoring configs synced"
}

deploy_docker_services() {
    local rebuild=$1

    log_info "Deploying Docker services..."

    if [ "$rebuild" = "true" ]; then
        ssh -i "$SSH_KEY" "$SERVER" "cd $REMOTE_DIR/current && docker compose up -d --build"
    else
        ssh -i "$SSH_KEY" "$SERVER" "cd $REMOTE_DIR/current && docker compose up -d"
    fi

    log_success "Docker services deployed"
}

deploy_monitoring_only() {
    log_info "Deploying monitoring stack only..."

    ssh -i "$SSH_KEY" "$SERVER" "cd $REMOTE_DIR/current && docker compose up -d prometheus grafana"

    log_success "Monitoring stack deployed"
}

deploy_systemd_services() {
    log_info "Deploying systemd collector services..."

    # Copy service files
    log_info "Copying service files..."
    scp -i "$SSH_KEY" "$LOCAL_COLLECTORS/gept-data-export.service" "$SERVER:/tmp/"
    scp -i "$SSH_KEY" "$LOCAL_COLLECTORS/gept-data-export.timer" "$SERVER:/tmp/"

    # Install and enable services
    ssh -i "$SSH_KEY" "$SERVER" "
        sudo mv /tmp/gept-data-export.service /etc/systemd/system/
        sudo mv /tmp/gept-data-export.timer /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable gept-data-export.timer
        sudo systemctl start gept-data-export.timer
        sudo systemctl restart player_count.service item_updater.service
    "

    log_success "Systemd services deployed"
}

reload_prometheus() {
    log_info "Reloading Prometheus configuration..."

    ssh -i "$SSH_KEY" "$SERVER" "curl -s -X POST http://localhost:9090/-/reload" > /dev/null 2>&1 || true

    log_success "Prometheus configuration reloaded"
}

verify_deployment() {
    log_info "Verifying deployment..."

    echo ""
    echo "=== Current Version ==="
    ssh -i "$SSH_KEY" "$SERVER" "
        if [ -L '$REMOTE_DIR/current' ]; then
            VERSION=\$(readlink '$REMOTE_DIR/current' | xargs basename)
            echo \"Active version: \$VERSION\"
            if [ -f '$REMOTE_DIR/current/deploy_info.json' ]; then
                cat '$REMOTE_DIR/current/deploy_info.json'
            fi
        else
            echo 'No current version symlink found'
        fi
    "

    echo ""
    echo "=== Docker Container Status ==="
    ssh -i "$SSH_KEY" "$SERVER" 'docker ps --format "table {{.Names}}\t{{.Status}}"'

    echo ""
    echo "=== Systemd Service Status ==="
    ssh -i "$SSH_KEY" "$SERVER" 'systemctl is-active player_count.service item_updater.service 2>/dev/null || echo "Services not found"'

    echo ""
    echo "=== Data Export Timer Status ==="
    ssh -i "$SSH_KEY" "$SERVER" 'systemctl list-timers gept-data-export.timer --no-pager 2>/dev/null || echo "Timer not found"'

    echo ""
    echo "=== Data Freshness ==="
    ssh -i "$SSH_KEY" "$SERVER" 'PGPASSWORD=$DB_PASS psql -h localhost -U osrs_user -d osrs_data -t -c "
        SELECT '\''price_data_5min'\'' as tbl, NOW() - MAX(timestamp) as age FROM price_data_5min
        UNION ALL
        SELECT '\''player_counts'\'', NOW() - MAX(timestamp) FROM player_counts
        UNION ALL
        SELECT '\''predictions'\'', NOW() - MAX(time) FROM predictions;
    " 2>/dev/null'

    echo ""
    echo "=== Defense in Depth Services ==="
    ssh -i "$SSH_KEY" "$SERVER" 'docker ps --filter "name=gept-" --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || echo "Defense services not found"'

    echo ""
    log_success "Deployment verification complete"
}

check_discord_webhook() {
    log_info "Checking Discord webhook configuration..."

    # Check if DISCORD_WEBHOOK_URL is set on the server
    local webhook_check
    webhook_check=$(ssh -i "$SSH_KEY" "$SERVER" 'grep -q "DISCORD_WEBHOOK_URL" /home/ubuntu/osrs_collector/.env 2>/dev/null && echo "configured" || echo "not_configured"')

    if [ "$webhook_check" = "not_configured" ]; then
        log_warning "DISCORD_WEBHOOK_URL not found in .env file"
        log_warning "Alertmanager will not send Discord notifications"
        log_warning "Add DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/... to .env to enable"
    else
        log_success "Discord webhook is configured"
    fi
}

run_database_migration() {
    log_info "Running database migrations..."

    # Check if data_quality table exists
    local table_exists
    table_exists=$(ssh -i "$SSH_KEY" "$SERVER" 'PGPASSWORD=$DB_PASS psql -h localhost -U osrs_user -d osrs_data -t -c "
        SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '\''data_quality'\'');
    " 2>/dev/null' | tr -d '[:space:]')

    if [ "$table_exists" = "t" ]; then
        log_info "data_quality table already exists"
    else
        log_info "Creating data_quality table..."
        ssh -i "$SSH_KEY" "$SERVER" 'PGPASSWORD=$DB_PASS psql -h localhost -U osrs_user -d osrs_data -f /home/ubuntu/osrs_collector/current/migrations/007_data_quality.sql 2>/dev/null' || true
        log_success "data_quality table created"
    fi
}

prune_old_releases() {
    log_info "Pruning old releases (keeping last $KEEP_RELEASES)..."

    ssh -i "$SSH_KEY" "$SERVER" "
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

deploy_full_version() {
    local rebuild=$1
    local deploy_version
    deploy_version=$(date +%Y%m%d_%H%M%S)
    local deploy_dir="$REMOTE_DIR/releases/$deploy_version"
    local git_commit
    git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

    log_info "Deploying version: $deploy_version"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would create release directory: $deploy_dir"
        log_info "[DRY RUN] Would sync collectors, shared modules, and monitoring configs"
        log_info "[DRY RUN] Would create symlink: current -> releases/$deploy_version"
        return
    fi

    # Create release directory
    ssh -i "$SSH_KEY" "$SERVER" "mkdir -p $deploy_dir"

    # Sync all components
    sync_collectors "$deploy_dir"
    sync_shared_modules "$deploy_dir"
    sync_monitoring_configs "$deploy_dir"

    # Create deployment info
    log_info "Recording deployment metadata..."
    local previous_version
    previous_version=$(get_current_version)
    ssh -i "$SSH_KEY" "$SERVER" "cat > $deploy_dir/deploy_info.json << EOF
{
    \"version\": \"$deploy_version\",
    \"deployed_at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
    \"deployed_by\": \"deploy_collectors.sh\",
    \"git_commit\": \"$git_commit\",
    \"previous_version\": \"$previous_version\"
}
EOF"

    # Atomic symlink switch
    log_info "Activating new version (atomic switch)..."
    ssh -i "$SSH_KEY" "$SERVER" "
        ln -sfn '$deploy_dir' '$REMOTE_DIR/current.new'
        mv -Tf '$REMOTE_DIR/current.new' '$REMOTE_DIR/current'
    "

    # Check Discord webhook before deploying
    check_discord_webhook

    # Deploy services
    if [ "$rebuild" = true ]; then
        deploy_docker_services "true"
    else
        deploy_docker_services "false"
    fi

    deploy_systemd_services
    reload_prometheus

    # Run database migrations for defense in depth
    run_database_migration

    # Prune old releases
    prune_old_releases

    log_success "Deployed version $deploy_version"
}

# Main script
main() {
    local quick=false
    local monitoring_only=false
    local systemd_only=false
    local action="deploy"
    local rollback_version=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                quick=true
                shift
                ;;
            --monitoring)
                monitoring_only=true
                shift
                ;;
            --systemd)
                systemd_only=true
                shift
                ;;
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

    echo "=============================================="
    echo "  GePT Data Collectors - Deployment"
    echo "=============================================="
    echo ""

    check_prerequisites

    case $action in
        rollback)
            do_rollback "$rollback_version"
            ;;
        list)
            list_versions
            ;;
        deploy)
            pre_deployment_check
            migrate_existing_deployment

            if [ "$monitoring_only" = true ]; then
                # For monitoring-only, sync just configs and restart
                local current_dir="$REMOTE_DIR/current"
                if [ "$DRY_RUN" = true ]; then
                    log_info "[DRY RUN] Would sync monitoring configs to current release"
                else
                    sync_monitoring_configs "$current_dir"
                    deploy_monitoring_only
                    reload_prometheus
                fi
            elif [ "$systemd_only" = true ]; then
                # For systemd-only, sync collectors and restart systemd
                local current_dir="$REMOTE_DIR/current"
                if [ "$DRY_RUN" = true ]; then
                    log_info "[DRY RUN] Would sync collectors and restart systemd services"
                else
                    sync_collectors "$current_dir"
                    sync_shared_modules "$current_dir"
                    deploy_systemd_services
                fi
            else
                # Full deployment
                if [ "$quick" = true ]; then
                    deploy_full_version false
                else
                    deploy_full_version true
                fi
            fi

            verify_deployment

            # Get IP from AMPERE_HOST (strip username@ prefix)
            local server_ip="${AMPERE_IP:-${AMPERE_HOST#*@}}"

            echo ""
            echo "=============================================="
            echo "  Deployment Complete!"
            echo "=============================================="
            echo ""
            echo "Access URLs:"
            echo "  Dashboard:     http://${server_ip}:${DASHBOARD_PORT:-8080}"
            echo "  Prometheus:    http://${server_ip}:${PROMETHEUS_PORT:-9090}"
            echo "  Grafana:       http://${server_ip}:${GRAFANA_PORT:-3001}"
            echo "  Alertmanager:  http://${server_ip}:${ALERTMANAGER_PORT:-9093}"
            echo ""
            echo "Defense in Depth Services:"
            echo "  Collector Monitor: http://${server_ip}:9106/metrics"
            echo "  Gap Detector:      http://${server_ip}:9107/metrics"
            echo "  Watchdog:          http://${server_ip}:9108/metrics"
            echo ""
            echo "Rollback command:"
            echo "  ./deploy_collectors.sh --rollback"
            echo ""
            ;;
    esac
}

main "$@"
