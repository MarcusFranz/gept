#!/bin/bash
# =============================================================================
# Migrate Data to Block Volume - Boot Volume at 82%
# =============================================================================
# Issue: #152
#
# This script moves large data directories from boot volume to /data (block vol)
# and creates symlinks to maintain compatibility.
#
# Run on Ampere server only! Requires sudo access.
#
# IMPORTANT: Run during maintenance window. Services will be stopped temporarily.
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Verify we're on Ampere
if [[ ! -d "/data" ]]; then
    error "Block volume /data not found. This script must run on Ampere."
fi

# Check disk usage before
log "Current disk usage:"
df -h / /data

# Confirm before proceeding
echo ""
warn "This script will stop PostgreSQL and Docker temporarily."
warn "Press Ctrl+C to cancel, or Enter to continue..."
read

# 1. Move GePT models to block volume
log "Step 1: Moving GePT models to block volume..."
if [[ -d "/home/ubuntu/gept/models" && ! -L "/home/ubuntu/gept/models" ]]; then
    sudo mkdir -p /data/gept
    sudo mv /home/ubuntu/gept/models /data/gept/models
    ln -s /data/gept/models /home/ubuntu/gept/models
    log "Models moved successfully"
else
    warn "Models already symlinked or not found, skipping"
fi

# 2. Move collector data to block volume
log "Step 2: Moving collector data to block volume..."
if [[ -d "/home/ubuntu/osrs_collector/data" && ! -L "/home/ubuntu/osrs_collector/data" ]]; then
    sudo mkdir -p /data/osrs_collector
    sudo mv /home/ubuntu/osrs_collector/data /data/osrs_collector/data
    ln -s /data/osrs_collector/data /home/ubuntu/osrs_collector/data
    log "Collector data moved successfully"
else
    warn "Collector data already symlinked or not found, skipping"
fi

# 3. Move PostgreSQL to block volume (requires stopping postgres)
log "Step 3: Moving PostgreSQL to block volume..."
PG_DATA_DIR="/var/lib/postgresql/14/main"
if [[ -d "$PG_DATA_DIR" && ! -L "$PG_DATA_DIR" ]]; then
    log "Stopping PostgreSQL..."
    sudo systemctl stop postgresql

    sudo mkdir -p /data/postgresql/14
    sudo mv "$PG_DATA_DIR" /data/postgresql/14/main
    sudo ln -s /data/postgresql/14/main "$PG_DATA_DIR"
    sudo chown -h postgres:postgres "$PG_DATA_DIR"

    log "Starting PostgreSQL..."
    sudo systemctl start postgresql

    # Verify postgres is working
    if sudo -u postgres psql -c "SELECT 1" &>/dev/null; then
        log "PostgreSQL moved and verified successfully"
    else
        error "PostgreSQL failed to start after migration!"
    fi
else
    warn "PostgreSQL already symlinked or not found, skipping"
fi

# 4. Move Docker to block volume (requires stopping docker)
log "Step 4: Moving Docker to block volume..."
DOCKER_DIR="/var/lib/docker"
if [[ -d "$DOCKER_DIR" && ! -L "$DOCKER_DIR" ]]; then
    log "Stopping Docker..."
    sudo systemctl stop docker

    sudo mkdir -p /data/docker
    sudo rsync -aP "$DOCKER_DIR/" /data/docker/
    sudo rm -rf "$DOCKER_DIR"
    sudo ln -s /data/docker "$DOCKER_DIR"

    log "Starting Docker..."
    sudo systemctl start docker

    # Verify docker is working
    if docker ps &>/dev/null; then
        log "Docker moved and verified successfully"
    else
        error "Docker failed to start after migration!"
    fi
else
    warn "Docker already symlinked or not found, skipping"
fi

# Check disk usage after
echo ""
log "Migration complete! New disk usage:"
df -h / /data

echo ""
log "Verification checklist:"
echo "  - [ ] PostgreSQL responding: sudo -u postgres psql -c 'SELECT 1'"
echo "  - [ ] Docker containers running: docker ps"
echo "  - [ ] Inference working: tail -f /home/ubuntu/gept/logs/inference.log"
echo ""
log "Done!"
