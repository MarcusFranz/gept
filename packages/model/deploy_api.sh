#!/bin/bash
# =============================================================================
# Deploy GePT Public API to Ampere Server
# =============================================================================
# Sets up the FastAPI server for webapp integration with gept.gg
#
# Usage:
#   ./deploy_api.sh              # Deploy and start API
#   ./deploy_api.sh --status     # Check API status
#   ./deploy_api.sh --firewall   # Configure firewall only
#   ./deploy_api.sh --restart    # Restart API service
#
# Requirements:
#   - SSH access to Ampere server
#   - .secrets/oracle_key.pem SSH key
# =============================================================================

set -euo pipefail

# Determine script directory
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source central server configuration
if [ -f "$LOCAL_DIR/config/servers.env" ]; then
    source "$LOCAL_DIR/config/servers.env"
fi

# Configuration
AMPERE_KEY="${AMPERE_SSH_KEY:-$LOCAL_DIR/.secrets/oracle_key.pem}"
if [[ ! "$AMPERE_KEY" = /* ]]; then
    AMPERE_KEY="$LOCAL_DIR/$AMPERE_KEY"
fi
REMOTE_DIR="${AMPERE_GEPT_DIR:-/home/ubuntu/gept}"
API_PORT="${API_PORT:-8000}"

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

ssh_cmd() {
    ssh -i "$AMPERE_KEY" -o StrictHostKeyChecking=no "$AMPERE_HOST" "$@"
}

scp_cmd() {
    scp -i "$AMPERE_KEY" -o StrictHostKeyChecking=no "$@"
}

show_help() {
    echo "GePT Public API - Deployment Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --status      Check API status and health"
    echo "  --firewall    Configure firewall rules only"
    echo "  --restart     Restart API service"
    echo "  --logs        Show recent API logs"
    echo "  --help        Show this help message"
    echo ""
    echo "API will be available at: http://${AMPERE_IP}:${API_PORT}"
}

check_status() {
    log_info "Checking API status..."

    # Check service status
    ssh_cmd "sudo systemctl status gept-api --no-pager" || true

    echo ""
    log_info "Testing health endpoint..."

    # Test health endpoint
    if curl -s --max-time 5 "http://${AMPERE_IP}:${API_PORT}/health" | python3 -m json.tool 2>/dev/null; then
        log_success "API is healthy!"
    else
        log_error "API health check failed"
    fi
}

configure_firewall() {
    log_info "Configuring firewall for port ${API_PORT}..."

    ssh_cmd << 'FIREWALL_EOF'
# Check if iptables rule exists
if ! sudo iptables -L INPUT -n | grep -q "dpt:8000"; then
    echo "Adding iptables rule for port 8000..."
    sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT

    # Save iptables rules (persist across reboot)
    if command -v netfilter-persistent &> /dev/null; then
        sudo netfilter-persistent save
    elif [ -f /etc/iptables/rules.v4 ]; then
        sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null
    fi
    echo "Firewall rule added and saved"
else
    echo "Firewall rule for port 8000 already exists"
fi

# Also check Oracle Cloud security list reminder
echo ""
echo "============================================"
echo "IMPORTANT: Oracle Cloud Security List"
echo "============================================"
echo "Ensure port 8000 is open in your Oracle Cloud"
echo "VCN security list / Network Security Group:"
echo ""
echo "1. Go to Oracle Cloud Console"
echo "2. Networking > Virtual Cloud Networks"
echo "3. Select your VCN > Security Lists"
echo "4. Add Ingress Rule:"
echo "   - Source: 0.0.0.0/0"
echo "   - Protocol: TCP"
echo "   - Destination Port: 8000"
echo "============================================"
FIREWALL_EOF

    log_success "Firewall configured"
}

show_logs() {
    log_info "Recent API logs:"
    ssh_cmd "sudo journalctl -u gept-api -n 50 --no-pager"
}

restart_service() {
    log_info "Restarting API service..."
    ssh_cmd "sudo systemctl restart gept-api"
    sleep 2
    check_status
}

deploy() {
    log_info "Deploying GePT API to ${AMPERE_HOST}..."

    # Step 1: Copy API files
    log_info "Step 1/5: Copying API files..."
    scp_cmd -r "$LOCAL_DIR/api" "${AMPERE_HOST}:${REMOTE_DIR}/"

    # Step 2: Install dependencies
    log_info "Step 2/5: Installing API dependencies..."
    ssh_cmd << 'DEPS_EOF'
cd /home/ubuntu/gept
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate gept
pip install -q fastapi uvicorn[standard] pydantic
DEPS_EOF

    # Step 3: Copy systemd service
    log_info "Step 3/5: Setting up systemd service..."
    scp_cmd "$LOCAL_DIR/config/systemd/gept-api.service" "${AMPERE_HOST}:/tmp/"
    ssh_cmd << 'SERVICE_EOF'
sudo mv /tmp/gept-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gept-api
SERVICE_EOF

    # Step 4: Configure firewall
    log_info "Step 4/5: Configuring firewall..."
    configure_firewall

    # Step 5: Start/restart service
    log_info "Step 5/5: Starting API service..."
    ssh_cmd "sudo systemctl restart gept-api"

    # Wait for service to start
    sleep 3

    # Verify deployment
    log_info "Verifying deployment..."
    check_status

    echo ""
    log_success "============================================"
    log_success "API Deployment Complete!"
    log_success "============================================"
    echo ""
    echo "Public URL: http://${AMPERE_IP}:${API_PORT}"
    echo "Health:     http://${AMPERE_IP}:${API_PORT}/health"
    echo "Docs:       http://${AMPERE_IP}:${API_PORT}/docs"
    echo ""
    echo "Endpoints:"
    echo "  GET /health              - Health check"
    echo "  GET /api/predictions     - Latest predictions"
    echo "  GET /api/opportunities   - Top trading opportunities"
    echo "  GET /api/items/{id}      - Item metadata"
    echo "  GET /api/items/search    - Search items by name"
    echo "  GET /api/stats           - Summary statistics"
    echo ""
    echo "CORS configured for:"
    echo "  - https://gept.gg"
    echo "  - https://www.gept.gg"
    echo "  - http://localhost:3000 (dev)"
    echo ""
}

# Parse arguments
case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --status)
        check_status
        exit 0
        ;;
    --firewall)
        configure_firewall
        exit 0
        ;;
    --restart)
        restart_service
        exit 0
        ;;
    --logs)
        show_logs
        exit 0
        ;;
    "")
        deploy
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
