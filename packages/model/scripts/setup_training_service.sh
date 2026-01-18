#!/bin/bash
# =============================================================================
# GePT Training Service Setup
# =============================================================================
# Installs and manages the continuous training systemd service on Hydra.
#
# Usage:
#   ./scripts/setup_training_service.sh install   # Install service
#   ./scripts/setup_training_service.sh start     # Start service
#   ./scripts/setup_training_service.sh stop      # Stop service
#   ./scripts/setup_training_service.sh status    # Check status
#   ./scripts/setup_training_service.sh logs      # View logs
#   ./scripts/setup_training_service.sh uninstall # Remove service
# =============================================================================

set -e

SERVICE_NAME="gept-training"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOCAL_SERVICE="config/systemd/${SERVICE_NAME}.service"
LOG_FILE="/home/ubuntu/gept/logs/training.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

install_service() {
    check_root

    log_info "Installing ${SERVICE_NAME} service..."

    # Create logs directory
    mkdir -p /home/ubuntu/gept/logs
    chown ubuntu:ubuntu /home/ubuntu/gept/logs

    # Copy service file
    if [ -f "$LOCAL_SERVICE" ]; then
        cp "$LOCAL_SERVICE" "$SERVICE_FILE"
        log_info "Copied service file to $SERVICE_FILE"
    else
        log_error "Service file not found: $LOCAL_SERVICE"
        exit 1
    fi

    # Reload systemd
    systemctl daemon-reload
    log_info "Reloaded systemd daemon"

    # Enable service
    systemctl enable "$SERVICE_NAME"
    log_info "Enabled ${SERVICE_NAME} service"

    echo ""
    log_info "Service installed successfully!"
    log_info "Start with: sudo systemctl start ${SERVICE_NAME}"
}

uninstall_service() {
    check_root

    log_info "Uninstalling ${SERVICE_NAME} service..."

    # Stop if running
    systemctl stop "$SERVICE_NAME" 2>/dev/null || true

    # Disable
    systemctl disable "$SERVICE_NAME" 2>/dev/null || true

    # Remove service file
    rm -f "$SERVICE_FILE"

    # Reload systemd
    systemctl daemon-reload

    log_info "Service uninstalled"
}

start_service() {
    check_root
    log_info "Starting ${SERVICE_NAME}..."
    systemctl start "$SERVICE_NAME"
    sleep 2
    systemctl status "$SERVICE_NAME" --no-pager
}

stop_service() {
    check_root
    log_info "Stopping ${SERVICE_NAME}..."
    systemctl stop "$SERVICE_NAME"
    log_info "Service stopped"
}

restart_service() {
    check_root
    log_info "Restarting ${SERVICE_NAME}..."
    systemctl restart "$SERVICE_NAME"
    sleep 2
    systemctl status "$SERVICE_NAME" --no-pager
}

show_status() {
    echo ""
    echo "=== Service Status ==="
    systemctl status "$SERVICE_NAME" --no-pager 2>/dev/null || echo "Service not installed"

    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo "GPU not available"

    echo ""
    echo "=== Recent Training Activity ==="
    if [ -f "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE"
    else
        echo "No log file found"
    fi
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        # Try journalctl
        journalctl -u "$SERVICE_NAME" -f
    fi
}

# Main
case "${1:-}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "GePT Training Service Manager"
        echo ""
        echo "Usage: $0 {install|uninstall|start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  install   - Install the systemd service"
        echo "  uninstall - Remove the systemd service"
        echo "  start     - Start the training service"
        echo "  stop      - Stop the training service"
        echo "  restart   - Restart the training service"
        echo "  status    - Show service and GPU status"
        echo "  logs      - Follow the training logs"
        exit 1
        ;;
esac
