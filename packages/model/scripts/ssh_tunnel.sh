#!/bin/bash
# SSH Tunnel Helper for Local Dataset Building
#
# This script creates an SSH tunnel to the Ampere server to forward PostgreSQL port 5432.
# This allows you to build datasets locally on your Mac while connecting to the production database.

set -e

AMPERE_IP="150.136.170.128"
SSH_KEY="${HOME}/.ssh/oracle_key.pem"
LOCAL_PORT=5432
REMOTE_PORT=5432

# Function to check if tunnel is already running
check_tunnel() {
    if pgrep -f "ssh.*${LOCAL_PORT}:localhost:${REMOTE_PORT}.*${AMPERE_IP}" > /dev/null; then
        return 0  # Tunnel is running
    else
        return 1  # Tunnel is not running
    fi
}

# Function to start tunnel
start_tunnel() {
    if check_tunnel; then
        echo "✓ SSH tunnel already running"
        return 0
    fi

    echo "Starting SSH tunnel to Ampere..."
    echo "  Local: localhost:${LOCAL_PORT} → Remote: ${AMPERE_IP}:${REMOTE_PORT}"

    # Create tunnel in background
    ssh -i "${SSH_KEY}" -L ${LOCAL_PORT}:localhost:${REMOTE_PORT} ubuntu@${AMPERE_IP} -N -f

    # Wait a moment for connection to establish
    sleep 2

    if check_tunnel; then
        echo "✓ SSH tunnel started successfully"
        echo ""
        echo "You can now build datasets locally with:"
        echo "  docker-compose run --rm dataset-builder python scripts/build_dataset.py --recipe <name>"
        echo ""
        echo "To stop the tunnel:"
        echo "  $0 stop"
    else
        echo "✗ Failed to start SSH tunnel"
        exit 1
    fi
}

# Function to stop tunnel
stop_tunnel() {
    if ! check_tunnel; then
        echo "✓ No SSH tunnel is running"
        return 0
    fi

    echo "Stopping SSH tunnel..."
    PID=$(pgrep -f "ssh.*${LOCAL_PORT}:localhost:${REMOTE_PORT}.*${AMPERE_IP}")
    kill "${PID}"

    # Wait for process to terminate
    sleep 1

    if ! check_tunnel; then
        echo "✓ SSH tunnel stopped"
    else
        echo "✗ Failed to stop SSH tunnel (PID: ${PID})"
        exit 1
    fi
}

# Function to show tunnel status
status_tunnel() {
    if check_tunnel; then
        PID=$(pgrep -f "ssh.*${LOCAL_PORT}:localhost:${REMOTE_PORT}.*${AMPERE_IP}")
        echo "✓ SSH tunnel is running (PID: ${PID})"
        echo "  Local: localhost:${LOCAL_PORT} → Remote: ${AMPERE_IP}:${REMOTE_PORT}"
    else
        echo "✗ SSH tunnel is not running"
    fi
}

# Main script
case "${1:-}" in
    start)
        start_tunnel
        ;;
    stop)
        stop_tunnel
        ;;
    status)
        status_tunnel
        ;;
    restart)
        stop_tunnel
        start_tunnel
        ;;
    *)
        echo "SSH Tunnel Helper for Dataset Building"
        echo ""
        echo "Usage: $0 {start|stop|status|restart}"
        echo ""
        echo "Commands:"
        echo "  start    - Start SSH tunnel to Ampere PostgreSQL"
        echo "  stop     - Stop SSH tunnel"
        echo "  status   - Check if tunnel is running"
        echo "  restart  - Restart SSH tunnel"
        echo ""
        echo "Example workflow:"
        echo "  1. ./scripts/ssh_tunnel.sh start"
        echo "  2. docker-compose run --rm dataset-builder python scripts/build_dataset.py --recipe recent_1min"
        echo "  3. ./scripts/ssh_tunnel.sh stop"
        exit 1
        ;;
esac
