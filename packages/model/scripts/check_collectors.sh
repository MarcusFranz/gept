#!/bin/bash
# =============================================================================
# GePT Data Collectors - Status Check Script
# =============================================================================
# Quick health check of all data collection services.
#
# Usage:
#   ./scripts/check_collectors.sh           # Full status check
#   ./scripts/check_collectors.sh --brief   # Brief status only
#   ./scripts/check_collectors.sh --json    # JSON output
#
# Exit codes:
#   0 - All services healthy
#   1 - One or more services unhealthy
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
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration
# NOTE: This repo is public; do not bake production hostnames/IPs into defaults.
: "${AMPERE_HOST:?AMPERE_HOST is required (e.g. ubuntu@your-host)}"
: "${AMPERE_SSH_KEY:?AMPERE_SSH_KEY is required (path to SSH private key)}"
SERVER="$AMPERE_HOST"
SSH_KEY="$AMPERE_SSH_KEY"
# Resolve relative path for SSH key
if [[ ! "$SSH_KEY" = /* ]]; then
    SSH_KEY="$REPO_ROOT/$SSH_KEY"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
BRIEF=false
JSON=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --brief) BRIEF=true; shift ;;
        --json) JSON=true; shift ;;
        *) shift ;;
    esac
done

# Check SSH key
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found at $SSH_KEY"
    exit 1
fi

# Run status check on server
STATUS=$(ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$SERVER" bash << 'ENDSSH'
#!/bin/bash

# Collect Docker status
DOCKER_STATUS=$(docker ps --format "{{.Names}}|{{.Status}}" 2>/dev/null)

# Collect systemd status
PLAYER_COUNT_STATUS=$(systemctl is-active player_count.service 2>/dev/null || echo "inactive")
ITEM_UPDATER_STATUS=$(systemctl is-active item_updater.service 2>/dev/null || echo "inactive")

# Collect data freshness (DB_PASS must be set on remote server)
export PGPASSWORD="\$DB_PASS"
DATA_FRESHNESS=$(psql -h localhost -U osrs_user -d osrs_data -t -A -F'|' -c "
SELECT
    'price_data_5min' as table_name,
    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))::int as age_seconds,
    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as records_1h
FROM price_data_5min
UNION ALL
SELECT
    'player_counts',
    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))::int,
    COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour')
FROM player_counts
UNION ALL
SELECT
    'predictions',
    EXTRACT(EPOCH FROM (NOW() - MAX(time)))::int,
    COUNT(*) FILTER (WHERE time > NOW() - INTERVAL '1 hour')
FROM predictions;
" 2>/dev/null)

# Collect disk usage
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')

# Collect compression stats
COMPRESSION=$(psql -h localhost -U osrs_user -d osrs_data -t -A -c "
SELECT
    (SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = 'price_data_5min' AND is_compressed) as compressed,
    (SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = 'price_data_5min') as total;
" 2>/dev/null)

# Output as structured data
echo "DOCKER:$DOCKER_STATUS"
echo "SYSTEMD:player_count=$PLAYER_COUNT_STATUS,item_updater=$ITEM_UPDATER_STATUS"
echo "DATA:$DATA_FRESHNESS"
echo "DISK:$DISK_USAGE"
echo "COMPRESSION:$COMPRESSION"
ENDSSH
)

# Parse and display results
if [ "$JSON" = true ]; then
    # JSON output
    echo "$STATUS" | python3 -c "
import sys

data = {'docker': {}, 'systemd': {}, 'data': {}, 'disk': 0, 'compression': {}}

for line in sys.stdin:
    line = line.strip()
    if line.startswith('DOCKER:'):
        containers = line[7:].split()
        for c in containers:
            if '|' in c:
                name, status = c.split('|', 1)
                data['docker'][name] = 'healthy' if 'healthy' in status.lower() or 'Up' in status else 'unhealthy'
    elif line.startswith('SYSTEMD:'):
        for item in line[8:].split(','):
            name, status = item.split('=')
            data['systemd'][name] = status
    elif line.startswith('DATA:'):
        for row in line[5:].split():
            if '|' in row:
                parts = row.split('|')
                if len(parts) >= 2:
                    data['data'][parts[0]] = {'age_seconds': int(parts[1]) if parts[1] else 0}
    elif line.startswith('DISK:'):
        data['disk'] = int(line[5:]) if line[5:].isdigit() else 0
    elif line.startswith('COMPRESSION:'):
        parts = line[12:].split('|')
        if len(parts) == 2:
            data['compression'] = {'compressed': int(parts[0]), 'total': int(parts[1])}

import json
print(json.dumps(data, indent=2))
"
    exit 0
fi

# Human-readable output
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          GePT Data Collectors - Status Report               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Docker containers
echo -e "${BLUE}Docker Containers:${NC}"
ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$SERVER" 'docker ps --format "{{.Names}}|{{.Status}}"' 2>/dev/null | while read line; do
    if [ -n "$line" ]; then
        name=$(echo "$line" | cut -d'|' -f1)
        status=$(echo "$line" | cut -d'|' -f2-)
        if echo "$status" | grep -qi "healthy\|Up"; then
            echo -e "  ${GREEN}✓${NC} $name: $status"
        else
            echo -e "  ${RED}✗${NC} $name: $status"
        fi
    fi
done
echo ""

# Systemd services
echo -e "${BLUE}Systemd Services:${NC}"
echo "$STATUS" | grep "^SYSTEMD:" | sed 's/SYSTEMD://' | tr ',' '\n' | while read line; do
    name=$(echo "$line" | cut -d'=' -f1)
    status=$(echo "$line" | cut -d'=' -f2)
    if [ "$status" = "active" ]; then
        echo -e "  ${GREEN}✓${NC} $name: $status"
    else
        echo -e "  ${RED}✗${NC} $name: $status"
    fi
done
echo ""

if [ "$BRIEF" = false ]; then
    # Data freshness
    echo -e "${BLUE}Data Freshness:${NC}"
    echo "$STATUS" | grep "^DATA:" | sed 's/DATA://' | tr ' ' '\n' | while read line; do
        if [ -n "$line" ]; then
            table=$(echo "$line" | cut -d'|' -f1)
            age=$(echo "$line" | cut -d'|' -f2)
            records=$(echo "$line" | cut -d'|' -f3)

            if [ -n "$age" ] && [ "$age" -lt 600 ]; then
                echo -e "  ${GREEN}✓${NC} $table: ${age}s ago ($records records/hr)"
            elif [ -n "$age" ] && [ "$age" -lt 3600 ]; then
                echo -e "  ${YELLOW}!${NC} $table: ${age}s ago ($records records/hr)"
            else
                echo -e "  ${RED}✗${NC} $table: ${age:-unknown}s ago"
            fi
        fi
    done
    echo ""

    # Disk usage
    echo -e "${BLUE}Disk Usage:${NC}"
    disk=$(echo "$STATUS" | grep "^DISK:" | sed 's/DISK://')
    if [ -n "$disk" ] && [ "$disk" -lt 70 ]; then
        echo -e "  ${GREEN}✓${NC} $disk% used"
    elif [ -n "$disk" ] && [ "$disk" -lt 85 ]; then
        echo -e "  ${YELLOW}!${NC} $disk% used"
    else
        echo -e "  ${RED}✗${NC} ${disk:-unknown}% used"
    fi
    echo ""

    # Compression status
    echo -e "${BLUE}TimescaleDB Compression:${NC}"
    compression=$(echo "$STATUS" | grep "^COMPRESSION:" | sed 's/COMPRESSION://')
    if [ -n "$compression" ]; then
        compressed=$(echo "$compression" | cut -d'|' -f1)
        total=$(echo "$compression" | cut -d'|' -f2)
        echo -e "  ${GREEN}✓${NC} $compressed/$total chunks compressed"
    fi
    echo ""
fi

# Access URLs
# Get IP from AMPERE_HOST (strip username@ prefix)
SERVER_IP="${AMPERE_IP:-${SERVER#*@}}"
echo -e "${BLUE}Access URLs:${NC}"
echo "  Dashboard:  http://${SERVER_IP}:${DASHBOARD_PORT:-8080}"
echo "  Prometheus: http://${SERVER_IP}:${PROMETHEUS_PORT:-9090}"
echo "  Grafana:    http://${SERVER_IP}:${GRAFANA_PORT:-3001}"
echo ""

# Check for any failures
if echo "$STATUS" | grep -q "inactive\|unhealthy\|Exited"; then
    echo -e "${RED}⚠ One or more services need attention${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All services healthy${NC}"
    exit 0
fi
