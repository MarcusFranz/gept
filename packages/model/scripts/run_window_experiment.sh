#!/bin/bash
# Window Experiment Runner for GePT
# Runs training window experiments on WSL machine

set -e

# Configuration
SSH_KEY=".secrets/oracle_key.pem"
AMPERE_HOST="ubuntu@150.136.170.128"
WSL_KEY="/home/ubuntu/.ssh/wsl_key.pem"
WSL_PORT="2222"

WINDOWS=(14 30 60 90 180)  # Days
MAX_ITEMS=20
ITERATIONS=500
DEPTH=6
MIN_ROWS=1800

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }

# SSH wrapper for WSL commands
wsl_cmd() {
    ssh -i "$SSH_KEY" "$AMPERE_HOST" "ssh -i $WSL_KEY -p $WSL_PORT ubuntu@localhost '$1'"
}

# Check WSL connectivity
log "Checking WSL connectivity..."
if ! wsl_cmd "echo connected" >/dev/null 2>&1; then
    error "Cannot connect to WSL. Check tunnel."
    exit 1
fi
log "WSL connection OK"

# Ensure DB_PASS is set
log "Verifying DB_PASS..."
if ! wsl_cmd "cat /home/ubuntu/.db_pass" >/dev/null 2>&1; then
    error "DB_PASS not set on WSL"
    exit 1
fi
log "DB_PASS verified"

# Results file
RESULTS_FILE="experiment_results_$(date '+%Y%m%d_%H%M%S').json"
echo "{" > "$RESULTS_FILE"
echo '  "experiment": "window_sweep",' >> "$RESULTS_FILE"
echo "  \"started\": \"$(date -Iseconds)\"," >> "$RESULTS_FILE"
echo '  "config": {' >> "$RESULTS_FILE"
echo "    \"windows\": [${WINDOWS[*]}]," >> "$RESULTS_FILE"
echo "    \"max_items\": $MAX_ITEMS," >> "$RESULTS_FILE"
echo "    \"iterations\": $ITERATIONS," >> "$RESULTS_FILE"
echo "    \"depth\": $DEPTH" >> "$RESULTS_FILE"
echo '  },' >> "$RESULTS_FILE"
echo '  "results": [' >> "$RESULTS_FILE"

first=true
for window in "${WINDOWS[@]}"; do
    log "========================================"
    log "Processing ${window}-day window..."
    log "========================================"

    RUN_ID="window_${window}d"

    # Prepare data
    log "Preparing data for $RUN_ID..."
    wsl_cmd "cd ~/gept && export DB_PASS=\$(cat /home/ubuntu/.db_pass) && python cloud/prepare_runpod_data.py --days $window --run-id $RUN_ID --max-items $MAX_ITEMS --min-rows $MIN_ROWS --output-dir /home/ubuntu/gept/data 2>&1" || {
        warn "Data prep failed for $RUN_ID"
        continue
    }

    # Train
    log "Training models for $RUN_ID..."
    wsl_cmd "cd ~/gept && export DB_PASS=\$(cat /home/ubuntu/.db_pass) && python cloud/train_runpod_multitarget.py --run-id $RUN_ID --all --cache-dir /home/ubuntu/gept/data --output-dir /home/ubuntu/gept/models --local --iterations $ITERATIONS --depth $DEPTH 2>&1" || {
        warn "Training failed for $RUN_ID"
        continue
    }

    # Extract results
    log "Extracting results for $RUN_ID..."
    SUMMARY=$(wsl_cmd "cat /home/ubuntu/gept/models/$RUN_ID/training_summary.json 2>/dev/null" || echo "{}")

    if [ "$first" = false ]; then
        echo "," >> "$RESULTS_FILE"
    fi
    first=false

    echo "    {" >> "$RESULTS_FILE"
    echo "      \"window_days\": $window," >> "$RESULTS_FILE"
    echo "      \"run_id\": \"$RUN_ID\"," >> "$RESULTS_FILE"
    echo "      \"summary\": $SUMMARY" >> "$RESULTS_FILE"
    echo -n "    }" >> "$RESULTS_FILE"

    log "Completed ${window}-day window"
done

echo "" >> "$RESULTS_FILE"
echo '  ],' >> "$RESULTS_FILE"
echo "  \"completed\": \"$(date -Iseconds)\"" >> "$RESULTS_FILE"
echo "}" >> "$RESULTS_FILE"

log "========================================"
log "EXPERIMENT COMPLETE"
log "Results saved to: $RESULTS_FILE"
log "========================================"
