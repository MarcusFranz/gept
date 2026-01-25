#!/bin/bash
# Deploy PatchTST Volume Tuning to Vast.ai
# Usage: ./scripts/deploy_vast_volume.sh <vast_ip> <vast_port>
#
# Prerequisites:
# 1. Precomputed data in data/precomputed/ with volume_targets
# 2. Vast instance with 8 GPUs (~$1/hr budget)
# 3. SSH key at .secrets/vast_key.pem

set -e

VAST_IP="${1:?Usage: $0 <vast_ip> <vast_port>}"
VAST_PORT="${2:?Usage: $0 <vast_ip> <vast_port>}"
SSH_KEY="${SSH_KEY:-.secrets/vast_key.pem}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  VAST.AI VOLUME TUNING DEPLOYMENT"
echo "=============================================="
echo "  Target: root@${VAST_IP}:${VAST_PORT}"
echo "  SSH Key: ${SSH_KEY}"
echo "=============================================="

# SSH helper
vast_ssh() {
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$VAST_PORT" "root@$VAST_IP" "$@"
}

# SCP helper
vast_scp() {
    scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$VAST_PORT" "$@"
}

echo ""
echo "[1/5] Testing SSH connection..."
vast_ssh "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
    echo "ERROR: Cannot connect to Vast instance"
    exit 1
}

echo ""
echo "[2/5] Creating workspace directories..."
vast_ssh "mkdir -p /workspace/{data,models,logs,src}"

echo ""
echo "[3/5] Uploading source code..."
# Upload pipeline code
rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $VAST_PORT" \
    --include='*.py' \
    --include='*/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "$REPO_ROOT/src/" "root@$VAST_IP:/workspace/src/"

# Upload configs
vast_scp "$REPO_ROOT/configs/vast_volume_tuning.yaml" "root@$VAST_IP:/workspace/config.yaml"

echo ""
echo "[4/5] Uploading precomputed data..."
if [ -d "$REPO_ROOT/data/precomputed" ]; then
    rsync -avz --progress \
        -e "ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $VAST_PORT" \
        "$REPO_ROOT/data/precomputed/" "root@$VAST_IP:/workspace/data/"
    echo "Data upload complete"
else
    echo "WARNING: No precomputed data found at $REPO_ROOT/data/precomputed"
    echo "You need to run stage3_precompute.py first or upload data manually"
fi

echo ""
echo "[5/5] Installing dependencies..."
vast_ssh "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true"
vast_ssh "pip install optuna numpy pandas pyyaml tqdm 2>/dev/null || true"

echo ""
echo "=============================================="
echo "  DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "To start training, SSH in and run:"
echo ""
echo "  ssh -i $SSH_KEY -p $VAST_PORT root@$VAST_IP"
echo ""
echo "  cd /workspace"
echo "  python -m src.pipeline.run_parallel_optuna \\"
echo "      --config config.yaml \\"
echo "      --n_gpus 8 \\"
echo "      --trials_per_gpu 4 \\"
echo "      --study_name patchtst_volume_v1 \\"
echo "      --enable-volume"
echo ""
echo "Monitor with: tail -f /workspace/optuna_gpu*.log"
echo "=============================================="
