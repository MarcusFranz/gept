#!/bin/bash
# Setup script for GPU training on Ubuntu with NVIDIA GPU
# Run this on the Ubuntu machine: bash setup_gpu_training.sh

set -e

echo "=== GePT GPU Training Setup ==="
echo ""

# Check NVIDIA driver
echo "1. Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "   NVIDIA driver not found. Installing..."
    sudo apt update
    sudo apt install -y nvidia-driver-535
    echo "   Driver installed. Please REBOOT and run this script again."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Check CUDA
echo "2. Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "   CUDA not found. Installing CUDA toolkit..."
    sudo apt install -y nvidia-cuda-toolkit
fi
nvcc --version 2>/dev/null || echo "   CUDA compiler not in PATH (that's OK, CatBoost bundles its own)"
echo ""

# Install Python dependencies
echo "3. Installing Python dependencies..."
sudo apt install -y python3-pip python3-venv
pip3 install --upgrade pip
pip3 install catboost numpy pandas scikit-learn pyarrow google-cloud-storage onnx onnxruntime skl2onnx

# Verify CatBoost GPU
echo ""
echo "4. Verifying CatBoost GPU support..."
python3 -c "
from catboost import CatBoostClassifier
try:
    m = CatBoostClassifier(task_type='GPU', devices='0', verbose=False)
    print('   ✓ CatBoost GPU support: WORKING')
except Exception as e:
    print(f'   ✗ CatBoost GPU support: FAILED - {e}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your GCP service account key to this machine"
echo "2. Set: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
echo "3. Run: python3 train_local.py --bucket osrs-models-mof --run-id discovery-20260110 --gpu"
