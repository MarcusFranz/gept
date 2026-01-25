#!/bin/bash
# Prepare training data with volume targets for Optuna tuning
#
# Usage: ./scripts/prepare_volume_training_data.sh [--items configs/top_1000_items.txt]
#
# This runs the full pipeline:
# 1. Convert CSV exports to per-item parquet
# 2. Run stage2 sampling
# 3. Run stage3 precompute with volume targets
#
# Estimated time: 30-60 minutes for 1000 items on modern CPU

set -e

ITEMS_FILE="${1:-configs/top_1000_items.txt}"
CSV_DIR="data/hydra_export"
CLEANED_DIR="data/cleaned"
SAMPLES_DIR="data/samples"
FEATURES_DIR="data/precomputed"

echo "=============================================="
echo "  VOLUME TRAINING DATA PREPARATION"
echo "=============================================="
echo "  Items file: ${ITEMS_FILE}"
echo "  CSV input:  ${CSV_DIR}"
echo "  Output:     ${FEATURES_DIR}"
echo "=============================================="

# Check prerequisites
if [ ! -f "$CSV_DIR/price_data_5min.csv" ]; then
    echo "ERROR: CSV exports not found in $CSV_DIR"
    echo "Expected: price_data_5min.csv, price_data_1h.csv, price_data_4h.csv"
    exit 1
fi

# Step 1: Convert CSV to parquet
echo ""
echo "[1/3] Converting CSV to per-item parquet files..."
python scripts/convert_csv_to_parquet.py \
    --csv-dir "$CSV_DIR" \
    --output-dir "$CLEANED_DIR" \
    --items "$ITEMS_FILE"

# Step 2: Generate samples
echo ""
echo "[2/3] Generating sample index..."
mkdir -p "$SAMPLES_DIR"
python -m src.pipeline.stage2_sample \
    --parquet "$CLEANED_DIR" \
    --output "$SAMPLES_DIR/samples.csv" \
    --train-ratio 0.8

# Step 3: Precompute features with volume targets
echo ""
echo "[3/3] Precomputing features with volume targets..."
python -m src.pipeline.stage3_precompute \
    --samples "$SAMPLES_DIR/samples.csv" \
    --parquet "$CLEANED_DIR" \
    --output "$FEATURES_DIR" \
    --enable-volume

echo ""
echo "=============================================="
echo "  DATA PREPARATION COMPLETE"
echo "=============================================="
echo ""
echo "Output directory: $FEATURES_DIR"
echo ""
ls -la "$FEATURES_DIR/train/" | head -10
echo ""
echo "Volume stats: $FEATURES_DIR/volume_stats.json"
echo ""
echo "To verify volume_targets are present:"
echo "  python -c \"import numpy as np; d=np.load('$FEATURES_DIR/train/chunk_0000.npz'); print(list(d.keys()))\""
echo ""
echo "Ready to deploy to Vast.ai!"
echo "=============================================="
