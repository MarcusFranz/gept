#!/bin/bash
# =============================================================================
# PatchTST Pipeline End-to-End Test
# =============================================================================
#
# Runs stages 1-3 of the training pipeline with test items and hourly sampling
# to verify the pipeline works correctly before running at full scale.
#
# Prerequisites:
#   - Database connection configured (DB_PASS environment variable)
#   - SSH tunnel to Ampere if running locally
#
# Usage:
#   ./scripts/test_pipeline.sh
#   ./scripts/test_pipeline.sh --skip-extract  # Skip stage 1 if already done
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(dirname "$SCRIPT_DIR")"
cd "$MODEL_DIR"

ITEMS_FILE="configs/items_test.txt"
DATA_DIR="data"
START_DATE="2024-06-01"
END_DATE="2025-01-15"
VAL_CUTOFF="2024-12-01"
SAMPLE_INTERVAL="1hour"  # Use hourly for faster testing

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "\n${GREEN}==== $1 ====${NC}\n"
}

echo_warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

echo_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Parse arguments
SKIP_EXTRACT=false
for arg in "$@"; do
    case $arg in
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
    esac
done

# Check prerequisites
if [ -z "$DB_PASS" ]; then
    echo_error "DB_PASS environment variable not set"
    echo "Set it with: export DB_PASS='your_password'"
    exit 1
fi

if [ ! -f "$ITEMS_FILE" ]; then
    echo_error "Items file not found: $ITEMS_FILE"
    exit 1
fi

NUM_ITEMS=$(wc -l < "$ITEMS_FILE" | tr -d ' ')
echo "Testing pipeline with $NUM_ITEMS items from $ITEMS_FILE"
echo "Date range: $START_DATE to $END_DATE"
echo "Sample interval: $SAMPLE_INTERVAL"
echo ""

# =============================================================================
# Stage 1: Extract & Clean
# =============================================================================
if [ "$SKIP_EXTRACT" = true ]; then
    echo_warn "Skipping Stage 1 (--skip-extract flag)"
else
    echo_step "Stage 1: Extract & Clean"

    python3 -m src.pipeline.stage1_extract \
        --items "$ITEMS_FILE" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --output "$DATA_DIR/cleaned/"

    if [ $? -ne 0 ]; then
        echo_error "Stage 1 failed"
        exit 1
    fi

    echo_step "Stage 1: Validation"
    python3 -m src.pipeline.stage1_extract \
        --validate-only \
        --output "$DATA_DIR/cleaned/"
fi

# Check Stage 1 output exists
if [ ! -d "$DATA_DIR/cleaned/5min" ]; then
    echo_error "Stage 1 output not found at $DATA_DIR/cleaned/5min"
    exit 1
fi

# =============================================================================
# Stage 2: Sample & Index
# =============================================================================
echo_step "Stage 2: Sample & Index"

python3 -m src.pipeline.stage2_sample \
    --input "$DATA_DIR/cleaned/" \
    --output "$DATA_DIR/samples/" \
    --val-cutoff "$VAL_CUTOFF" \
    --interval "$SAMPLE_INTERVAL"

if [ $? -ne 0 ]; then
    echo_error "Stage 2 failed"
    exit 1
fi

# Check output
if [ ! -f "$DATA_DIR/samples/samples.csv" ]; then
    echo_error "Stage 2 output not found"
    exit 1
fi

TOTAL_SAMPLES=$(tail -n +2 "$DATA_DIR/samples/samples.csv" | wc -l | tr -d ' ')
echo "Generated $TOTAL_SAMPLES samples"

# =============================================================================
# Stage 3: Precompute Features
# =============================================================================
echo_step "Stage 3: Precompute Features"

python3 -m src.pipeline.stage3_precompute \
    --samples "$DATA_DIR/samples/samples.csv" \
    --parquet "$DATA_DIR/cleaned/" \
    --output "$DATA_DIR/features/" \
    --chunk-size 1000  # Smaller chunks for testing

if [ $? -ne 0 ]; then
    echo_error "Stage 3 failed"
    exit 1
fi

echo_step "Stage 3: Validation"
python3 -m src.pipeline.stage3_precompute \
    --validate-only \
    --output "$DATA_DIR/features/"

# =============================================================================
# Summary
# =============================================================================
echo_step "Pipeline Test Complete!"

echo "Output directories:"
echo "  - Cleaned data: $DATA_DIR/cleaned/"
echo "  - Sample index: $DATA_DIR/samples/"
echo "  - Features:     $DATA_DIR/features/"
echo ""

# Count outputs
NUM_5MIN=$(ls "$DATA_DIR/cleaned/5min"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
NUM_TRAIN_CHUNKS=$(ls "$DATA_DIR/features/train"/chunk_*.npz 2>/dev/null | wc -l | tr -d ' ')
NUM_VAL_CHUNKS=$(ls "$DATA_DIR/features/val"/chunk_*.npz 2>/dev/null | wc -l | tr -d ' ')

echo "Results:"
echo "  - Items extracted: $NUM_5MIN"
echo "  - Total samples: $TOTAL_SAMPLES"
echo "  - Train chunks: $NUM_TRAIN_CHUNKS"
echo "  - Val chunks: $NUM_VAL_CHUNKS"
echo ""

echo -e "${GREEN}All stages completed successfully!${NC}"
echo ""
echo "To run training:"
echo "  python3 -m src.pipeline.stage4_train --config configs/research_small.yaml --data-dir $DATA_DIR"
