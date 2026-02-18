#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

IMAGE_TAG="${IMAGE_TAG:-gept3-train:latest}"
DOCKERFILE="${DOCKERFILE:-$ROOT/packages/inference/Dockerfile.gept3-train}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$ROOT}"

INFER_HOST_DIR="$ROOT/packages/inference"
INFER_CONTAINER_DIR="/workspace/gept/packages/inference"

DATASET="${DATASET:-$INFER_CONTAINER_DIR/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz}"
RUN_NAME="${RUN_NAME:-gept3_hybrid_cuda_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b2_ga4}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker_not_found" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "docker_unavailable" >&2
  exit 1
fi

echo "[build] image=${IMAGE_TAG} dockerfile=${DOCKERFILE}"
docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" "$BUILD_CONTEXT"

echo "[run] image=${IMAGE_TAG} run_name=${RUN_NAME}"
exec docker run --rm --gpus all \
  -e ROOT=/workspace/gept \
  -e RUN_NAME="$RUN_NAME" \
  -e DATASET="$DATASET" \
  -e AUTO_RESUME="${AUTO_RESUME:-1}" \
  -e CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-1}" \
  -e DEVICE="${DEVICE:-cuda}" \
  -e PRECISION="${PRECISION:-fp16}" \
  -e EPOCHS="${EPOCHS:-12}" \
  -e BATCH_SIZE="${BATCH_SIZE:-2}" \
  -e EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}" \
  -e GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}" \
  -e D_MODEL="${D_MODEL:-384}" \
  -e N_HEADS="${N_HEADS:-12}" \
  -e N_LAYERS="${N_LAYERS:-6}" \
  -e TEMPORAL_N_HEADS="${TEMPORAL_N_HEADS:-6}" \
  -e TEMPORAL_N_LAYERS="${TEMPORAL_N_LAYERS:-3}" \
  -e LR="${LR:-2.5e-4}" \
  -v "$INFER_HOST_DIR/data:$INFER_CONTAINER_DIR/data" \
  -v "$INFER_HOST_DIR/models:$INFER_CONTAINER_DIR/models" \
  -v "$INFER_HOST_DIR/logs:$INFER_CONTAINER_DIR/logs" \
  "$IMAGE_TAG"
