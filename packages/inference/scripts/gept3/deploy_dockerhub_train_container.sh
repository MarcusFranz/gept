#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-}"
IMAGE_REPO="${IMAGE_REPO:-}"
TAG="${TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gept3-train}"

if [[ -z "$IMAGE_REPO" ]]; then
  if [[ -z "$DOCKERHUB_USERNAME" ]]; then
    echo "set IMAGE_REPO=<dockerhub_user>/<repo> or DOCKERHUB_USERNAME=<dockerhub_user>" >&2
    exit 1
  fi
  IMAGE_REPO="${DOCKERHUB_USERNAME}/gept3-train"
fi
IMAGE="${IMAGE_REPO}:${TAG}"

INFER_CONTAINER_DIR="/workspace/gept/packages/inference"
HOST_BASE="${HOST_BASE:-/opt/gept/inference}"
HOST_DATA_DIR="${HOST_DATA_DIR:-$HOST_BASE/data}"
HOST_MODELS_DIR="${HOST_MODELS_DIR:-$HOST_BASE/models}"
HOST_LOGS_DIR="${HOST_LOGS_DIR:-$HOST_BASE/logs}"
MOUNT_DATA="${MOUNT_DATA:-1}"
MOUNT_MODELS="${MOUNT_MODELS:-1}"
MOUNT_LOGS="${MOUNT_LOGS:-1}"

RUN_NAME="${RUN_NAME:-gept3_hybrid_cuda_scaleup_s128_t48_n4500_d384_l6_tl3_e12_b2_ga4}"
DATASET="${DATASET:-$INFER_CONTAINER_DIR/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz}"

PULL_IMAGE="${PULL_IMAGE:-1}"
REMOVE_EXISTING="${REMOVE_EXISTING:-1}"
DETACH="${DETACH:-1}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker_not_found" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "docker_unavailable" >&2
  exit 1
fi

if [[ "$MOUNT_DATA" == "1" || "$MOUNT_DATA" == "true" || "$MOUNT_DATA" == "TRUE" ]]; then
  mkdir -p "$HOST_DATA_DIR"
fi
if [[ "$MOUNT_MODELS" == "1" || "$MOUNT_MODELS" == "true" || "$MOUNT_MODELS" == "TRUE" ]]; then
  mkdir -p "$HOST_MODELS_DIR"
fi
if [[ "$MOUNT_LOGS" == "1" || "$MOUNT_LOGS" == "true" || "$MOUNT_LOGS" == "TRUE" ]]; then
  mkdir -p "$HOST_LOGS_DIR"
fi

if [[ "$PULL_IMAGE" == "1" || "$PULL_IMAGE" == "true" || "$PULL_IMAGE" == "TRUE" ]]; then
  echo "[pull] $IMAGE"
  docker pull "$IMAGE"
fi

if docker ps -a --format '{{.Names}}' | rg -x "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [[ "$REMOVE_EXISTING" == "1" || "$REMOVE_EXISTING" == "true" || "$REMOVE_EXISTING" == "TRUE" ]]; then
    echo "[replace] removing existing container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  else
    echo "container_exists name=$CONTAINER_NAME and REMOVE_EXISTING=0" >&2
    exit 1
  fi
fi

args=(
  run
  --name "$CONTAINER_NAME"
  --gpus all
  --restart "$RESTART_POLICY"
  -e ROOT=/workspace/gept
  -e RUN_NAME="$RUN_NAME"
  -e DATASET="$DATASET"
  -e AUTO_RESUME="${AUTO_RESUME:-1}"
  -e CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-1}"
  -e DEVICE="${DEVICE:-cuda}"
  -e PRECISION="${PRECISION:-fp16}"
  -e EPOCHS="${EPOCHS:-12}"
  -e BATCH_SIZE="${BATCH_SIZE:-2}"
  -e EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
  -e EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-250}"
  -e GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
  -e D_MODEL="${D_MODEL:-384}"
  -e N_HEADS="${N_HEADS:-12}"
  -e N_LAYERS="${N_LAYERS:-6}"
  -e TEMPORAL_N_HEADS="${TEMPORAL_N_HEADS:-6}"
  -e TEMPORAL_N_LAYERS="${TEMPORAL_N_LAYERS:-3}"
  -e LR="${LR:-2.5e-4}"
)

if [[ "$MOUNT_DATA" == "1" || "$MOUNT_DATA" == "true" || "$MOUNT_DATA" == "TRUE" ]]; then
  args+=( -v "$HOST_DATA_DIR:$INFER_CONTAINER_DIR/data" )
fi
if [[ "$MOUNT_MODELS" == "1" || "$MOUNT_MODELS" == "true" || "$MOUNT_MODELS" == "TRUE" ]]; then
  args+=( -v "$HOST_MODELS_DIR:$INFER_CONTAINER_DIR/models" )
fi
if [[ "$MOUNT_LOGS" == "1" || "$MOUNT_LOGS" == "true" || "$MOUNT_LOGS" == "TRUE" ]]; then
  args+=( -v "$HOST_LOGS_DIR:$INFER_CONTAINER_DIR/logs" )
fi

if [[ "$DETACH" == "1" || "$DETACH" == "true" || "$DETACH" == "TRUE" ]]; then
  args+=( -d )
fi

args+=( "$IMAGE" )

echo "[run] image=$IMAGE container=$CONTAINER_NAME"
docker "${args[@]}"

echo "container_name=$CONTAINER_NAME"
echo "host_data_dir=$HOST_DATA_DIR"
echo "host_models_dir=$HOST_MODELS_DIR"
echo "host_logs_dir=$HOST_LOGS_DIR"
if [[ "$MOUNT_LOGS" == "1" || "$MOUNT_LOGS" == "true" || "$MOUNT_LOGS" == "TRUE" ]]; then
  echo "tip: tail training log with: tail -f $HOST_LOGS_DIR/$RUN_NAME.log"
else
  echo "tip: tail training log with: docker logs -f $CONTAINER_NAME"
fi
