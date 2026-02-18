#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
PLATFORM="${PLATFORM:-linux/amd64}"

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-}"
IMAGE_REPO="${IMAGE_REPO:-}"
TAG="${TAG:-}"
PUSH_LATEST="${PUSH_LATEST:-1}"
LOGIN="${LOGIN:-1}"
PUSH="${PUSH:-1}"

INCLUDE_ALL_DATA="${INCLUDE_ALL_DATA:-0}"
# Comma-separated. Paths may be absolute or repo-relative.
DATASET_PATHS="${DATASET_PATHS:-packages/inference/data/gept3_hybrid_snap_365d_preditems_allmodels_s128_t48_n4500_min24_multitask_h1-24.npz}"
KEEP_CONTEXT="${KEEP_CONTEXT:-0}"

if [[ -z "$IMAGE_REPO" ]]; then
  if [[ -z "$DOCKERHUB_USERNAME" ]]; then
    echo "set IMAGE_REPO=<dockerhub_user>/<repo> or DOCKERHUB_USERNAME=<dockerhub_user>" >&2
    exit 1
  fi
  IMAGE_REPO="${DOCKERHUB_USERNAME}/gept3-train-allinone"
fi

if [[ -z "$TAG" ]]; then
  if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    TAG="$(git -C "$ROOT" rev-parse --short HEAD)"
  else
    TAG="$(date +%Y%m%d-%H%M%S)"
  fi
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker_not_found" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "docker_unavailable" >&2
  exit 1
fi

if [[ "$LOGIN" == "1" || "$LOGIN" == "true" || "$LOGIN" == "TRUE" ]]; then
  if [[ -n "${DOCKERHUB_TOKEN:-}" && -n "${DOCKERHUB_USERNAME:-}" ]]; then
    echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
  else
    docker login
  fi
fi

if ! docker buildx inspect >/dev/null 2>&1; then
  docker buildx create --use >/dev/null
fi

TMP_CTX="$(mktemp -d /tmp/gept3-allinone-build-XXXXXX)"
if [[ "$KEEP_CONTEXT" != "1" && "$KEEP_CONTEXT" != "true" && "$KEEP_CONTEXT" != "TRUE" ]]; then
  trap 'rm -rf "$TMP_CTX"' EXIT
fi

mkdir -p "$TMP_CTX/packages/inference"

cp "$ROOT/packages/inference/requirements.txt" "$TMP_CTX/packages/inference/requirements.txt"
cp "$ROOT/packages/inference/Dockerfile.gept3-train-allinone" "$TMP_CTX/packages/inference/Dockerfile.gept3-train-allinone"
cp -R "$ROOT/packages/inference/src" "$TMP_CTX/packages/inference/src"
cp -R "$ROOT/packages/inference/scripts" "$TMP_CTX/packages/inference/scripts"
mkdir -p "$TMP_CTX/packages/inference/data"

if [[ "$INCLUDE_ALL_DATA" == "1" || "$INCLUDE_ALL_DATA" == "true" || "$INCLUDE_ALL_DATA" == "TRUE" ]]; then
  echo "[context] including all files from packages/inference/data"
  if [[ -d "$ROOT/packages/inference/data" ]]; then
    cp -R "$ROOT/packages/inference/data/." "$TMP_CTX/packages/inference/data/"
  fi
else
  IFS=',' read -r -a paths <<< "$DATASET_PATHS"
  for raw in "${paths[@]}"; do
    p="$(echo "$raw" | xargs)"
    [[ -z "$p" ]] && continue
    if [[ "$p" != /* ]]; then
      p="$ROOT/$p"
    fi
    if [[ ! -f "$p" ]]; then
      echo "dataset_missing path=$p" >&2
      exit 1
    fi
    echo "[context] adding dataset: $p"
    cp "$p" "$TMP_CTX/packages/inference/data/"
  done
fi

ls -lh "$TMP_CTX/packages/inference/data" | sed 's/^/[context-data] /'

IMAGE_TAGGED="${IMAGE_REPO}:${TAG}"
TAGS=( -t "$IMAGE_TAGGED" )
if [[ "$PUSH_LATEST" == "1" || "$PUSH_LATEST" == "true" || "$PUSH_LATEST" == "TRUE" ]]; then
  TAGS+=( -t "${IMAGE_REPO}:latest" )
fi

echo "[buildx] context=$TMP_CTX platform=$PLATFORM"
if [[ "$PUSH" == "1" || "$PUSH" == "true" || "$PUSH" == "TRUE" ]]; then
  echo "[push] image=$IMAGE_TAGGED"
  docker buildx build \
    --platform "$PLATFORM" \
    -f "$TMP_CTX/packages/inference/Dockerfile.gept3-train-allinone" \
    "${TAGS[@]}" \
    --push \
    "$TMP_CTX"
  echo "published_image=${IMAGE_TAGGED}"
  if [[ "$PUSH_LATEST" == "1" || "$PUSH_LATEST" == "true" || "$PUSH_LATEST" == "TRUE" ]]; then
    echo "published_image=${IMAGE_REPO}:latest"
  fi
else
  echo "[build-only] image=$IMAGE_TAGGED"
  docker buildx build \
    --platform "$PLATFORM" \
    -f "$TMP_CTX/packages/inference/Dockerfile.gept3-train-allinone" \
    "${TAGS[@]}" \
    --load \
    "$TMP_CTX"
  echo "built_local_image=${IMAGE_TAGGED}"
fi
if [[ "$KEEP_CONTEXT" == "1" || "$KEEP_CONTEXT" == "true" || "$KEEP_CONTEXT" == "TRUE" ]]; then
  echo "saved_build_context=$TMP_CTX"
fi
