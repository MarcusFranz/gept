#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
DOCKERFILE="${DOCKERFILE:-$ROOT/packages/inference/Dockerfile.gept3-train}"
BUILD_CONTEXT="${BUILD_CONTEXT:-$ROOT}"
PLATFORM="${PLATFORM:-linux/amd64}"

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-}"
IMAGE_REPO="${IMAGE_REPO:-}"
TAG="${TAG:-}"
PUSH_LATEST="${PUSH_LATEST:-1}"
LOGIN="${LOGIN:-1}"

if [[ -z "$IMAGE_REPO" ]]; then
  if [[ -z "$DOCKERHUB_USERNAME" ]]; then
    echo "set IMAGE_REPO=<dockerhub_user>/<repo> or DOCKERHUB_USERNAME=<dockerhub_user>" >&2
    exit 1
  fi
  IMAGE_REPO="${DOCKERHUB_USERNAME}/gept3-train"
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

IMAGE_TAGGED="${IMAGE_REPO}:${TAG}"
TAGS=( -t "$IMAGE_TAGGED" )
if [[ "$PUSH_LATEST" == "1" || "$PUSH_LATEST" == "true" || "$PUSH_LATEST" == "TRUE" ]]; then
  TAGS+=( -t "${IMAGE_REPO}:latest" )
fi

echo "[buildx] dockerfile=$DOCKERFILE context=$BUILD_CONTEXT platform=$PLATFORM"
echo "[push] image=$IMAGE_TAGGED"

docker buildx build \
  --platform "$PLATFORM" \
  -f "$DOCKERFILE" \
  "${TAGS[@]}" \
  --push \
  "$BUILD_CONTEXT"

echo "published_image=${IMAGE_TAGGED}"
if [[ "$PUSH_LATEST" == "1" || "$PUSH_LATEST" == "true" || "$PUSH_LATEST" == "TRUE" ]]; then
  echo "published_image=${IMAGE_REPO}:latest"
fi
