#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
INFER_DIR="$ROOT/packages/inference"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$INFER_DIR/.venv312}"
TORCH_PROFILE="${TORCH_PROFILE:-nightly-cu128}" # nightly-cu128 | stable-cu124 | custom
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision torchaudio}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python_not_found bin=$PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$INFER_DIR"

log "creating_venv path=$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

log "install_base_deps"
python -m pip install --upgrade pip

grep -v '^torch' "$INFER_DIR/requirements.txt" > /tmp/gept-requirements.notorch.txt
pip install -r /tmp/gept-requirements.notorch.txt

case "$TORCH_PROFILE" in
  nightly-cu128)
    log "install_torch profile=nightly-cu128"
    pip install --upgrade --pre --index-url https://download.pytorch.org/whl/nightly/cu128 $TORCH_PACKAGES
    ;;
  stable-cu124)
    log "install_torch profile=stable-cu124"
    pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
    ;;
  custom)
    if [[ -z "$TORCH_INDEX_URL" ]]; then
      echo "TORCH_PROFILE=custom requires TORCH_INDEX_URL" >&2
      exit 1
    fi
    log "install_torch profile=custom index=$TORCH_INDEX_URL"
    pip install --upgrade --pre --index-url "$TORCH_INDEX_URL" $TORCH_PACKAGES
    ;;
  *)
    echo "invalid TORCH_PROFILE=$TORCH_PROFILE (expected nightly-cu128|stable-cu124|custom)" >&2
    exit 1
    ;;
esac

log "verify_torch_cuda"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    a = torch.randn((2048, 2048), device='cuda')
    b = torch.randn((2048, 2048), device='cuda')
    c = (a @ b).mean().item()
    torch.cuda.synchronize()
    print('cuda_matmul_ok', c)
PY

log "setup_done venv=$VENV_DIR"
