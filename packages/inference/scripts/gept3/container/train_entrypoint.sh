#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/gept}"
INFER_DIR="$ROOT/packages/inference"
RUN_SCRIPT="${RUN_SCRIPT:-$INFER_DIR/scripts/gept3/run_hybrid_cuda_scaleup.sh}"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "run_script_missing path=$RUN_SCRIPT" >&2
  exit 1
fi

exec bash "$RUN_SCRIPT"
