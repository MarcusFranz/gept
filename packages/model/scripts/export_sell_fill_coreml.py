#!/usr/bin/env python3
"""Export a trained sell-fill model artifact to CoreML (Apple runtime).

Important:
- CoreML inference does NOT run on Ampere Linux.
- This export is useful for local Apple validation or future Apple clients.
- For Ampere inference, use the TorchScript export (`model_torchscript.pt`).
"""

from __future__ import annotations

import os
import sys

# Allow `import src...` when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
from pathlib import Path

import numpy as np
import torch

from src.fillprob.artifact import SellFillArtifact


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True, help="Directory containing artifact.json and model.pt")
    ap.add_argument("--out", required=True, help="Output path (.mlpackage recommended)")
    args = ap.parse_args()

    try:
        import coremltools as ct  # type: ignore
    except Exception as e:
        raise SystemExit(
            "coremltools is not installed in this environment. Install it with:\n"
            "  pip install coremltools\n"
            f"Original import error: {e}"
        )

    artifact = SellFillArtifact.load_dir(args.artifact_dir)
    model = artifact.build_model()
    model.eval()

    n_features = artifact.model_config.n_features

    # TorchScript trace for conversion.
    example_x = torch.zeros((1, n_features), dtype=torch.float32)
    example_item = torch.zeros((1,), dtype=torch.int32)  # will be cast to long inside model
    example_off = torch.zeros((1, 1), dtype=torch.float32)
    example_hor = torch.ones((1, 1), dtype=torch.float32)
    traced = torch.jit.trace(model, (example_x, example_item, example_off, example_hor))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=example_x.shape, dtype=np.float32),
            ct.TensorType(name="item_id", shape=example_item.shape, dtype=np.int32),
            ct.TensorType(name="offset", shape=example_off.shape, dtype=np.float32),
            ct.TensorType(name="horizon_norm", shape=example_hor.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_units=ct.ComputeUnit.ALL,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved_coreml={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
