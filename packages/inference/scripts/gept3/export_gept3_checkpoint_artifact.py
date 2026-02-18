#!/usr/bin/env python3
"""Export a GePT3 hybrid training checkpoint into an inference artifact directory.

This is intended for promoting a checkpoint (for example epoch4.pt) to a beta
inference artifact even when full training did not finish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow `import src...` when run as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.gept3.artifact import GePT3Artifact
from src.gept3.config import GePT3ModelConfig
from src.gept3.hybrid_model import GePT3HybridModel


def _load_npz_meta(path: str) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {
        "feature_cols": [str(v) for v in arr["feature_cols"].tolist()],
        "horizons_hours": [int(v) for v in arr["horizons_hours"].tolist()],
        "offsets": [float(v) for v in arr["offsets"].tolist()],
    }

    if "x_seq" in arr.files:
        x_key = "x_seq"
    elif "x" in arr.files:
        x_key = "x"
    else:
        raise RuntimeError("Could not find x_seq/x in donor dataset")

    x = arr[x_key]
    if x.ndim != 4:
        raise RuntimeError(f"Expected {x_key} ndim=4, got shape={x.shape}")
    out["seq_len"] = int(x.shape[2])
    out["n_features"] = int(x.shape[3])

    item = arr["item_id"]
    mask = arr["mask"].astype(bool, copy=False)
    valid_items = item[mask]
    out["n_items_from_data"] = int(valid_items.max()) + 1 if valid_items.size else 1
    return out


def _compute_feature_norm(path: str) -> tuple[list[float], list[float]]:
    arr = np.load(path, allow_pickle=True)
    x = arr["x_seq"] if "x_seq" in arr.files else arr["x"]
    mask = arr["mask"].astype(bool, copy=False)
    if x.ndim != 4:
        raise RuntimeError(f"Expected x_seq/x ndim=4, got shape={x.shape}")

    n_feat = int(x.shape[3])
    sum_v = np.zeros((n_feat,), dtype=np.float64)
    sumsq_v = np.zeros((n_feat,), dtype=np.float64)
    n = 0

    # Iterate snapshots in chunks to avoid loading everything at once.
    chunk = 64
    n_snap = int(x.shape[0])
    for i in range(0, n_snap, chunk):
        xb = x[i : i + chunk]
        mb = mask[i : i + chunk]
        vals = xb[mb]  # (N_items, T, F)
        if vals.size == 0:
            continue
        vals = vals.reshape(-1, n_feat).astype(np.float64, copy=False)
        sum_v += vals.sum(axis=0)
        sumsq_v += (vals * vals).sum(axis=0)
        n += vals.shape[0]

    if n == 0:
        raise RuntimeError("No valid values found while computing feature normalization")

    mean = (sum_v / n).astype(np.float32)
    var = (sumsq_v / n) - np.square(mean.astype(np.float64))
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean.tolist(), std.tolist()


def _infer_config_from_checkpoint(ckpt: dict[str, Any], meta: dict[str, Any]) -> GePT3ModelConfig:
    state = ckpt["model_state_dict"]
    args = dict(ckpt.get("args", {}))

    d_model = int(state["item_embedding.weight"].shape[1])
    n_items_state = int(state["item_embedding.weight"].shape[0])
    n_features_state = int(state["temporal_feature_proj.0.weight"].shape[1])
    temporal_len = int(state["temporal_pos_embedding.weight"].shape[0])
    n_horizons = int(state["sell_head.horizon_embedding.weight"].shape[0])
    n_offsets = int(state["sell_head.offset_embedding.weight"].shape[0])

    enable_buy = any(k.startswith("buy_head.") for k in state.keys())
    enable_price = any(k.startswith("price_head.") for k in state.keys())
    enable_regime = any(k.startswith("regime_head.") for k in state.keys())
    n_price_quantiles = int(args.get("n_price_quantiles", 5))
    price_quantiles = tuple(np.linspace(0.1, 0.9, n_price_quantiles, dtype=np.float32).tolist())

    return GePT3ModelConfig(
        n_items=max(n_items_state, int(meta.get("n_items_from_data", 1))),
        n_features=n_features_state if n_features_state > 0 else int(meta["n_features"]),
        n_horizons=n_horizons,
        n_offsets=n_offsets,
        d_model=d_model,
        n_heads=int(args.get("n_heads", 8)),
        n_layers=int(args.get("n_layers", 4)),
        ff_mult=float(args.get("ff_mult", 4.0)),
        dropout=float(args.get("dropout", 0.10)),
        temporal_len=temporal_len if temporal_len > 0 else int(meta["seq_len"]),
        temporal_n_heads=int(args.get("temporal_n_heads", 4)),
        temporal_n_layers=int(args.get("temporal_n_layers", 2)),
        n_price_quantiles=n_price_quantiles,
        price_quantiles=price_quantiles,
        enable_sell_head=True,
        enable_buy_head=enable_buy,
        enable_price_head=enable_price,
        enable_regime_head=enable_regime,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Export GePT3 hybrid checkpoint as inference artifact")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt (e.g., epoch4.pt)")
    ap.add_argument(
        "--donor-dataset",
        required=True,
        help="Hybrid dataset NPZ used as metadata/norm source (feature_cols/horizons/offsets)",
    )
    ap.add_argument("--out-dir", required=True, help="Output artifact directory")
    ap.add_argument("--skip-torchscript", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint missing model_state_dict")

    meta = _load_npz_meta(args.donor_dataset)
    feature_mean, feature_std = _compute_feature_norm(args.donor_dataset)
    cfg = _infer_config_from_checkpoint(ckpt, meta)

    artifact = GePT3Artifact(
        model_config=cfg,
        feature_cols=list(meta["feature_cols"]),
        horizons_hours=list(meta["horizons_hours"]),
        offsets=list(meta["offsets"]),
        feature_mean=feature_mean,
        feature_std=feature_std,
        model_state_dict=ckpt["model_state_dict"],
        notes={
            "source_checkpoint": str(args.checkpoint),
            "donor_dataset": str(args.donor_dataset),
            "export_reason": "beta_promotion",
            "restored_epoch": int(ckpt.get("epoch", 0)),
            "checkpoint_run_name": ckpt.get("run_name"),
        },
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact.save_dir(out_dir)

    # Optional TorchScript export for faster/stabler inference deploys.
    if not args.skip_torchscript:
        m = GePT3HybridModel(cfg)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        try:
            scripted = torch.jit.script(m)
            scripted.save(str(out_dir / "model_torchscript.pt"))
            print(f"saved_torchscript={out_dir / 'model_torchscript.pt'}")
        except Exception as exc:
            print(f"torchscript_export_failed={exc}")

    report = {
        "checkpoint": str(args.checkpoint),
        "donor_dataset": str(args.donor_dataset),
        "restored_epoch": int(ckpt.get("epoch", 0)),
        "out_dir": str(out_dir),
        "config": cfg.__dict__,
        "n_features": len(meta["feature_cols"]),
        "n_horizons": len(meta["horizons_hours"]),
        "n_offsets": len(meta["offsets"]),
    }
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"saved_artifact_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

