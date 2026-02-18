#!/usr/bin/env python3
"""Train a sell-fill probability model (V3).

This trains a light MLP classifier that predicts:
  P(sell at current_high*(1+offset) fills within horizon_hours)

Training is done in PyTorch (works on Apple Silicon via MPS).
For Ampere inference, we export the same model to TorchScript.

CoreML notes:
- CoreML is an Apple runtime/format; it does not run on Ampere Linux.
- We can export to CoreML for local Apple inference, but Ampere should use TorchScript/ONNX.
"""

from __future__ import annotations

import os
import sys

# Allow `import src...` when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.fillprob.artifact import SellFillArtifact
from src.fillprob.dataset import SellFillDatasetArrays, load_dataset_npz
from src.fillprob.model import SellFillMLP, SellFillModelConfig


def _pick_device(device_arg: str) -> torch.device:
    forced = (device_arg or "auto").strip().lower()
    if forced != "auto":
        if forced == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested --device=cuda but CUDA is not available")
        if forced == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested --device=mps but MPS is not available")
        return torch.device(forced)

    # Auto priority: CUDA (Hydra/Ampere) -> MPS (Apple Silicon) -> CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _time_split_indices(ts: np.ndarray, val_days: int, test_days: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by time: train | val | test based on last timestamp."""
    if len(ts) == 0:
        raise ValueError("Empty ts")
    max_ts = int(ts.max())
    test_cut = max_ts - int(test_days) * 86400
    val_cut = test_cut - int(val_days) * 86400

    train_idx = np.where(ts < val_cut)[0]
    val_idx = np.where((ts >= val_cut) & (ts < test_cut))[0]
    test_idx = np.where(ts >= test_cut)[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Split produced empty partition(s): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
            f"Try larger lookback_days or smaller val/test windows."
        )
    return train_idx, val_idx, test_idx


def _compute_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def _sample_pairs(
    rng: np.random.Generator,
    batch_size: int,
    n_horizons: int,
    n_offsets: int,
) -> tuple[np.ndarray, np.ndarray]:
    h_idx = rng.integers(0, n_horizons, size=(batch_size,), dtype=np.int32)
    o_idx = rng.integers(0, n_offsets, size=(batch_size,), dtype=np.int32)
    return h_idx, o_idx


@torch.no_grad()
def _eval_sampled(
    model: SellFillMLP,
    ds: SellFillDatasetArrays,
    idx: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    horizons: np.ndarray,
    offsets: np.ndarray,
    device: torch.device,
    max_rows: int = 50000,
    seed: int = 123,
) -> dict:
    rng = np.random.default_rng(seed)
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
        idx.sort()

    x = (ds.x[idx] - x_mean) / x_std
    item_id = ds.item_id[idx].astype(np.int64, copy=False)
    y = ds.y[idx]

    n = len(idx)
    n_h = len(horizons)
    n_o = len(offsets)

    # One random pair per row for a cheap but stable eval.
    h_idx, o_idx = _sample_pairs(rng, n, n_h, n_o)
    y_bin = y[np.arange(n), h_idx, o_idx].astype(np.float32)

    x_t = torch.from_numpy(x).to(device)
    item_t = torch.from_numpy(item_id).to(device)
    off_t = torch.from_numpy(offsets[o_idx].astype(np.float32).reshape(-1, 1)).to(device)
    hor_t = torch.from_numpy((horizons[h_idx].astype(np.float32) / 24.0).reshape(-1, 1)).to(device)

    logits = model(x_t, item_t, off_t, hor_t)
    prob = torch.sigmoid(logits).detach().cpu().numpy()

    eps = 1e-6
    prob_c = np.clip(prob, eps, 1.0 - eps)
    brier = float(np.mean((prob - y_bin) ** 2))
    logloss = float(-np.mean(y_bin * np.log(prob_c) + (1.0 - y_bin) * np.log(1.0 - prob_c)))

    return {
        "n": int(n),
        "brier": brier,
        "logloss": logloss,
        "pos_rate": float(y_bin.mean()),
        "mean_pred": float(prob.mean()),
    }


def train(args) -> int:
    ds, cfg = load_dataset_npz(args.dataset)
    horizons = np.asarray(list(cfg.horizons_hours), dtype=np.int32)
    offsets = np.asarray(list(cfg.offsets), dtype=np.float32)

    train_idx, val_idx, test_idx = _time_split_indices(ds.ts, args.val_days, args.test_days)

    x_mean, x_std = _compute_norm(ds.x[train_idx])

    n_items = int(ds.item_id.max()) + 1
    model_cfg = SellFillModelConfig(
        n_items=n_items,
        n_features=int(ds.x.shape[1]),
        item_embed_dim=args.item_embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = SellFillMLP(model_cfg)

    device = _pick_device(args.device)
    print(f"training_device={device}", flush=True)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rng = np.random.default_rng(args.seed)

    # Precompute for speed.
    y_all = ds.y

    def iter_batches(idxs: np.ndarray):
        idxs = idxs.copy()
        rng.shuffle(idxs)
        for i in range(0, len(idxs), args.batch_size):
            yield idxs[i : i + args.batch_size]

    for epoch in range(1, args.epochs + 1):
        model.train(True)
        losses = []

        for batch_idx in iter_batches(train_idx):
            x = (ds.x[batch_idx] - x_mean) / x_std
            item_id = ds.item_id[batch_idx].astype(np.int64, copy=False)

            x_t = torch.from_numpy(x).to(device)
            item_t = torch.from_numpy(item_id).to(device)

            # Sample K (horizon, offset) pairs per row and average the loss.
            loss_accum = 0.0
            for _ in range(int(args.pairs_per_row)):
                h_idx, o_idx = _sample_pairs(rng, len(batch_idx), len(horizons), len(offsets))
                y_bin = y_all[batch_idx, h_idx, o_idx].astype(np.float32)

                off_t = torch.from_numpy(offsets[o_idx].reshape(-1, 1)).to(device)
                hor_t = torch.from_numpy((horizons[h_idx].astype(np.float32) / 24.0).reshape(-1, 1)).to(device)
                y_t = torch.from_numpy(y_bin).to(device)

                logits = model(x_t, item_t, off_t, hor_t)
                loss = F.binary_cross_entropy_with_logits(logits, y_t)
                loss_accum = loss_accum + loss

            loss = loss_accum / float(args.pairs_per_row)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        model.train(False)
        train_loss = float(np.mean(losses)) if losses else float("nan")

        val_metrics = _eval_sampled(
            model=model,
            ds=ds,
            idx=val_idx,
            x_mean=x_mean,
            x_std=x_std,
            horizons=horizons,
            offsets=offsets,
            device=device,
            max_rows=args.eval_rows,
            seed=args.seed + 1,
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_brier={val_metrics['brier']:.4f} val_logloss={val_metrics['logloss']:.4f} "
            f"val_pos_rate={val_metrics['pos_rate']:.3f} val_mean_pred={val_metrics['mean_pred']:.3f}",
            flush=True,
        )

    # Final test metrics.
    test_metrics = _eval_sampled(
        model=model,
        ds=ds,
        idx=test_idx,
        x_mean=x_mean,
        x_std=x_std,
        horizons=horizons,
        offsets=offsets,
        device=device,
        max_rows=args.eval_rows,
        seed=args.seed + 2,
    )
    print(
        f"test_brier={test_metrics['brier']:.4f} test_logloss={test_metrics['logloss']:.4f} "
        f"test_pos_rate={test_metrics['pos_rate']:.3f} test_mean_pred={test_metrics['mean_pred']:.3f}",
        flush=True,
    )

    # Save artifact.
    out_dir = Path(args.out_dir)
    if args.run_name:
        out_dir = out_dir / args.run_name
    else:
        out_dir = out_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    artifact = SellFillArtifact(
        model_config=model_cfg,
        fillprob_config=cfg,
        feature_cols=ds.feature_cols,
        feature_mean=x_mean.astype(np.float32).tolist(),
        feature_std=x_std.astype(np.float32).tolist(),
        item_index_mode="identity",
        model_state_dict=model.state_dict(),
    )
    artifact.save_dir(out_dir)

    # Save a small training report for quick inspection.
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "n_rows": int(ds.x.shape[0]),
        "n_features": int(ds.x.shape[1]),
        "n_items": int(n_items),
        "offsets": offsets.tolist(),
        "horizons_hours": horizons.tolist(),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    # Export TorchScript for Ampere inference.
    try:
        example_x = torch.zeros((1, ds.x.shape[1]), dtype=torch.float32)
        example_item = torch.zeros((1,), dtype=torch.int64)
        example_off = torch.zeros((1, 1), dtype=torch.float32)
        example_hor = torch.ones((1, 1), dtype=torch.float32) * (24.0 / 24.0)
        traced = torch.jit.trace(model.cpu(), (example_x, example_item, example_off, example_hor))
        traced.save(str(out_dir / "model_torchscript.pt"))
        print(f"saved_torchscript={out_dir / 'model_torchscript.pt'}")
    except Exception as e:
        print(f"torchscript_export_failed: {e}")

    print(f"saved_artifact_dir={out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to sell-fill dataset NPZ")
    ap.add_argument("--out-dir", default="packages/inference/models/sell_fill_v1", help="Output dir for artifacts")
    ap.add_argument("--run-name", default="", help="Optional subdir name under out-dir")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pairs-per-row", type=int, default=4, help="Random (horizon,offset) pairs sampled per row per step")

    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--item-embed-dim", type=int, default=16)

    ap.add_argument("--val-days", type=int, default=7)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--eval-rows", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
