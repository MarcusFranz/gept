#!/usr/bin/env python3
"""Build GePT3 hybrid temporal cross-item snapshot dataset from row NPZ."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `import src...` when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.gept3.hybrid_dataset import build_hybrid_snapshot_dataset_from_row_npz, save_hybrid_snapshot_npz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-dataset", required=True, help="Input row NPZ (sell-only or multi-head)")
    ap.add_argument("--out", required=True, help="Output hybrid snapshot NPZ path")
    ap.add_argument("--seq-len", type=int, default=96, help="Per-item temporal sequence length")
    ap.add_argument("--snapshot-size", type=int, default=128)
    ap.add_argument("--min-items-per-snapshot", type=int, default=24)
    ap.add_argument("--max-snapshots", type=int, default=12000)
    ap.add_argument("--select-mode", choices=["uniform", "random"], default="uniform")
    ap.add_argument("--row-pick-mode", choices=["random", "first"], default="random")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = build_hybrid_snapshot_dataset_from_row_npz(
        row_npz_path=args.row_dataset,
        seq_len=args.seq_len,
        snapshot_size=args.snapshot_size,
        min_items_per_snapshot=args.min_items_per_snapshot,
        max_snapshots=args.max_snapshots,
        select_mode=args.select_mode,
        row_pick_mode=args.row_pick_mode,
        seed=args.seed,
    )
    save_hybrid_snapshot_npz(ds, out)

    print(f"saved_hybrid_snapshot_dataset={out}")
    print(
        f"snapshots={ds.x_seq.shape[0]} snapshot_size={ds.x_seq.shape[1]} "
        f"seq_len={ds.x_seq.shape[2]} n_features={ds.x_seq.shape[3]}"
    )
    print(f"n_horizons={ds.y_sell.shape[2]} n_offsets={ds.y_sell.shape[3]}")
    print(
        f"has_y_buy={ds.y_buy is not None} "
        f"has_y_price={ds.y_price_high is not None and ds.y_price_low is not None} "
        f"has_y_regime={ds.y_regime is not None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
