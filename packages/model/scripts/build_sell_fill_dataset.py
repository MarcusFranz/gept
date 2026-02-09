#!/usr/bin/env python3
"""Build a sell-fill dataset NPZ from TimescaleDB.

Requires:
1) DB env vars loaded (see packages/model/.env)
2) SSH tunnel to Ampere DB on localhost:5432 (or set DB_HOST accordingly)
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

from src.fillprob.config import FillProbConfig
from src.fillprob.dataset import build_and_save_dataset_npz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument(
        "--item-source",
        choices=["tradeable", "predictions"],
        default="predictions",
        help="Which item universe to use. 'predictions' matches latest predictions batch (recommended).",
    )
    ap.add_argument(
        "--predictions-model-id",
        default="",
        help="Optional model_id filter when --item-source=predictions (empty means no filter).",
    )
    ap.add_argument("--lookback-days", type=int, default=60)
    ap.add_argument("--max-items", type=int, default=200)
    ap.add_argument("--min-volume", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=3, help="Take every Nth 5m row (3 => 15m)")
    ap.add_argument("--max-rows-per-item", type=int, default=0, help="Optional cap (0 disables)")
    ap.add_argument("--log-every", type=int, default=25, help="Print progress every N items")
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = FillProbConfig()
    max_rows = args.max_rows_per_item if args.max_rows_per_item and args.max_rows_per_item > 0 else None

    build_and_save_dataset_npz(
        out_path=str(out_path),
        min_volume=args.min_volume,
        lookback_days=args.lookback_days,
        max_items=args.max_items,
        sample_every=args.sample_every,
        max_rows_per_item=max_rows,
        seed=args.seed,
        cfg=cfg,
        item_source=args.item_source,
        predictions_model_id=(args.predictions_model_id or None),
        log_every=args.log_every,
    )
    print(f"saved_dataset={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
