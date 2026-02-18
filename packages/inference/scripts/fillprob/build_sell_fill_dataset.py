#!/usr/bin/env python3
"""Build a sell-fill dataset NPZ from TimescaleDB.

Requires:
1) DB env vars loaded (DB_PASS is required; DB_HOST/DB_PORT optional)
2) For production DB access from your laptop: an SSH tunnel to Ampere DB on localhost:5432
"""

from __future__ import annotations

import os
import sys

# Allow `import src...` when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
from pathlib import Path

from src.fillprob.config import FillProbConfig
from src.fillprob.dataset import build_and_save_dataset_npz


def _ensure_db_env_from_connection_string() -> None:
    """Populate DB_* env vars from DB_CONNECTION_STRING if DB_PASS is missing.

    The repo's engine uses DB_CONNECTION_STRING; the inference/db_utils module expects DB_PASS
    (plus optional DB_HOST/DB_PORT/DB_USER/DB_NAME). This keeps local workflows simple.
    """
    if os.getenv("DB_PASS"):
        return
    cs = os.getenv("DB_CONNECTION_STRING")
    if not cs:
        return
    try:
        from urllib.parse import urlparse

        u = urlparse(cs)
        if u.scheme not in ("postgres", "postgresql"):
            return
        if u.password:
            os.environ.setdefault("DB_PASS", u.password)
        if u.hostname:
            os.environ.setdefault("DB_HOST", u.hostname)
        if u.port:
            os.environ.setdefault("DB_PORT", str(u.port))
        if u.username:
            os.environ.setdefault("DB_USER", u.username)
        if u.path and u.path != "/":
            os.environ.setdefault("DB_NAME", u.path.lstrip("/"))
    except Exception:
        # If parsing fails, let downstream code raise on missing DB_PASS.
        return


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
        help="Optional model_id filter when --item-source=predictions (empty disables filtering).",
    )
    ap.add_argument("--lookback-days", type=int, default=90)
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap (0 disables)")
    ap.add_argument("--min-volume", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=1, help="Take every Nth 5m row (1=full, 3 => 15m)")
    ap.add_argument("--max-rows-per-item", type=int, default=0, help="Optional cap (0 disables)")
    ap.add_argument("--log-every", type=int, default=25, help="Print progress every N items")
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = FillProbConfig()
    max_rows = args.max_rows_per_item if args.max_rows_per_item and args.max_rows_per_item > 0 else None
    max_items = args.max_items if args.max_items and args.max_items > 0 else None
    pred_model_id = args.predictions_model_id.strip() if args.predictions_model_id is not None else ""

    _ensure_db_env_from_connection_string()

    build_and_save_dataset_npz(
        out_path=str(out_path),
        min_volume=args.min_volume,
        lookback_days=args.lookback_days,
        max_items=max_items or 10_000_000,
        sample_every=args.sample_every,
        max_rows_per_item=max_rows,
        seed=args.seed,
        cfg=cfg,
        item_source=args.item_source,
        predictions_model_id=(pred_model_id or None),
        log_every=args.log_every,
    )
    print(f"saved_dataset={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
