#!/usr/bin/env python3
"""Build GePT3 multi-head row dataset NPZ from TimescaleDB."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `import src...` when running as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.fillprob.config import FillProbConfig
from src.gept3.row_dataset import build_and_save_gept3_row_dataset_npz


def _ensure_db_env_from_connection_string() -> None:
    """Populate DB_* env vars from DB_CONNECTION_STRING if DB_PASS is missing."""
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
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output row dataset NPZ path")
    ap.add_argument("--lookback-days", type=int, default=180)
    ap.add_argument("--item-source", choices=["tradeable", "predictions"], default="predictions")
    ap.add_argument(
        "--predictions-model-id",
        default="",
        help="Optional model_id filter when --item-source=predictions",
    )
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap (0 disables)")
    ap.add_argument("--min-volume", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=1, help="Take every Nth 5m row (1=full)")
    ap.add_argument("--max-rows-per-item", type=int, default=0, help="Optional cap (0 disables)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--regime-hours", type=int, default=6)
    ap.add_argument("--regime-ret-threshold", type=float, default=0.010)
    ap.add_argument("--regime-drawdown-threshold", type=float, default=0.015)
    ap.add_argument("--price-delta-clip-abs", type=float, default=0.30)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = FillProbConfig()
    max_rows = args.max_rows_per_item if args.max_rows_per_item and args.max_rows_per_item > 0 else None
    max_items = args.max_items if args.max_items and args.max_items > 0 else 10_000_000
    pred_model_id = args.predictions_model_id.strip() if args.predictions_model_id is not None else ""

    _ensure_db_env_from_connection_string()

    build_and_save_gept3_row_dataset_npz(
        out_path=str(out_path),
        min_volume=args.min_volume,
        lookback_days=args.lookback_days,
        max_items=max_items,
        sample_every=args.sample_every,
        max_rows_per_item=max_rows,
        seed=args.seed,
        cfg=cfg,
        item_source=args.item_source,
        predictions_model_id=(pred_model_id or None),
        log_every=args.log_every,
        regime_hours=args.regime_hours,
        regime_ret_threshold=args.regime_ret_threshold,
        regime_drawdown_threshold=args.regime_drawdown_threshold,
        price_delta_clip_abs=args.price_delta_clip_abs,
    )
    print(f"saved_dataset={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
