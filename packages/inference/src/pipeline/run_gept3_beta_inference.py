#!/usr/bin/env python3
"""GePT3 hybrid beta inference overlay writer.

Reads the latest complete base snapshot (default model_id=gept_v2), computes a
fresh sell-fill probability grid with a GePT3 hybrid artifact, and writes rows
under a beta model_id for the engine/web beta toggle.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import torch

# Allow `import src...` when run as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.feature_engine import FeatureEngine, Granularity
from src.gept3.artifact import GePT3Artifact
from src.gept3.hybrid_model import GePT3HybridModel


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemSeq:
    item_id: int
    x_seq: np.ndarray  # (T, F)
    current_high: float
    current_low: float


def _get_db_connection():
    host = os.getenv("DB_HOST", "localhost")
    port = int(os.getenv("DB_PORT", "5432"))
    dbname = os.getenv("DB_NAME", "osrs_data")
    user = os.getenv("DB_USER", "osrs_user")
    password = os.getenv("DB_PASS")
    if not password:
        raise RuntimeError("DB_PASS is required")
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def _load_latest_full_base_rows(
    conn,
    base_model_id: str,
    lookback_hours: int = 2,
    recent_snapshots: int = 24,
    min_snapshot_items: int = 300,
) -> pd.DataFrame:
    times_sql = f"""
    SELECT time, COUNT(DISTINCT item_id) AS n_items
    FROM predictions
    WHERE model_id = %s
      AND time > NOW() - (%s || ' hours')::interval
    GROUP BY time
    ORDER BY time DESC
    LIMIT {int(recent_snapshots)}
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable*",
            category=UserWarning,
        )
        tdf = pd.read_sql(times_sql, conn, params=(base_model_id, int(lookback_hours)))
    if tdf.empty:
        raise RuntimeError(f"No recent base snapshots found for model_id={base_model_id}")

    chosen = tdf[tdf["n_items"] >= int(min_snapshot_items)]
    if not chosen.empty:
        chosen_time = chosen.iloc[0]["time"]
        chosen_items = int(chosen.iloc[0]["n_items"])
    else:
        chosen_time = tdf.iloc[0]["time"]
        chosen_items = int(tdf.iloc[0]["n_items"])

    logger.info("selected base snapshot time=%s n_items=%s", chosen_time, chosen_items)

    rows_sql = """
    SELECT
      item_id,
      item_name,
      hour_offset,
      offset_pct,
      buy_price,
      sell_price,
      current_high,
      current_low,
      confidence
    FROM predictions
    WHERE model_id = %s
      AND time = %s
    ORDER BY item_id, hour_offset, offset_pct
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable*",
            category=UserWarning,
        )
        df = pd.read_sql(rows_sql, conn, params=(base_model_id, chosen_time))
    if df.empty:
        raise RuntimeError(f"No base prediction rows found for model_id={base_model_id}")
    return df


def _load_item_price_5m(conn, item_id: int, lookback_days: int = 10) -> pd.DataFrame:
    sql = """
    SELECT timestamp, avg_high_price, avg_low_price, high_price_volume, low_price_volume
    FROM price_data_5min
    WHERE item_id = %s
      AND timestamp > NOW() - (%s || ' days')::interval
      AND avg_high_price IS NOT NULL
      AND avg_low_price IS NOT NULL
    ORDER BY timestamp ASC
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable*",
            category=UserWarning,
        )
        df = pd.read_sql(sql, conn, params=(int(item_id), int(lookback_days)))
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _tail_or_pad_rows(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] >= target_len:
        return x[-target_len:, :]
    pad = np.zeros((target_len - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.vstack([pad, x]).astype(np.float32, copy=False)


def _build_item_sequences(
    conn,
    item_ids: List[int],
    feature_cols: List[str],
    seq_len: int,
    lookback_days: int = 10,
    log_every: int = 50,
) -> Dict[int, ItemSeq]:
    fe = FeatureEngine(granularity=Granularity.FIVE_MIN)
    out: Dict[int, ItemSeq] = {}

    for i, item_id in enumerate(item_ids, start=1):
        try:
            df = _load_item_price_5m(conn, item_id=item_id, lookback_days=lookback_days)
            if df.empty:
                continue
            df_feat = fe.compute_features(df)
            if df_feat.empty:
                continue

            cur = df_feat.iloc[-1]
            current_high = float(cur.get("avg_high_price", np.nan))
            current_low = float(cur.get("avg_low_price", np.nan))
            if not np.isfinite(current_high) or not np.isfinite(current_low) or current_high <= 0 or current_low <= 0:
                continue

            # Preserve artifact feature ordering and fill missing columns with zeros
            # so minor schema drift does not zero out entire item coverage.
            vals: List[np.ndarray] = []
            for c in feature_cols:
                if c in df_feat.columns:
                    col = df_feat[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(
                        dtype=np.float32, copy=False
                    )
                else:
                    col = np.zeros((len(df_feat),), dtype=np.float32)
                vals.append(col.reshape(-1, 1))
            x = np.concatenate(vals, axis=1).astype(np.float32, copy=False)
            x = _tail_or_pad_rows(x, target_len=seq_len)
            out[int(item_id)] = ItemSeq(
                item_id=int(item_id),
                x_seq=x,
                current_high=current_high,
                current_low=current_low,
            )
        except Exception:
            continue

        if log_every and (i == 1 or i % int(log_every) == 0 or i == len(item_ids)):
            logger.info("sequences: %s/%s items ready=%s", i, len(item_ids), len(out))

    return out


def _confidence_from_prob(p: float) -> str:
    if p >= 0.80:
        return "high"
    if p >= 0.60:
        return "medium"
    return "low"


def _write_predictions_copy(conn, rows: List[dict], table_name: str = "predictions") -> int:
    if not rows:
        return 0

    columns = [
        "time",
        "item_id",
        "item_name",
        "hour_offset",
        "target_hour",
        "offset_pct",
        "fill_probability",
        "expected_value",
        "buy_price",
        "sell_price",
        "current_high",
        "current_low",
        "confidence",
        "model_id",
    ]

    buffer = io.StringIO()
    for row in rows:
        vals = [str(row[col]) if row.get(col) is not None else "" for col in columns]
        buffer.write("\t".join(vals) + "\n")
    buffer.seek(0)

    cur = conn.cursor()
    cur.copy_from(buffer, table_name, columns=columns, null="")
    conn.commit()
    cur.close()
    return len(rows)


def _pick_device(device_arg: str) -> torch.device:
    forced = (device_arg or "auto").strip().lower()
    if forced != "auto":
        if forced == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested --device=cuda but CUDA is not available")
        if forced == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Requested --device=mps but MPS is not available")
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _infer_sell_grid(
    model: GePT3HybridModel,
    artifact: GePT3Artifact,
    seq_map: Dict[int, ItemSeq],
    device: torch.device,
    chunk_size: int,
) -> Dict[int, np.ndarray]:
    item_ids = sorted(seq_map.keys())
    if not item_ids:
        return {}

    mean = np.asarray(artifact.feature_mean, dtype=np.float32)
    std = np.asarray(artifact.feature_std, dtype=np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)

    out: Dict[int, np.ndarray] = {}
    model.eval()

    for i in range(0, len(item_ids), max(1, int(chunk_size))):
        chunk_ids = item_ids[i : i + max(1, int(chunk_size))]
        x = np.stack([seq_map[it].x_seq for it in chunk_ids], axis=0).astype(np.float32, copy=False)  # (I, T, F)
        x = (x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])  # (1, I, T, F)
        item_t = np.asarray(chunk_ids, dtype=np.int64).reshape(1, -1)
        mask = np.ones((1, len(chunk_ids)), dtype=bool)

        x_t = torch.from_numpy(x).to(device)
        item_ids_t = torch.from_numpy(item_t).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        with torch.no_grad():
            pred = model(x_t, item_ids_t, mask_t)
            probs = torch.sigmoid(pred["sell_logits"]).detach().cpu().numpy().astype(np.float32)  # (1, I, H, O)
        probs = np.clip(probs, 0.01, 0.99)

        for j, item_id in enumerate(chunk_ids):
            out[int(item_id)] = probs[0, j]
    return out


def run(args) -> int:
    start = datetime.now(timezone.utc)
    logger.info("GEPT3 BETA INFERENCE START")

    device = _pick_device(args.device)
    logger.info("device=%s", device)

    artifact = GePT3Artifact.load_dir(args.artifact_dir)
    model = GePT3HybridModel(artifact.model_config)
    model.load_state_dict(artifact.model_state_dict)
    model.to(device)
    model.eval()

    horizons = np.asarray(artifact.horizons_hours, dtype=np.int32)
    offsets = np.asarray(artifact.offsets, dtype=np.float32)
    seq_len = int(artifact.model_config.temporal_len)
    if seq_len <= 0:
        raise RuntimeError("Artifact temporal_len must be > 0 for hybrid inference")

    conn = _get_db_connection()
    try:
        base_df = _load_latest_full_base_rows(
            conn,
            base_model_id=args.base_model_id,
            lookback_hours=args.snapshot_lookback_hours,
            recent_snapshots=args.recent_snapshots,
            min_snapshot_items=args.min_snapshot_items,
        )
        # Keep only V3 horizon rows.
        base_df = base_df[(base_df["hour_offset"] >= 1) & (base_df["hour_offset"] <= 24)].copy()
        logger.info("loaded base rows=%s model_id=%s", len(base_df), args.base_model_id)

        item_ids = sorted(set(int(x) for x in base_df["item_id"].tolist()))
        if args.max_items and args.max_items > 0:
            item_ids = item_ids[: int(args.max_items)]
            base_df = base_df[base_df["item_id"].isin(item_ids)].copy()
            logger.info("max_items applied: %s", len(item_ids))

        seq_map = _build_item_sequences(
            conn=conn,
            item_ids=item_ids,
            feature_cols=artifact.feature_cols,
            seq_len=seq_len,
            lookback_days=args.feature_lookback_days,
            log_every=args.log_every,
        )
        if not seq_map:
            raise RuntimeError("No item sequences available")

        sell_grid_by_item = _infer_sell_grid(
            model=model,
            artifact=artifact,
            seq_map=seq_map,
            device=device,
            chunk_size=args.chunk_size,
        )
        if not sell_grid_by_item:
            raise RuntimeError("No sell grid predictions produced")

        now = datetime.now(timezone.utc)
        out_rows: List[dict] = []

        for _, r in base_df.iterrows():
            item_id = int(r["item_id"])
            grid = sell_grid_by_item.get(item_id)
            if grid is None:
                continue

            h = int(r["hour_offset"])
            # Pick exact horizon when present, else nearest horizon.
            h_idx = int(np.argmin(np.abs(horizons - h)))
            h_probs = grid[h_idx]  # (O,)

            off = float(r["offset_pct"])
            if off <= float(offsets.min()):
                p = float(h_probs[np.argmin(offsets)])
            elif off >= float(offsets.max()):
                p = float(h_probs[np.argmax(offsets)])
            else:
                # Interpolate to support base offsets (e.g., 0.0125 / 0.0175).
                p = float(np.interp(off, offsets.astype(np.float64), h_probs.astype(np.float64)))
            p = float(np.clip(p, 0.01, 0.99))

            buy = float(r["buy_price"]) if r["buy_price"] is not None else 0.0
            sell = float(r["sell_price"]) if r["sell_price"] is not None else 0.0
            profit_pct = (sell - buy) / buy if buy > 0 else 0.0
            ev = p * profit_pct

            item_seq = seq_map.get(item_id)
            out_rows.append(
                {
                    "time": now,
                    "item_id": item_id,
                    "item_name": str(r["item_name"]) if r["item_name"] is not None else "",
                    "hour_offset": h,
                    "target_hour": now + timedelta(hours=h),
                    "offset_pct": off,
                    "fill_probability": p,
                    "expected_value": float(ev),
                    "buy_price": buy,
                    "sell_price": sell,
                    "current_high": float(item_seq.current_high) if item_seq else None,
                    "current_low": float(item_seq.current_low) if item_seq else None,
                    "confidence": _confidence_from_prob(p),
                    "model_id": args.out_model_id,
                }
            )

        written = 0
        if not args.dry_run:
            written = _write_predictions_copy(conn, out_rows, table_name=args.table)
        logger.info(
            "complete rows_written=%s candidate=%s base_rows=%s seq_items=%s model_id=%s dry_run=%s",
            written,
            len(out_rows),
            len(base_df),
            len(seq_map),
            args.out_model_id,
            args.dry_run,
        )
    finally:
        conn.close()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("GEPT3 BETA INFERENCE DONE elapsed=%.1fs", elapsed)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GePT3 hybrid beta inference overlay writer")
    ap.add_argument("--artifact-dir", required=True, help="Path containing artifact.json + model.pt")
    ap.add_argument("--base-model-id", default="gept_v2", help="Base model_id snapshot to rescore")
    ap.add_argument("--out-model-id", default="gept3_hybrid_e4_beta", help="model_id to write for beta selection")
    ap.add_argument("--table", default="predictions", help="Target table")
    ap.add_argument("--recent-snapshots", type=int, default=60, help="How many recent base snapshots to inspect")
    ap.add_argument("--snapshot-lookback-hours", type=int, default=2)
    ap.add_argument("--min-snapshot-items", type=int, default=300)
    ap.add_argument("--feature-lookback-days", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=256, help="Items per forward pass")
    ap.add_argument("--max-items", type=int, default=0, help="Optional cap for smoke tests")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
