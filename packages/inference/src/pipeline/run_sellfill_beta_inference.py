#!/usr/bin/env python3
"""Sell-fill beta inference overlay.

Reads the latest full prediction snapshot from a base model (default: gept_v2),
re-scores rows with a trained sell-fill TorchScript model, and writes a fresh
prediction snapshot under a new beta model_id.

This allows the existing engine/web beta-model toggle to use the sell-fill model
without changing API contracts.
"""

from __future__ import annotations

import argparse
import io
import json
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


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactMeta:
    feature_cols: List[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    n_items: int


def _load_artifact_meta(artifact_dir: str) -> ArtifactMeta:
    p = os.path.join(artifact_dir, "artifact.json")
    with open(p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = [str(x) for x in meta["feature_cols"]]
    mean = np.asarray(meta["feature_mean"], dtype=np.float32)
    std = np.asarray(meta["feature_std"], dtype=np.float32)
    std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
    n_items = int(meta["model_config"]["n_items"])
    return ArtifactMeta(feature_cols=feature_cols, feature_mean=mean, feature_std=std, n_items=n_items)


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
    """Load rows from a recent complete snapshot for base_model_id.

    Uses a small recent time window for speed, selecting the newest snapshot with at
    least min_snapshot_items, else falls back to the newest available snapshot.
    """
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


def _build_latest_feature_vectors(
    conn,
    item_ids: List[int],
    feature_cols: List[str],
    lookback_days: int = 10,
    log_every: int = 50,
) -> Dict[int, np.ndarray]:
    fe = FeatureEngine(granularity=Granularity.FIVE_MIN)
    out: Dict[int, np.ndarray] = {}

    for i, item_id in enumerate(item_ids, start=1):
        try:
            df = _load_item_price_5m(conn, item_id=item_id, lookback_days=lookback_days)
            if df.empty:
                continue
            df_feat = fe.compute_features(df)
            if df_feat.empty:
                continue
            row = df_feat.iloc[-1]
            vals: List[float] = []
            for c in feature_cols:
                v = row[c] if c in row.index else 0.0
                if pd.isna(v) or np.isinf(v):
                    v = 0.0
                vals.append(float(v))
            out[int(item_id)] = np.asarray(vals, dtype=np.float32)
        except Exception:
            continue

        if log_every and (i == 1 or i % int(log_every) == 0 or i == len(item_ids)):
            logger.info("features: %s/%s items ready=%s", i, len(item_ids), len(out))

    return out


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


def _confidence_from_prob(p: float) -> str:
    if p >= 0.80:
        return "high"
    if p >= 0.60:
        return "medium"
    return "low"


def run(args) -> int:
    start = datetime.now(timezone.utc)
    logger.info("SELL-FILL BETA INFERENCE START")

    conn = _get_db_connection()
    try:
        base_df = _load_latest_full_base_rows(
            conn,
            base_model_id=args.base_model_id,
            lookback_hours=args.snapshot_lookback_hours,
            recent_snapshots=args.recent_snapshots,
            min_snapshot_items=args.min_snapshot_items,
        )
        logger.info("loaded base rows=%s model_id=%s", len(base_df), args.base_model_id)

        item_ids = sorted(set(int(x) for x in base_df["item_id"].tolist()))
        artifact = _load_artifact_meta(args.artifact_dir)
        logger.info("artifact n_features=%s n_items=%s", len(artifact.feature_cols), artifact.n_items)

        item_vec = _build_latest_feature_vectors(
            conn=conn,
            item_ids=item_ids,
            feature_cols=artifact.feature_cols,
            lookback_days=args.feature_lookback_days,
            log_every=args.log_every,
        )
        if not item_vec:
            raise RuntimeError("no item feature vectors available")

        model_path = os.path.join(args.artifact_dir, "model_torchscript.pt")
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()

        now = datetime.now(timezone.utc)
        build_rows: List[dict] = []

        valid_mask = []
        x_rows = []
        item_rows = []
        off_rows = []
        hor_rows = []

        # Build tensors in row order.
        for _, r in base_df.iterrows():
            item_id = int(r["item_id"])
            h = int(r["hour_offset"])
            if h < 1 or h > 24:
                valid_mask.append(False)
                continue
            vec = item_vec.get(item_id)
            if vec is None:
                valid_mask.append(False)
                continue
            offset = float(r["offset_pct"])

            x = (vec - artifact.feature_mean) / artifact.feature_std
            item_idx = min(max(item_id, 0), artifact.n_items - 1)

            x_rows.append(x)
            item_rows.append(item_idx)
            off_rows.append([offset])
            hor_rows.append([float(h) / 24.0])
            valid_mask.append(True)

        if not x_rows:
            raise RuntimeError("no valid base rows after feature/horizon checks")

        x_t = torch.from_numpy(np.asarray(x_rows, dtype=np.float32))
        item_t = torch.from_numpy(np.asarray(item_rows, dtype=np.int64))
        off_t = torch.from_numpy(np.asarray(off_rows, dtype=np.float32))
        hor_t = torch.from_numpy(np.asarray(hor_rows, dtype=np.float32))

        with torch.no_grad():
            logits = model(x_t, item_t, off_t, hor_t)
            probs = torch.sigmoid(logits).numpy().astype(np.float32)
        probs = np.clip(probs, 0.01, 0.99)

        # Rebuild output rows (same order as base_df filtered by valid rows).
        p_i = 0
        for idx, (_, r) in enumerate(base_df.iterrows()):
            if not valid_mask[idx]:
                continue
            p = float(probs[p_i])
            p_i += 1

            buy = float(r["buy_price"]) if r["buy_price"] is not None else 0.0
            sell = float(r["sell_price"]) if r["sell_price"] is not None else 0.0
            if buy > 0:
                profit_pct = (sell - buy) / buy
            else:
                profit_pct = 0.0
            ev = p * profit_pct

            h = int(r["hour_offset"])
            row = {
                "time": now,
                "item_id": int(r["item_id"]),
                "item_name": str(r["item_name"]) if r["item_name"] is not None else "",
                "hour_offset": h,
                "target_hour": now + timedelta(hours=h),
                "offset_pct": float(r["offset_pct"]),
                "fill_probability": p,
                "expected_value": float(ev),
                "buy_price": buy,
                "sell_price": sell,
                "current_high": float(r["current_high"]) if r["current_high"] is not None else None,
                "current_low": float(r["current_low"]) if r["current_low"] is not None else None,
                "confidence": _confidence_from_prob(p),
                "model_id": args.out_model_id,
            }
            build_rows.append(row)

        written = 0
        if not args.dry_run:
            written = _write_predictions_copy(conn, build_rows, table_name=args.table)
        logger.info(
            "complete rows_written=%s candidate=%s base_rows=%s feature_items=%s model_id=%s dry_run=%s",
            written,
            len(build_rows),
            len(base_df),
            len(item_vec),
            args.out_model_id,
            args.dry_run,
        )
    finally:
        conn.close()

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info("SELL-FILL BETA INFERENCE DONE elapsed=%.1fs", elapsed)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Sell-fill beta inference overlay writer")
    ap.add_argument("--artifact-dir", required=True, help="Path containing artifact.json + model_torchscript.pt")
    ap.add_argument("--base-model-id", default="gept_v2", help="Base model_id snapshot to rescore")
    ap.add_argument("--out-model-id", default="sellfill_beta_v1", help="model_id to write for beta selection")
    ap.add_argument("--table", default="predictions", help="Target table")
    ap.add_argument("--recent-snapshots", type=int, default=60, help="How many recent base snapshots to inspect")
    ap.add_argument("--snapshot-lookback-hours", type=int, default=2)
    ap.add_argument("--min-snapshot-items", type=int, default=300)
    ap.add_argument("--feature-lookback-days", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
