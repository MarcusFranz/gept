from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import warnings

import numpy as np
import pandas as pd

from src.db_utils import get_simple_connection
from src.feature_engine import FeatureEngine, Granularity

from .config import FillProbConfig
from .labels import build_sell_label_tensor, compute_future_max_high


def get_tradeable_items(
    conn,
    min_volume: int = 500,
    lookback_days: int = 7,
    limit: Optional[int] = None,
) -> list[int]:
    """Return item_ids with sufficient recent trading volume and history."""
    sql = f"""
    SELECT item_id
    FROM (
        SELECT p.item_id
        FROM price_data_5min p
        WHERE p.timestamp > NOW() - INTERVAL '{int(lookback_days)} days'
          AND (p.high_price_volume + p.low_price_volume) > %s
          AND p.avg_high_price IS NOT NULL
          AND p.avg_low_price IS NOT NULL
        GROUP BY p.item_id
        HAVING COUNT(*) > 500
    ) sub
    ORDER BY item_id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    cur = conn.cursor()
    cur.execute(sql, (int(min_volume),))
    rows = cur.fetchall()
    cur.close()
    return [int(r[0]) for r in rows]


def get_items_from_latest_predictions(
    conn,
    model_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[int]:
    """Return item_ids that appear in the most recent *full* predictions batch.

    This matches "the same items we have now" as defined by whatever is currently
    writing to the `predictions` table in production.
    """
    # The predictions writer can insert partial batches (smaller item counts) a few seconds
    # before the "full" batch timestamp. Using MAX(time) can therefore lock onto an
    # incomplete universe and shrink coverage unexpectedly.
    #
    # We avoid scanning a large time window by only considering the most recent N distinct
    # timestamps and choosing the one with the largest distinct item_id count.
    # This yields a stable "current items" universe without needing an explicit batch_id.
    sql = f"""
    WITH recent_times AS (
      SELECT DISTINCT time
      FROM predictions
      WHERE time > NOW() - INTERVAL '7 days'
        {'AND model_id = %s' if model_id else ''}
      ORDER BY time DESC
      LIMIT 60
    ),
    counts AS (
      SELECT p.time, COUNT(DISTINCT p.item_id) AS n_items
      FROM predictions p
      JOIN recent_times rt ON p.time = rt.time
      WHERE 1=1
        {'AND p.model_id = %s' if model_id else ''}
      GROUP BY p.time
    ),
    best AS (
      SELECT time
      FROM counts
      ORDER BY n_items DESC, time DESC
      LIMIT 1
    )
    SELECT DISTINCT p.item_id
    FROM predictions p
    JOIN best b ON p.time = b.time
    WHERE 1=1
      {'AND p.model_id = %s' if model_id else ''}
    ORDER BY p.item_id
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    cur = conn.cursor()
    if model_id:
        # model_id placeholder appears 3 times: recent_times, counts, final select.
        cur.execute(sql, (model_id, model_id, model_id))
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    return [int(r[0]) for r in rows]


def load_price_data_5min(
    conn,
    item_id: int,
    start_ts: datetime,
    end_ts: datetime,
) -> pd.DataFrame:
    """Load 5-minute data for one item between [start_ts, end_ts]."""
    sql = """
    SELECT timestamp, avg_high_price, avg_low_price, high_price_volume, low_price_volume
    FROM price_data_5min
    WHERE item_id = %s
      AND timestamp >= %s
      AND timestamp <= %s
      AND avg_high_price IS NOT NULL
      AND avg_low_price IS NOT NULL
    ORDER BY timestamp ASC
    """
    # Pandas warns when passed a raw DB-API connection; it's safe here.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pandas only supports SQLAlchemy connectable*",
            category=UserWarning,
        )
        df = pd.read_sql(sql, conn, params=(int(item_id), start_ts, end_ts))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _safe_float32_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    x = df.loc[:, list(cols)].copy()
    # Replace inf and NaN with zeros. (We normalize later.)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32, copy=True)


def _timestamps_to_unix_seconds(ts: pd.Series) -> np.ndarray:
    """Convert timestamp series to unix seconds across pandas datetime units.

    Depending on pandas/pyarrow versions, integer datetime casts may be in ns/us/ms/s.
    Detect the unit by magnitude and scale to seconds robustly.
    """
    raw = pd.to_datetime(ts, utc=True).astype("int64").to_numpy(dtype=np.int64, copy=False)
    if raw.size == 0:
        return raw

    max_abs = int(np.max(np.abs(raw)))
    # Typical epoch magnitudes: ns~1e18, us~1e15, ms~1e12, s~1e9.
    if max_abs >= 100_000_000_000_000_000:  # >= 1e17 => ns
        scale = 1_000_000_000
    elif max_abs >= 100_000_000_000_000:  # >= 1e14 => us
        scale = 1_000_000
    elif max_abs >= 100_000_000_000:  # >= 1e11 => ms
        scale = 1_000
    else:
        scale = 1
    return (raw // scale).astype(np.int64, copy=False)


@dataclass(frozen=True)
class SellFillDatasetArrays:
    """In-memory dataset arrays for sell-fill modeling."""

    x: np.ndarray  # (n, n_features) float32
    item_id: np.ndarray  # (n,) int32
    ts: np.ndarray  # (n,) int64 unix seconds
    y: np.ndarray  # (n, n_horizons, n_offsets) uint8
    feature_cols: list[str]


def build_sell_fill_dataset(
    conn,
    item_ids: Sequence[int],
    cfg: FillProbConfig,
    lookback_days: int,
    sample_every: int = 3,
    max_rows_per_item: Optional[int] = None,
    seed: int = 1337,
    log_every: int = 25,
) -> SellFillDatasetArrays:
    """Build a sell-fill dataset from TimescaleDB using existing price history.

    This uses only existing data (price history) and does not require any new model.
    """
    rng = np.random.default_rng(seed)

    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=int(lookback_days))

    fe = FeatureEngine(granularity=Granularity.FIVE_MIN)
    feature_cols = fe.get_feature_columns()

    xs: list[np.ndarray] = []
    item_ids_out: list[np.ndarray] = []
    ts_out: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    max_periods = max(cfg.horizon_periods.values())

    start_wall = datetime.now(timezone.utc)
    n_failed = 0
    n_skipped = 0

    for i, item_id in enumerate(item_ids, start=1):
        try:
            df = load_price_data_5min(conn, int(item_id), start_ts, end_ts)
            if df.empty:
                n_skipped += 1
                continue

            # Compute features (past-only).
            df_feat = fe.compute_features(df)

            # Build labels from future extrema.
            high = df_feat["avg_high_price"].ffill()
            # Only compute the future max(high) needed for sell-fill labels.
            future_max_high_by_h = compute_future_max_high(high=high, horizon_periods=cfg.horizon_periods)
            y = build_sell_label_tensor(
                current_high=high,
                future_max_high_by_h=future_max_high_by_h,
                offsets=cfg.offsets,
                horizons_hours=cfg.horizons_hours,
            )

            # Drop rows too close to the end to have labels for the max horizon.
            if len(df_feat) <= max_periods + 10:
                n_skipped += 1
                continue
            valid_n = len(df_feat) - max_periods
            df_feat = df_feat.iloc[:valid_n].reset_index(drop=True)
            y = y[:valid_n]

            # Sampling to keep dataset size manageable.
            if sample_every and sample_every > 1:
                df_feat = df_feat.iloc[:: int(sample_every)].reset_index(drop=True)
                y = y[:: int(sample_every)]

            if max_rows_per_item is not None and len(df_feat) > int(max_rows_per_item):
                idx = rng.choice(len(df_feat), size=int(max_rows_per_item), replace=False)
                idx.sort()
                df_feat = df_feat.iloc[idx].reset_index(drop=True)
                y = y[idx]

            x = _safe_float32_matrix(df_feat, feature_cols)
            ts = _timestamps_to_unix_seconds(df_feat["timestamp"])
            item_arr = np.full((len(df_feat),), int(item_id), dtype=np.int32)

            xs.append(x)
            item_ids_out.append(item_arr)
            ts_out.append(ts)
            ys.append(y.astype(np.uint8, copy=False))

        except Exception:
            n_failed += 1
            continue

        if log_every and (i == 1 or i % int(log_every) == 0 or i == len(item_ids)):
            elapsed_s = (datetime.now(timezone.utc) - start_wall).total_seconds()
            rate = i / max(elapsed_s, 1e-6)
            remaining = max(0, len(item_ids) - i)
            eta_s = remaining / max(rate, 1e-6)
            total_rows = sum(int(a.shape[0]) for a in xs) if xs else 0
            print(
                f"[build_sell_fill_dataset] items={i}/{len(item_ids)} "
                f"failed={n_failed} skipped={n_skipped} rows={total_rows} "
                f"rate={rate:.2f} items/s eta={eta_s/60.0:.1f} min",
                flush=True,
            )

    if not xs:
        raise RuntimeError("No dataset rows produced (check item_ids / DB connectivity).")

    x_all = np.concatenate(xs, axis=0)
    item_all = np.concatenate(item_ids_out, axis=0)
    ts_all = np.concatenate(ts_out, axis=0)
    y_all = np.concatenate(ys, axis=0)

    # Sort by timestamp for time-based splits.
    order = np.argsort(ts_all)
    x_all = x_all[order]
    item_all = item_all[order]
    ts_all = ts_all[order]
    y_all = y_all[order]

    return SellFillDatasetArrays(
        x=x_all,
        item_id=item_all,
        ts=ts_all,
        y=y_all,
        feature_cols=list(feature_cols),
    )


def save_dataset_npz(ds: SellFillDatasetArrays, cfg: FillProbConfig, out_path: str) -> None:
    """Save dataset + config to an NPZ file."""
    np.savez_compressed(
        out_path,
        x=ds.x,
        item_id=ds.item_id,
        ts=ds.ts,
        y=ds.y,
        feature_cols=np.asarray(ds.feature_cols, dtype=object),
        horizons_hours=np.asarray(list(cfg.horizons_hours), dtype=np.int32),
        offsets=np.asarray(list(cfg.offsets), dtype=np.float32),
        granularity=np.asarray([cfg.granularity], dtype=object),
    )


def load_dataset_npz(path: str) -> tuple[SellFillDatasetArrays, FillProbConfig]:
    arr = np.load(path, allow_pickle=True)
    feature_cols = [str(x) for x in arr["feature_cols"].tolist()]
    horizons = [int(x) for x in arr["horizons_hours"].tolist()]
    offsets = [float(x) for x in arr["offsets"].tolist()]
    granularity = str(arr["granularity"].tolist()[0])
    cfg = FillProbConfig(horizons_hours=tuple(horizons), offsets=tuple(offsets), granularity=granularity)
    ds = SellFillDatasetArrays(
        x=arr["x"].astype(np.float32, copy=False),
        item_id=arr["item_id"].astype(np.int32, copy=False),
        ts=arr["ts"].astype(np.int64, copy=False),
        y=arr["y"].astype(np.uint8, copy=False),
        feature_cols=feature_cols,
    )
    return ds, cfg


def build_and_save_dataset_npz(
    out_path: str,
    min_volume: int = 500,
    lookback_days: int = 60,
    max_items: int = 200,
    sample_every: int = 3,
    max_rows_per_item: Optional[int] = None,
    seed: int = 1337,
    cfg: Optional[FillProbConfig] = None,
    item_source: str = "tradeable",
    predictions_model_id: Optional[str] = None,
    log_every: int = 25,
) -> str:
    """Convenience wrapper for scripts."""
    cfg = cfg or FillProbConfig()
    conn = get_simple_connection()
    try:
        if item_source == "predictions":
            item_ids = get_items_from_latest_predictions(conn, model_id=predictions_model_id, limit=max_items)
        elif item_source == "tradeable":
            item_ids = get_tradeable_items(conn, min_volume=min_volume, lookback_days=7, limit=max_items)
        else:
            raise ValueError(f"Unknown item_source: {item_source} (expected 'tradeable' or 'predictions')")
        ds = build_sell_fill_dataset(
            conn=conn,
            item_ids=item_ids,
            cfg=cfg,
            lookback_days=lookback_days,
            sample_every=sample_every,
            max_rows_per_item=max_rows_per_item,
            seed=seed,
            log_every=log_every,
        )
    finally:
        conn.close()

    save_dataset_npz(ds, cfg, out_path)
    return out_path
