from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.db_utils import get_simple_connection
from src.feature_engine import FeatureEngine, Granularity
from src.fillprob.config import FillProbConfig
from src.fillprob.dataset import (
    get_items_from_latest_predictions,
    get_tradeable_items,
    load_price_data_5min,
)
from src.fillprob.labels import (
    build_buy_label_tensor,
    build_sell_label_tensor,
    compute_future_extrema,
    compute_future_max_high,
)


@dataclass(frozen=True)
class GePT3RowDatasetArrays:
    x: np.ndarray  # (n, n_features) float32
    item_id: np.ndarray  # (n,) int32
    ts: np.ndarray  # (n,) int64 unix seconds
    y_sell: np.ndarray  # (n, n_horizons, n_offsets) uint8
    y_buy: np.ndarray  # (n, n_horizons, n_offsets) uint8
    y_price_high: np.ndarray  # (n, n_horizons) float32
    y_price_low: np.ndarray  # (n, n_horizons) float32
    y_regime: np.ndarray  # (n,) uint8
    feature_cols: list[str]


def _safe_float32_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    x = df.loc[:, list(cols)].copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=np.float32, copy=True)


def _timestamps_to_unix_seconds(ts: pd.Series) -> np.ndarray:
    raw = pd.to_datetime(ts, utc=True).astype("int64").to_numpy(dtype=np.int64, copy=False)
    if raw.size == 0:
        return raw

    max_abs = int(np.max(np.abs(raw)))
    if max_abs >= 100_000_000_000_000_000:
        scale = 1_000_000_000
    elif max_abs >= 100_000_000_000_000:
        scale = 1_000_000
    elif max_abs >= 100_000_000_000:
        scale = 1_000
    else:
        scale = 1
    return (raw // scale).astype(np.int64, copy=False)


def _compute_price_delta_labels(
    high: pd.Series,
    low: pd.Series,
    future_max_high_by_h: dict[int, pd.Series],
    future_min_low_by_h: dict[int, pd.Series],
    horizons_hours: Sequence[int],
    clip_abs: float,
) -> tuple[np.ndarray, np.ndarray]:
    cur_high = high.to_numpy(dtype=np.float32, copy=False)
    cur_low = low.to_numpy(dtype=np.float32, copy=False)
    denom_high = np.maximum(cur_high, 1e-6)
    denom_low = np.maximum(cur_low, 1e-6)

    n = len(high)
    h_count = len(horizons_hours)
    out_high = np.zeros((n, h_count), dtype=np.float32)
    out_low = np.zeros((n, h_count), dtype=np.float32)

    for hi, h in enumerate(horizons_hours):
        fut_high = future_max_high_by_h[h].to_numpy(dtype=np.float32, copy=False)
        fut_low = future_min_low_by_h[h].to_numpy(dtype=np.float32, copy=False)

        high_delta = np.nan_to_num((fut_high / denom_high) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
        low_delta = np.nan_to_num((fut_low / denom_low) - 1.0, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_abs > 0:
            high_delta = np.clip(high_delta, -clip_abs, clip_abs)
            low_delta = np.clip(low_delta, -clip_abs, clip_abs)

        out_high[:, hi] = high_delta
        out_low[:, hi] = low_delta

    return out_high, out_low


def _compute_regime_labels(
    high: pd.Series,
    low: pd.Series,
    periods: int,
    ret_threshold: float,
    drawdown_threshold: float,
) -> np.ndarray:
    mid = ((high + low) / 2.0).ffill()
    future_mid = mid.shift(-periods)
    future_min_low = low.rolling(periods, min_periods=1).min().shift(-periods)

    ret = (future_mid / mid) - 1.0
    drawdown = (future_min_low / mid) - 1.0

    # "Bad regime" if near-term drift is strongly negative or drawdown is deep.
    y = ((ret <= -abs(ret_threshold)) | (drawdown <= -abs(drawdown_threshold))).fillna(False).astype(np.uint8)
    return y.to_numpy(dtype=np.uint8, copy=False)


def build_gept3_row_dataset(
    conn,
    item_ids: Sequence[int],
    cfg: FillProbConfig,
    lookback_days: int,
    sample_every: int = 1,
    max_rows_per_item: Optional[int] = None,
    seed: int = 1337,
    log_every: int = 25,
    regime_hours: int = 6,
    regime_ret_threshold: float = 0.010,
    regime_drawdown_threshold: float = 0.015,
    price_delta_clip_abs: float = 0.30,
) -> GePT3RowDatasetArrays:
    rng = np.random.default_rng(seed)

    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=int(lookback_days))
    fe = FeatureEngine(granularity=Granularity.FIVE_MIN)
    feature_cols = fe.get_feature_columns()

    xs: list[np.ndarray] = []
    item_ids_out: list[np.ndarray] = []
    ts_out: list[np.ndarray] = []
    y_sell_out: list[np.ndarray] = []
    y_buy_out: list[np.ndarray] = []
    y_price_high_out: list[np.ndarray] = []
    y_price_low_out: list[np.ndarray] = []
    y_regime_out: list[np.ndarray] = []

    max_periods = max(cfg.horizon_periods.values())
    regime_periods = max(1, int(regime_hours) * int(cfg.periods_per_hour))
    min_valid_tail = max(max_periods, regime_periods)

    start_wall = datetime.now(timezone.utc)
    n_failed = 0
    n_skipped = 0

    for i, item_id in enumerate(item_ids, start=1):
        try:
            df = load_price_data_5min(conn, int(item_id), start_ts, end_ts)
            if df.empty:
                n_skipped += 1
                continue

            df_feat = fe.compute_features(df)
            high = df_feat["avg_high_price"].ffill()
            low = df_feat["avg_low_price"].ffill()

            future_max_high_by_h = compute_future_max_high(high=high, horizon_periods=cfg.horizon_periods)
            _, future_min_low_by_h = compute_future_extrema(
                high=high, low=low, horizon_periods=cfg.horizon_periods
            )

            y_sell = build_sell_label_tensor(
                current_high=high,
                future_max_high_by_h=future_max_high_by_h,
                offsets=cfg.offsets,
                horizons_hours=cfg.horizons_hours,
            )
            y_buy = build_buy_label_tensor(
                current_low=low,
                future_min_low_by_h=future_min_low_by_h,
                offsets=cfg.offsets,
                horizons_hours=cfg.horizons_hours,
            )
            y_price_high, y_price_low = _compute_price_delta_labels(
                high=high,
                low=low,
                future_max_high_by_h=future_max_high_by_h,
                future_min_low_by_h=future_min_low_by_h,
                horizons_hours=cfg.horizons_hours,
                clip_abs=float(price_delta_clip_abs),
            )
            y_regime = _compute_regime_labels(
                high=high,
                low=low,
                periods=regime_periods,
                ret_threshold=float(regime_ret_threshold),
                drawdown_threshold=float(regime_drawdown_threshold),
            )

            if len(df_feat) <= min_valid_tail + 10:
                n_skipped += 1
                continue

            valid_n = len(df_feat) - min_valid_tail
            df_feat = df_feat.iloc[:valid_n].reset_index(drop=True)
            y_sell = y_sell[:valid_n]
            y_buy = y_buy[:valid_n]
            y_price_high = y_price_high[:valid_n]
            y_price_low = y_price_low[:valid_n]
            y_regime = y_regime[:valid_n]

            if sample_every and sample_every > 1:
                step = int(sample_every)
                df_feat = df_feat.iloc[::step].reset_index(drop=True)
                y_sell = y_sell[::step]
                y_buy = y_buy[::step]
                y_price_high = y_price_high[::step]
                y_price_low = y_price_low[::step]
                y_regime = y_regime[::step]

            if max_rows_per_item is not None and len(df_feat) > int(max_rows_per_item):
                idx = rng.choice(len(df_feat), size=int(max_rows_per_item), replace=False)
                idx.sort()
                df_feat = df_feat.iloc[idx].reset_index(drop=True)
                y_sell = y_sell[idx]
                y_buy = y_buy[idx]
                y_price_high = y_price_high[idx]
                y_price_low = y_price_low[idx]
                y_regime = y_regime[idx]

            x = _safe_float32_matrix(df_feat, feature_cols)
            ts = _timestamps_to_unix_seconds(df_feat["timestamp"])
            item_arr = np.full((len(df_feat),), int(item_id), dtype=np.int32)

            xs.append(x)
            item_ids_out.append(item_arr)
            ts_out.append(ts)
            y_sell_out.append(y_sell.astype(np.uint8, copy=False))
            y_buy_out.append(y_buy.astype(np.uint8, copy=False))
            y_price_high_out.append(y_price_high.astype(np.float32, copy=False))
            y_price_low_out.append(y_price_low.astype(np.float32, copy=False))
            y_regime_out.append(y_regime.astype(np.uint8, copy=False))
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
                f"[build_gept3_row_dataset] items={i}/{len(item_ids)} failed={n_failed} "
                f"skipped={n_skipped} rows={total_rows} rate={rate:.2f} items/s eta={eta_s/60.0:.1f} min",
                flush=True,
            )

    if not xs:
        raise RuntimeError("No dataset rows produced (check item_ids / DB connectivity).")

    x_all = np.concatenate(xs, axis=0)
    item_all = np.concatenate(item_ids_out, axis=0)
    ts_all = np.concatenate(ts_out, axis=0)
    y_sell_all = np.concatenate(y_sell_out, axis=0)
    y_buy_all = np.concatenate(y_buy_out, axis=0)
    y_price_high_all = np.concatenate(y_price_high_out, axis=0)
    y_price_low_all = np.concatenate(y_price_low_out, axis=0)
    y_regime_all = np.concatenate(y_regime_out, axis=0)

    order = np.argsort(ts_all)
    return GePT3RowDatasetArrays(
        x=x_all[order],
        item_id=item_all[order],
        ts=ts_all[order],
        y_sell=y_sell_all[order],
        y_buy=y_buy_all[order],
        y_price_high=y_price_high_all[order],
        y_price_low=y_price_low_all[order],
        y_regime=y_regime_all[order],
        feature_cols=list(feature_cols),
    )


def save_gept3_row_dataset_npz(
    ds: GePT3RowDatasetArrays,
    cfg: FillProbConfig,
    out_path: str,
    regime_hours: int,
    regime_ret_threshold: float,
    regime_drawdown_threshold: float,
) -> None:
    np.savez_compressed(
        out_path,
        x=ds.x,
        item_id=ds.item_id,
        ts=ds.ts,
        y_sell=ds.y_sell,
        y_buy=ds.y_buy,
        y_price_high=ds.y_price_high,
        y_price_low=ds.y_price_low,
        y_regime=ds.y_regime,
        feature_cols=np.asarray(ds.feature_cols, dtype=object),
        horizons_hours=np.asarray(list(cfg.horizons_hours), dtype=np.int32),
        offsets=np.asarray(list(cfg.offsets), dtype=np.float32),
        granularity=np.asarray([cfg.granularity], dtype=object),
        regime_hours=np.asarray([int(regime_hours)], dtype=np.int32),
        regime_ret_threshold=np.asarray([float(regime_ret_threshold)], dtype=np.float32),
        regime_drawdown_threshold=np.asarray([float(regime_drawdown_threshold)], dtype=np.float32),
    )


def build_and_save_gept3_row_dataset_npz(
    out_path: str,
    min_volume: int = 500,
    lookback_days: int = 180,
    max_items: int = 2_000,
    sample_every: int = 1,
    max_rows_per_item: Optional[int] = None,
    seed: int = 1337,
    cfg: Optional[FillProbConfig] = None,
    item_source: str = "predictions",
    predictions_model_id: Optional[str] = None,
    log_every: int = 25,
    regime_hours: int = 6,
    regime_ret_threshold: float = 0.010,
    regime_drawdown_threshold: float = 0.015,
    price_delta_clip_abs: float = 0.30,
) -> str:
    cfg = cfg or FillProbConfig()
    conn = get_simple_connection()
    try:
        if item_source == "predictions":
            item_ids = get_items_from_latest_predictions(conn, model_id=predictions_model_id, limit=max_items)
        elif item_source == "tradeable":
            item_ids = get_tradeable_items(conn, min_volume=min_volume, lookback_days=7, limit=max_items)
        else:
            raise ValueError(f"Unknown item_source: {item_source} (expected 'tradeable' or 'predictions')")

        ds = build_gept3_row_dataset(
            conn=conn,
            item_ids=item_ids,
            cfg=cfg,
            lookback_days=lookback_days,
            sample_every=sample_every,
            max_rows_per_item=max_rows_per_item,
            seed=seed,
            log_every=log_every,
            regime_hours=regime_hours,
            regime_ret_threshold=regime_ret_threshold,
            regime_drawdown_threshold=regime_drawdown_threshold,
            price_delta_clip_abs=price_delta_clip_abs,
        )
    finally:
        conn.close()

    save_gept3_row_dataset_npz(
        ds=ds,
        cfg=cfg,
        out_path=out_path,
        regime_hours=regime_hours,
        regime_ret_threshold=regime_ret_threshold,
        regime_drawdown_threshold=regime_drawdown_threshold,
    )
    return out_path
