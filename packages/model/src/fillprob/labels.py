from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def _future_rolling_max(series: pd.Series, periods: int) -> pd.Series:
    """Max over the *next* `periods` points (excluding current), aligned at current index."""
    # rolling max looks backwards; shift(-periods) aligns it to "current".
    return series.rolling(periods, min_periods=1).max().shift(-periods)


def _future_rolling_min(series: pd.Series, periods: int) -> pd.Series:
    """Min over the *next* `periods` points (excluding current), aligned at current index."""
    return series.rolling(periods, min_periods=1).min().shift(-periods)


def compute_future_extrema(
    high: pd.Series,
    low: pd.Series,
    horizon_periods: Dict[int, int],
) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
    """Precompute future max(high) and min(low) for each horizon.

    Returns:
        (future_max_high, future_min_low): each maps horizon_hours -> Series aligned to current index.
    """
    future_max_high: dict[int, pd.Series] = {}
    future_min_low: dict[int, pd.Series] = {}
    for hours, periods in horizon_periods.items():
        future_max_high[hours] = _future_rolling_max(high, periods)
        future_min_low[hours] = _future_rolling_min(low, periods)
    return future_max_high, future_min_low


def compute_future_max_high(
    high: pd.Series,
    horizon_periods: Dict[int, int],
) -> dict[int, pd.Series]:
    """Precompute future max(high) for each horizon."""
    out: dict[int, pd.Series] = {}
    for hours, periods in horizon_periods.items():
        out[hours] = _future_rolling_max(high, periods)
    return out


def sell_fills_within_horizon(
    current_high: pd.Series,
    future_max_high: pd.Series,
    offset: float,
) -> np.ndarray:
    """Binary label: would a sell at current_high*(1+offset) fill within horizon?"""
    target = current_high * (1.0 + float(offset))
    # `future_max_high` can be NaN near the end; treat as not-fill.
    return (future_max_high >= target).fillna(False).astype(np.int8).to_numpy()


def buy_fills_within_horizon(
    current_low: pd.Series,
    future_min_low: pd.Series,
    offset: float,
) -> np.ndarray:
    """Binary label: would a buy at current_low*(1-offset) fill within horizon?"""
    target = current_low * (1.0 - float(offset))
    return (future_min_low <= target).fillna(False).astype(np.int8).to_numpy()


def build_sell_label_tensor(
    current_high: pd.Series,
    future_max_high_by_h: dict[int, pd.Series],
    offsets: Sequence[float],
    horizons_hours: Sequence[int],
) -> np.ndarray:
    """Return labels shaped (n_rows, n_horizons, n_offsets) for sell fills."""
    # Vectorize across offsets for speed.
    cur = current_high.to_numpy(dtype=np.float32, copy=False)
    offs = np.asarray(list(offsets), dtype=np.float32)
    targets = cur.reshape(-1, 1) * (1.0 + offs.reshape(1, -1))

    y = np.zeros((len(cur), len(horizons_hours), len(offs)), dtype=np.int8)
    for hi, h in enumerate(horizons_hours):
        fut = future_max_high_by_h[h].to_numpy(dtype=np.float32, copy=False)
        # NaNs appear near the end due to shift(-periods); treat as not-fill.
        fut = np.nan_to_num(fut, nan=-np.inf)
        y[:, hi, :] = (fut.reshape(-1, 1) >= targets).astype(np.int8)
    return y


def build_buy_label_tensor(
    current_low: pd.Series,
    future_min_low_by_h: dict[int, pd.Series],
    offsets: Sequence[float],
    horizons_hours: Sequence[int],
) -> np.ndarray:
    """Return labels shaped (n_rows, n_horizons, n_offsets) for buy fills."""
    n = len(current_low)
    y = np.zeros((n, len(horizons_hours), len(offsets)), dtype=np.int8)
    for hi, h in enumerate(horizons_hours):
        future_min_low = future_min_low_by_h[h]
        for oi, o in enumerate(offsets):
            y[:, hi, oi] = buy_fills_within_horizon(current_low, future_min_low, o)
    return y
