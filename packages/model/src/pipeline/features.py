"""
Feature Computation Module
==========================

Compute features at three time resolutions for PatchTST model:
- Recent: 5-min, 24h lookback, 6 features
- Medium: 1-hour, 7-day lookback, 10 features
- Long: 4-hour, 30-day lookback, 10 features

All features are normalized relative to current mid-price to be
scale-invariant across items with vastly different price ranges.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.config import DataConfig

logger = logging.getLogger(__name__)


def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad with zeros at start or truncate to target length.

    Padding at start (not end) so most recent data is always at the end,
    preserving temporal alignment for the model.
    """
    if len(arr) >= target_len:
        return arr[-target_len:]
    else:
        padding = np.zeros((target_len - len(arr), arr.shape[1]), dtype=arr.dtype)
        return np.vstack([padding, arr])


def safe_log(x: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Safe log transform: log(x + eps) to handle zeros."""
    return np.log(np.maximum(x, eps))


def compute_recent_features(
    df: pd.DataFrame,
    current_mid: float,
    target_len: int = 288
) -> np.ndarray:
    """
    Compute 5-minute resolution features (24h lookback).

    Features (6):
    1. high_price - normalized by current mid
    2. low_price - normalized by current mid
    3. high_volume - log-transformed
    4. low_volume - log-transformed
    5. spread_ratio - (high - low) / mid
    6. staleness - cumulative missing indicator

    Args:
        df: 5-min data with columns: avg_high_price, avg_low_price,
            high_price_volume, low_price_volume
        current_mid: Current mid price for normalization
        target_len: Output sequence length (default 288 = 24h * 12)

    Returns:
        Array of shape (target_len, 6)
    """
    if len(df) == 0 or current_mid <= 0:
        return np.zeros((target_len, 6), dtype=np.float32)

    df = df.copy()

    # Forward-fill prices, leave volumes as-is
    high_price = df['avg_high_price'].ffill().fillna(current_mid).values
    low_price = df['avg_low_price'].ffill().fillna(current_mid).values
    high_vol = df['high_price_volume'].fillna(0).values
    low_vol = df['low_price_volume'].fillna(0).values

    # Normalize prices relative to current mid
    high_norm = high_price / current_mid
    low_norm = low_price / current_mid

    # Log-transform volumes
    high_vol_log = safe_log(high_vol)
    low_vol_log = safe_log(low_vol)

    # Spread ratio
    mid = (high_price + low_price) / 2
    spread_ratio = np.where(mid > 0, (high_price - low_price) / mid, 0)

    # Staleness: cumulative count of originally missing values
    was_missing = df['avg_high_price'].isna()
    staleness = was_missing.cumsum().values.astype(np.float32)

    features = np.column_stack([
        high_norm,
        low_norm,
        high_vol_log,
        low_vol_log,
        spread_ratio,
        staleness
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def compute_medium_features(
    df: pd.DataFrame,
    current_mid: float,
    target_len: int = 168
) -> np.ndarray:
    """
    Compute 1-hour resolution features (7-day lookback).

    Features (10):
    1. high_close - normalized
    2. low_close - normalized
    3. high_volume - log
    4. low_volume - log
    5. spread - (high_close - low_close) / mid
    6. sample_count - samples per bar (12 expected)
    7. staleness - expected - actual sample count
    8. high_range - (high_high - high_low) / mid
    9. low_range - (low_high - low_low) / mid
    10. total_volume - log(high_vol + low_vol)

    Args:
        df: 1-hour OHLC data from stage1
        current_mid: Current mid price for normalization
        target_len: Output sequence length (default 168 = 7 days * 24)

    Returns:
        Array of shape (target_len, 10)
    """
    if len(df) == 0 or current_mid <= 0:
        return np.zeros((target_len, 10), dtype=np.float32)

    df = df.copy()

    # Get columns with defaults
    high_close = df.get('high_close', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    low_close = df.get('low_close', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    high_vol = df.get('high_volume', pd.Series(0, index=df.index)).fillna(0).values
    low_vol = df.get('low_volume', pd.Series(0, index=df.index)).fillna(0).values
    sample_count = df.get('sample_count', pd.Series(12, index=df.index)).fillna(12).values
    high_high = df.get('high_high', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    high_low = df.get('high_low', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    low_high = df.get('low_high', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    low_low = df.get('low_low', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values

    # Normalize prices
    high_close_norm = high_close / current_mid
    low_close_norm = low_close / current_mid

    # Log volumes
    high_vol_log = safe_log(high_vol)
    low_vol_log = safe_log(low_vol)

    # Spread
    mid = (high_close + low_close) / 2
    spread = np.where(mid > 0, (high_close - low_close) / mid, 0)

    # Staleness (expected 12 samples per hour)
    staleness = 12.0 - sample_count

    # Ranges normalized
    high_range = np.where(mid > 0, (high_high - high_low) / mid, 0)
    low_range = np.where(mid > 0, (low_high - low_low) / mid, 0)

    # Total volume
    total_vol_log = safe_log(high_vol + low_vol)

    features = np.column_stack([
        high_close_norm,
        low_close_norm,
        high_vol_log,
        low_vol_log,
        spread,
        sample_count,
        staleness,
        high_range,
        low_range,
        total_vol_log
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def compute_long_features(
    df: pd.DataFrame,
    current_mid: float,
    target_len: int = 180
) -> np.ndarray:
    """
    Compute 4-hour resolution features (30-day lookback).

    Same 10 features as medium resolution.

    Args:
        df: 4-hour OHLC data from stage1
        current_mid: Current mid price for normalization
        target_len: Output sequence length (default 180 = 30 days * 6)

    Returns:
        Array of shape (target_len, 10)
    """
    # Same feature computation as medium, just different resolution
    return compute_medium_features(df, current_mid, target_len)


def compute_targets(
    df_5min: pd.DataFrame,
    timestamp: pd.Timestamp,
    current_mid: float,
    horizons: Tuple[int, ...] = (1, 2, 4, 8, 12, 24, 48)
) -> np.ndarray:
    """
    Compute target values for each prediction horizon.

    Targets are percentage movements relative to current mid price:
    - max_high: max(high_price) over horizon / current_mid - 1
    - min_low: min(low_price) over horizon / current_mid - 1

    Args:
        df_5min: 5-minute price data (must extend beyond timestamp)
        timestamp: Current timestamp
        current_mid: Current mid price
        horizons: Tuple of horizon hours (default: 1, 2, 4, 8, 12, 24, 48)

    Returns:
        Array of shape (n_horizons, 2) with (max_high_pct, min_low_pct)
    """
    if current_mid <= 0:
        return np.zeros((len(horizons), 2), dtype=np.float32)

    targets = []

    for horizon in horizons:
        end_time = timestamp + pd.Timedelta(hours=horizon)

        # Get future window
        if 'timestamp' in df_5min.columns:
            mask = (df_5min['timestamp'] > timestamp) & (df_5min['timestamp'] <= end_time)
            window = df_5min[mask]
        else:
            # Assume timestamp is index
            window = df_5min[(df_5min.index > timestamp) & (df_5min.index <= end_time)]

        if len(window) > 0:
            max_high = window['avg_high_price'].max()
            min_low = window['avg_low_price'].min()

            # Percentage change from current mid
            max_high_pct = (max_high / current_mid - 1) if pd.notna(max_high) else 0.0
            min_low_pct = (min_low / current_mid - 1) if pd.notna(min_low) else 0.0
        else:
            max_high_pct = 0.0
            min_low_pct = 0.0

        targets.append([max_high_pct, min_low_pct])

    return np.array(targets, dtype=np.float32)


def compute_sample_features(
    df_5min: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    timestamp: pd.Timestamp,
    config: Optional[DataConfig] = None
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute all features for a single sample.

    Args:
        df_5min: 5-minute data for the item (timestamp as column or index)
        df_1h: 1-hour OHLC data for the item
        df_4h: 4-hour OHLC data for the item
        timestamp: Sample timestamp
        config: DataConfig with lookback/horizon settings

    Returns:
        Dictionary with 'recent', 'medium', 'long', 'targets', 'current_mid'
        or None if computation fails
    """
    if config is None:
        config = DataConfig()

    try:
        # Get current price
        if 'timestamp' in df_5min.columns:
            current_row = df_5min[df_5min['timestamp'] == timestamp]
            if len(current_row) == 0:
                return None
            current_high = current_row['avg_high_price'].iloc[0]
            current_low = current_row['avg_low_price'].iloc[0]
        else:
            if timestamp not in df_5min.index:
                return None
            current_high = df_5min.loc[timestamp, 'avg_high_price']
            current_low = df_5min.loc[timestamp, 'avg_low_price']

        current_mid = (current_high + current_low) / 2
        if current_mid <= 0 or pd.isna(current_mid):
            return None

        # Recent features (5-min, 24h lookback)
        recent_start = timestamp - pd.Timedelta(hours=config.recent_hours)
        if 'timestamp' in df_5min.columns:
            recent_df = df_5min[(df_5min['timestamp'] > recent_start) & (df_5min['timestamp'] <= timestamp)]
        else:
            recent_df = df_5min[(df_5min.index > recent_start) & (df_5min.index <= timestamp)]
        recent = compute_recent_features(recent_df, current_mid, config.recent_len)

        # Medium features (1-hour, 7-day lookback)
        medium_start = timestamp - pd.Timedelta(days=config.medium_days)
        hour_ts = timestamp.floor('h')
        if 'timestamp' in df_1h.columns:
            medium_df = df_1h[(df_1h['timestamp'] > medium_start) & (df_1h['timestamp'] <= hour_ts)]
        else:
            medium_df = df_1h[(df_1h.index > medium_start) & (df_1h.index <= hour_ts)]
        medium = compute_medium_features(medium_df, current_mid, config.medium_len)

        # Long features (4-hour, 30-day lookback)
        long_start = timestamp - pd.Timedelta(days=config.long_days)
        hour = timestamp.hour
        block_hour = hour - (hour % 4)
        block_ts = timestamp.replace(hour=block_hour, minute=0, second=0, microsecond=0)
        if 'timestamp' in df_4h.columns:
            long_df = df_4h[(df_4h['timestamp'] > long_start) & (df_4h['timestamp'] <= block_ts)]
        else:
            long_df = df_4h[(df_4h.index > long_start) & (df_4h.index <= block_ts)]
        long_seq = compute_long_features(long_df, current_mid, config.long_len)

        # Targets
        targets = compute_targets(df_5min, timestamp, current_mid, config.horizons)

        return {
            'recent': recent,
            'medium': medium,
            'long': long_seq,
            'targets': targets,
            'current_mid': np.float32(current_mid),
            'current_high': np.float32(current_high),
            'current_low': np.float32(current_low),
        }

    except Exception as e:
        logger.debug(f"Feature computation failed for {timestamp}: {e}")
        return None
