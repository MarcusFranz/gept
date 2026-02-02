#!/usr/bin/env python3
"""
PatchTST Production Inference Script
=====================================

Runs inference on all tradeable items and writes predictions to TimescaleDB.

Features are computed identically to training:
- Recent: 5-min, 24h lookback (288 x 6)
- Medium: 1-hour, 7-day lookback (168 x 10)
- Long: 4-hour, 30-day lookback (180 x 10)

Output mapping:
- Model outputs: high_quantiles, low_quantiles [batch, 7 horizons, 5 quantiles]
- Horizons: [1, 2, 4, 8, 12, 24, 48] hours
- Quantiles: [0.1, 0.25, 0.5, 0.75, 0.9]

Usage:
    python run_patchtst_inference.py --model-path models/patchtst/best_model.pt
"""

import argparse
import io
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Horizons matching training
HORIZONS = [1, 2, 4, 8, 12, 24, 48]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
OFFSET_PCTS = [0.0125, 0.015, 0.0175, 0.02, 0.025]  # 1.25% to 2.5% margins (covers all risk levels)


# =============================================================================
# FEATURE COMPUTATION (matches training pipeline exactly)
# =============================================================================

def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros at start or truncate to target length."""
    if len(arr) >= target_len:
        return arr[-target_len:]
    else:
        padding = np.zeros((target_len - len(arr), arr.shape[1]), dtype=arr.dtype)
        return np.vstack([padding, arr])


def safe_log(x: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Safe log transform: log(x + eps) to handle zeros."""
    return np.log(np.maximum(x, eps))


def compute_recent_features(df: pd.DataFrame, current_mid: float, target_len: int = 288) -> np.ndarray:
    """Compute 5-minute resolution features (24h lookback)."""
    if len(df) == 0 or current_mid <= 0:
        return np.zeros((target_len, 6), dtype=np.float32)

    df = df.copy()
    high_price = df['avg_high_price'].ffill().fillna(current_mid).values
    low_price = df['avg_low_price'].ffill().fillna(current_mid).values
    high_vol = df['high_price_volume'].fillna(0).values
    low_vol = df['low_price_volume'].fillna(0).values

    high_norm = high_price / current_mid
    low_norm = low_price / current_mid
    high_vol_log = safe_log(high_vol)
    low_vol_log = safe_log(low_vol)

    mid = (high_price + low_price) / 2
    spread_ratio = np.where(mid > 0, (high_price - low_price) / mid, 0)
    was_missing = df['avg_high_price'].isna()
    staleness = was_missing.cumsum().values.astype(np.float32)

    features = np.column_stack([
        high_norm, low_norm, high_vol_log, low_vol_log, spread_ratio, staleness
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def aggregate_to_hourly(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min data to 1-hour OHLC format."""
    df = df_5min.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    hourly = df.resample('1h').agg({
        'avg_high_price': ['first', 'max', 'min', 'last'],
        'avg_low_price': ['first', 'max', 'min', 'last'],
        'high_price_volume': 'sum',
        'low_price_volume': 'sum'
    })
    hourly.columns = ['_'.join(col).strip() for col in hourly.columns.values]

    hourly = hourly.rename(columns={
        'avg_high_price_last': 'high_close',
        'avg_high_price_max': 'high_high',
        'avg_high_price_min': 'high_low',
        'avg_low_price_last': 'low_close',
        'avg_low_price_max': 'low_high',
        'avg_low_price_min': 'low_low',
        'high_price_volume_sum': 'high_volume',
        'low_price_volume_sum': 'low_volume'
    })

    sample_counts = df.resample('1h').size()
    hourly['sample_count'] = sample_counts

    return hourly.dropna(subset=['high_close', 'low_close'])


def aggregate_to_4hourly(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min data to 4-hour OHLC format."""
    df = df_5min.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    agg = df.resample('4h').agg({
        'avg_high_price': ['first', 'max', 'min', 'last'],
        'avg_low_price': ['first', 'max', 'min', 'last'],
        'high_price_volume': 'sum',
        'low_price_volume': 'sum'
    })
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]

    agg = agg.rename(columns={
        'avg_high_price_last': 'high_close',
        'avg_high_price_max': 'high_high',
        'avg_high_price_min': 'high_low',
        'avg_low_price_last': 'low_close',
        'avg_low_price_max': 'low_high',
        'avg_low_price_min': 'low_low',
        'high_price_volume_sum': 'high_volume',
        'low_price_volume_sum': 'low_volume'
    })

    sample_counts = df.resample('4h').size()
    agg['sample_count'] = sample_counts

    return agg.dropna(subset=['high_close', 'low_close'])


def compute_medium_features(df: pd.DataFrame, current_mid: float, target_len: int = 168) -> np.ndarray:
    """Compute 1-hour resolution features (7-day lookback)."""
    if len(df) == 0 or current_mid <= 0:
        return np.zeros((target_len, 10), dtype=np.float32)

    df = df.copy()

    high_close = df.get('high_close', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    low_close = df.get('low_close', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    high_vol = df.get('high_volume', pd.Series(0, index=df.index)).fillna(0).values
    low_vol = df.get('low_volume', pd.Series(0, index=df.index)).fillna(0).values
    sample_count = df.get('sample_count', pd.Series(12, index=df.index)).fillna(12).values
    high_high = df.get('high_high', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    high_low = df.get('high_low', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    low_high = df.get('low_high', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values
    low_low = df.get('low_low', pd.Series(current_mid, index=df.index)).ffill().fillna(current_mid).values

    high_close_norm = high_close / current_mid
    low_close_norm = low_close / current_mid
    high_vol_log = safe_log(high_vol)
    low_vol_log = safe_log(low_vol)

    mid = (high_close + low_close) / 2
    spread = np.where(mid > 0, (high_close - low_close) / mid, 0)
    staleness = 12.0 - sample_count
    high_range = np.where(mid > 0, (high_high - high_low) / mid, 0)
    low_range = np.where(mid > 0, (low_high - low_low) / mid, 0)
    total_vol_log = safe_log(high_vol + low_vol)

    features = np.column_stack([
        high_close_norm, low_close_norm, high_vol_log, low_vol_log,
        spread, sample_count, staleness, high_range, low_range, total_vol_log
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def compute_long_features(df: pd.DataFrame, current_mid: float, target_len: int = 180) -> np.ndarray:
    """Compute 4-hour resolution features (30-day lookback)."""
    return compute_medium_features(df, current_mid, target_len)


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection():
    """Get database connection from environment or defaults."""
    return psycopg2.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        port=int(os.environ.get('DB_PORT', 5432)),
        dbname=os.environ.get('DB_NAME', 'osrs_data'),
        user=os.environ.get('DB_USER', 'osrs_user'),
        password=os.environ.get('DB_PASSWORD', 'osrs_price_data_2024')
    )


def get_tradeable_items(conn, min_volume: int = 500) -> pd.DataFrame:
    """Get all items with sufficient trading volume."""
    query = """
    SELECT item_id, name, cnt
    FROM (
        SELECT i.item_id, i.name, COUNT(*) as cnt
        FROM items i
        JOIN price_data_5min p ON i.item_id = p.item_id
        WHERE p.timestamp > NOW() - INTERVAL '7 days'
        AND p.high_price_volume + p.low_price_volume > %s
        GROUP BY i.item_id, i.name
        HAVING COUNT(*) > 500
    ) sub
    ORDER BY cnt DESC
    """
    return pd.read_sql(query, conn, params=(min_volume,))


def load_item_data(conn, item_id: int, lookback_days: int = 31) -> pd.DataFrame:
    """Load 5-min price data for an item."""
    query = """
    SELECT timestamp, avg_high_price, avg_low_price,
           high_price_volume, low_price_volume
    FROM price_data_5min
    WHERE item_id = %s
    AND timestamp > NOW() - INTERVAL '%s days'
    AND avg_high_price IS NOT NULL
    ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn, params=(item_id, lookback_days))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def compute_item_features(df_5min: pd.DataFrame, now: pd.Timestamp) -> Optional[Dict]:
    """Compute all features for a single item at current time."""
    if len(df_5min) < 288:  # Need at least 24h of data
        return None

    # Get current price (most recent row)
    if 'timestamp' in df_5min.columns:
        df_5min = df_5min.set_index('timestamp')

    # Find the most recent timestamp
    current_ts = df_5min.index.max()
    current_row = df_5min.loc[current_ts]

    current_high = current_row['avg_high_price']
    current_low = current_row['avg_low_price']
    current_mid = (current_high + current_low) / 2

    if current_mid <= 0 or pd.isna(current_mid):
        return None

    # Reset index for filtering
    df_5min = df_5min.reset_index()

    # Recent features (24h)
    recent_start = current_ts - pd.Timedelta(hours=24)
    recent_df = df_5min[(df_5min['timestamp'] > recent_start) & (df_5min['timestamp'] <= current_ts)]
    recent = compute_recent_features(recent_df, current_mid, 288)

    # Medium features (7 days)
    medium_start = current_ts - pd.Timedelta(days=7)
    medium_5min = df_5min[(df_5min['timestamp'] > medium_start) & (df_5min['timestamp'] <= current_ts)]
    df_1h = aggregate_to_hourly(medium_5min)
    medium = compute_medium_features(df_1h, current_mid, 168)

    # Long features (30 days)
    long_start = current_ts - pd.Timedelta(days=30)
    long_5min = df_5min[(df_5min['timestamp'] > long_start) & (df_5min['timestamp'] <= current_ts)]
    df_4h = aggregate_to_4hourly(long_5min)
    long_seq = compute_long_features(df_4h, current_mid, 180)

    return {
        'recent': recent,
        'medium': medium,
        'long': long_seq,
        'current_mid': current_mid,
        'current_high': current_high,
        'current_low': current_low,
    }


def convert_predictions_to_rows(
    item_id: int,
    item_name: str,
    current_high: float,
    current_low: float,
    current_mid: float,
    high_quantiles: np.ndarray,  # [7, 5]
    low_quantiles: np.ndarray,   # [7, 5]
    now: datetime,
    model_id: str
) -> List[Dict]:
    """
    Convert model output to prediction rows.

    high_quantiles[h, q] = predicted (max_high / current_mid - 1) at horizon h, quantile q
    low_quantiles[h, q] = predicted (min_low / current_mid - 1) at horizon h, quantile q
    """
    rows = []

    for h_idx, horizon in enumerate(HORIZONS):
        target_hour = now + timedelta(hours=horizon)

        # Get median predictions (quantile index 2 = 0.5)
        high_pct = high_quantiles[h_idx, 2]  # Median max_high prediction
        low_pct = low_quantiles[h_idx, 2]    # Median min_low prediction

        # Convert to absolute prices
        # high_pct = max_high/current_mid - 1, so max_high = current_mid * (1 + high_pct)
        predicted_max_high = current_mid * (1 + high_pct)
        predicted_min_low = current_mid * (1 + low_pct)

        for offset_pct in OFFSET_PCTS:
            # Buy scenario: We place buy offer BELOW current_high, wait for fill,
            # then sell at predicted_max_high (or higher)
            buy_price = current_high * (1 - offset_pct)
            sell_price = predicted_max_high

            # Expected profit percentage
            if buy_price > 0:
                profit_pct = (sell_price - buy_price) / buy_price
            else:
                profit_pct = 0

            # Fill probability estimation based on spread of quantiles
            # If 10th percentile > 0, high confidence the price will rise
            high_q10 = high_quantiles[h_idx, 0]  # 10th percentile
            high_q90 = high_quantiles[h_idx, 4]  # 90th percentile

            # Simple fill probability: how likely is max_high > offset_pct?
            # If even the 10th percentile exceeds offset, very likely to fill
            if high_q10 >= offset_pct:
                fill_prob = 0.90
            elif high_pct >= offset_pct:
                fill_prob = 0.50 + 0.40 * (high_pct - offset_pct) / max(offset_pct, 0.01)
            else:
                fill_prob = 0.10 + 0.40 * max(0, high_pct / offset_pct)

            fill_prob = np.clip(fill_prob, 0.01, 0.99)

            # Expected value = fill_probability * profit - (1 - fill_probability) * cost
            # Simplified: fill_prob * profit_pct
            expected_value = fill_prob * profit_pct

            # Raw quantile spread — confidence assigned later relative to batch
            spread = high_q90 - high_q10

            # Compute auxiliary features for the row
            # (These would ideally come from actual calculations, but we approximate)
            rows.append({
                'time': now,
                'item_id': item_id,
                'item_name': item_name,
                'hour_offset': horizon,
                'target_hour': target_hour,
                'offset_pct': offset_pct,
                'fill_probability': float(fill_prob),
                'expected_value': float(expected_value),
                'buy_price': float(buy_price),
                'sell_price': float(sell_price),
                'current_high': float(current_high),
                'current_low': float(current_low),
                'confidence': None,
                'spread': float(spread),
                'model_id': model_id,
            })

    return rows


def assign_confidence_labels(rows: List[Dict]) -> List[Dict]:
    """Assign confidence labels relative to the batch spread distribution.

    Uses percentile-based thresholds so confidence is distributed across
    high/medium/low rather than clustering in one bucket. Applies an
    absolute floor: if median spread > 0.15, best confidence is capped
    at 'medium' (the whole batch is too uncertain for 'high').
    """
    if not rows:
        return rows

    spreads = np.array([r['spread'] for r in rows])
    p33 = np.percentile(spreads, 33)
    p66 = np.percentile(spreads, 66)
    median_spread = np.median(spreads)

    # If overall uncertainty is very high, cap best label at medium
    cap_high = median_spread > 0.15

    for row in rows:
        s = row['spread']
        if s <= p33:
            row['confidence'] = 'medium' if cap_high else 'high'
        elif s <= p66:
            row['confidence'] = 'medium'
        else:
            row['confidence'] = 'low'
        del row['spread']

    return rows


def write_predictions_copy(conn, rows: List[Dict], table_name: str = 'predictions'):
    """Write predictions using COPY protocol for speed.

    Note: We INSERT without DELETE - predictions accumulate over time and
    can be queried by timestamp. Old predictions are cleaned up by
    TimescaleDB retention policies or scheduled jobs.
    """
    if not rows:
        return 0

    columns = [
        'time', 'item_id', 'item_name', 'hour_offset', 'target_hour',
        'offset_pct', 'fill_probability', 'expected_value',
        'buy_price', 'sell_price', 'current_high', 'current_low',
        'confidence', 'model_id'
    ]

    # Build CSV buffer
    buffer = io.StringIO()
    for row in rows:
        values = [str(row[col]) if row[col] is not None else '' for col in columns]
        buffer.write('\t'.join(values) + '\n')
    buffer.seek(0)

    cur = conn.cursor()

    # Copy new predictions (no delete - historical predictions are kept)
    cur.copy_from(buffer, table_name, columns=columns, null='')
    conn.commit()
    cur.close()

    return len(rows)


# =============================================================================
# MAIN INFERENCE
# =============================================================================

def load_model(model_path: str, device: torch.device):
    """Load the PatchTST model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config_dict = checkpoint.get('config', {})
    model_config_dict = config_dict.get('model', {})

    # Import model classes
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.model import PatchTSTModel, ModelConfig

    model_config = ModelConfig(
        n_items=model_config_dict.get('n_items', 1200),
        item_embed_dim=model_config_dict.get('item_embed_dim', 32),
        d_model=model_config_dict.get('d_model', 384),
        n_heads=model_config_dict.get('n_heads', 8),
        n_layers=model_config_dict.get('n_layers', 5),
        dropout=model_config_dict.get('dropout', 0.1),
        enable_volume_head=model_config_dict.get('enable_volume_head', True),
    )

    model = PatchTSTModel(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train(False)

    return model


def run_inference(args):
    """Main inference function."""
    start_time = time.time()
    now = datetime.now()

    logger.info("="*60)
    logger.info(f"PATCHTST INFERENCE - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)

    # Device selection
    device = torch.device('cpu')  # Ampere has no GPU
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model_load_start = time.time()
    model = load_model(args.model_path, device)
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.1f}s")

    # Get model ID from filename
    model_id = f"patchtst_{os.path.basename(os.path.dirname(args.model_path))}"

    # Connect to database
    conn = get_db_connection()
    logger.info("Connected to database")

    # Get tradeable items
    items_df = get_tradeable_items(conn, min_volume=args.min_volume)
    logger.info(f"Found {len(items_df)} tradeable items")

    # Process items in batches
    all_rows = []
    feature_time = 0
    inference_time = 0

    batch_recent = []
    batch_medium = []
    batch_long = []
    batch_item_ids = []
    batch_metadata = []

    now_ts = pd.Timestamp.now(tz='UTC')

    for idx, row in items_df.iterrows():
        item_id = row['item_id']
        item_name = row['name']

        # Load and compute features
        feat_start = time.time()
        df_5min = load_item_data(conn, item_id, lookback_days=31)

        if len(df_5min) < 500:
            continue

        features = compute_item_features(df_5min, now_ts)
        feature_time += time.time() - feat_start

        if features is None:
            continue

        batch_recent.append(features['recent'])
        batch_medium.append(features['medium'])
        batch_long.append(features['long'])
        batch_item_ids.append(item_id % model.config.n_items)
        batch_metadata.append({
            'item_id': item_id,
            'item_name': item_name,
            'current_mid': features['current_mid'],
            'current_high': features['current_high'],
            'current_low': features['current_low'],
        })

        # Process batch when full
        if len(batch_recent) >= args.batch_size:
            inf_start = time.time()

            recent_t = torch.from_numpy(np.array(batch_recent)).to(device)
            medium_t = torch.from_numpy(np.array(batch_medium)).to(device)
            long_t = torch.from_numpy(np.array(batch_long)).to(device)
            item_ids_t = torch.tensor(batch_item_ids).to(device)

            with torch.no_grad():
                outputs = model(recent_t, medium_t, long_t, item_ids_t)

            high_q = outputs['high_quantiles'].cpu().numpy()
            low_q = outputs['low_quantiles'].cpu().numpy()

            for i, meta in enumerate(batch_metadata):
                rows = convert_predictions_to_rows(
                    meta['item_id'], meta['item_name'],
                    meta['current_high'], meta['current_low'], meta['current_mid'],
                    high_q[i], low_q[i], now, model_id
                )
                all_rows.extend(rows)

            inference_time += time.time() - inf_start

            batch_recent = []
            batch_medium = []
            batch_long = []
            batch_item_ids = []
            batch_metadata = []

    # Process remaining batch
    if batch_recent:
        inf_start = time.time()

        recent_t = torch.from_numpy(np.array(batch_recent)).to(device)
        medium_t = torch.from_numpy(np.array(batch_medium)).to(device)
        long_t = torch.from_numpy(np.array(batch_long)).to(device)
        item_ids_t = torch.tensor(batch_item_ids).to(device)

        with torch.no_grad():
            outputs = model(recent_t, medium_t, long_t, item_ids_t)

        high_q = outputs['high_quantiles'].cpu().numpy()
        low_q = outputs['low_quantiles'].cpu().numpy()

        for i, meta in enumerate(batch_metadata):
            rows = convert_predictions_to_rows(
                meta['item_id'], meta['item_name'],
                meta['current_high'], meta['current_low'], meta['current_mid'],
                high_q[i], low_q[i], now, model_id
            )
            all_rows.extend(rows)

        inference_time += time.time() - inf_start

    # Assign confidence labels relative to batch distribution
    all_rows = assign_confidence_labels(all_rows)
    conf_counts = {}
    for r in all_rows:
        conf_counts[r['confidence']] = conf_counts.get(r['confidence'], 0) + 1
    logger.info(f"Confidence distribution: {conf_counts}")

    # Write predictions
    if not args.dry_run:
        write_start = time.time()
        written = write_predictions_copy(conn, all_rows, args.table)
        write_time = time.time() - write_start
        logger.info(f"Wrote {written} predictions in {write_time:.1f}s")
    else:
        logger.info(f"[DRY RUN] Would write {len(all_rows)} predictions")

    conn.close()

    total_time = time.time() - start_time

    logger.info("="*60)
    logger.info("COMPLETE")
    logger.info("="*60)
    logger.info(f"Items processed: {len(items_df)}")
    logger.info(f"Predictions generated: {len(all_rows)}")
    logger.info(f"Model load time: {model_load_time:.1f}s")
    logger.info(f"Feature time: {feature_time:.1f}s")
    logger.info(f"Inference time: {inference_time:.1f}s")
    logger.info(f"Total time: {total_time:.1f}s")

    return 0


def main():
    parser = argparse.ArgumentParser(description='PatchTST Production Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Inference batch size')
    parser.add_argument('--min-volume', type=int, default=500,
                       help='Minimum trading volume to include item')
    parser.add_argument('--table', type=str, default='predictions',
                       help='Target table (predictions or predictions_staging)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate predictions without writing to DB')
    args = parser.parse_args()

    try:
        return run_inference(args)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
