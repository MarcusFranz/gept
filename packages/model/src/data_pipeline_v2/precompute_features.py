"""
Precompute training features using all available CPU cores.

Uses pyarrow predicate pushdown to efficiently load only needed item data per worker.
This allows 200+ workers without OOM because each worker loads only its assigned items.

Usage:
    python precompute_features.py --data-dir /dev/shm/gept --output-dir /dev/shm/gept/precomputed --workers 200
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    recent_hours: int = 24      # 5-min resolution
    medium_days: int = 7        # 1-hour resolution
    long_days: int = 30         # 4-hour resolution

    @property
    def recent_len(self) -> int:
        return self.recent_hours * 12  # 288

    @property
    def medium_len(self) -> int:
        return self.medium_days * 24   # 168

    @property
    def long_len(self) -> int:
        return self.long_days * 6      # 180

    horizons: tuple = (1, 2, 4, 8, 12, 24, 48)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all parquet files."""
    logger.info(f"Loading data from {data_dir}")

    df_5min = pd.read_parquet(data_dir / "price_data_5min.parquet")
    df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'], utc=True)
    df_5min = df_5min.set_index(['item_id', 'timestamp']).sort_index()

    df_1h = pd.read_parquet(data_dir / "price_data_1h.parquet")
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)
    df_1h = df_1h.set_index(['item_id', 'timestamp']).sort_index()

    df_4h = pd.read_parquet(data_dir / "price_data_4h.parquet")
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], utc=True)
    df_4h = df_4h.set_index(['item_id', 'timestamp']).sort_index()

    logger.info(f"  5min: {len(df_5min):,} rows")
    logger.info(f"  1h: {len(df_1h):,} rows")
    logger.info(f"  4h: {len(df_4h):,} rows")

    return df_5min, df_1h, df_4h


def load_item_data_filtered(data_dir: Path, item_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parquet files filtered to specific items using predicate pushdown."""
    # Use pyarrow's filter to read only needed rows
    filters = [('item_id', 'in', item_ids)]

    df_5min = pq.read_table(data_dir / "price_data_5min.parquet", filters=filters).to_pandas()
    df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'], utc=True)
    df_5min = df_5min.set_index(['item_id', 'timestamp']).sort_index()

    df_1h = pq.read_table(data_dir / "price_data_1h.parquet", filters=filters).to_pandas()
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)
    df_1h = df_1h.set_index(['item_id', 'timestamp']).sort_index()

    df_4h = pq.read_table(data_dir / "price_data_4h.parquet", filters=filters).to_pandas()
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], utc=True)
    df_4h = df_4h.set_index(['item_id', 'timestamp']).sort_index()

    return df_5min, df_1h, df_4h


def build_sample_index(
    df_5min: pd.DataFrame,
    item_ids: list[int],
    start_date: datetime,
    end_date: datetime,
    config: DataConfig,
    sample_interval_hours: int = 1
) -> list[tuple[int, pd.Timestamp]]:
    """Build index of valid (item_id, timestamp) sample points."""
    samples = []

    min_date = pd.Timestamp(to_utc(start_date + timedelta(days=config.long_days)))
    max_date = pd.Timestamp(to_utc(end_date - timedelta(hours=max(config.horizons))))

    for item_id in item_ids:
        if item_id not in df_5min.index.get_level_values(0):
            continue

        item_data = df_5min.loc[item_id]
        timestamps = item_data.index

        valid_ts = timestamps[(timestamps >= min_date) & (timestamps < max_date)]

        sample_every = sample_interval_hours * 12
        for i, ts in enumerate(valid_ts):
            if i % sample_every == 0:
                samples.append((item_id, ts))

    return samples


def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros at start or truncate to target length."""
    if len(arr) >= target_len:
        return arr[-target_len:]
    else:
        padding = np.zeros((target_len - len(arr), arr.shape[1]))
        return np.vstack([padding, arr])


def prepare_5min_features(df: pd.DataFrame, target_len: int) -> np.ndarray:
    """Prepare 5-min features: high, low, high_vol, low_vol, spread, staleness"""
    if len(df) == 0:
        return np.zeros((target_len, 6), dtype=np.float32)

    df = df.copy()
    df["avg_high_price"] = df["avg_high_price"].ffill()
    df["avg_low_price"] = df["avg_low_price"].ffill()

    features = np.column_stack([
        df["avg_high_price"].fillna(0).values,
        df["avg_low_price"].fillna(0).values,
        df["high_price_volume"].fillna(0).values,
        df["low_price_volume"].fillna(0).values,
        (df["avg_high_price"] - df["avg_low_price"]).fillna(0).values,
        df["avg_high_price"].isna().cumsum().values
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def prepare_aggregated_features(df: pd.DataFrame, target_len: int) -> np.ndarray:
    """Prepare 1h/4h features from OHLC bars."""
    if len(df) == 0:
        return np.zeros((target_len, 10), dtype=np.float32)

    df = df.copy()

    high_close = df.get("high_close", pd.Series(0, index=df.index))
    low_close = df.get("low_close", pd.Series(0, index=df.index))
    high_vol = df.get("high_volume", pd.Series(0, index=df.index))
    low_vol = df.get("low_volume", pd.Series(0, index=df.index))
    sample_count = df.get("sample_count", pd.Series(12, index=df.index))
    high_high = df.get("high_high", high_close)
    high_low = df.get("high_low", high_close)
    low_high = df.get("low_high", low_close)
    low_low = df.get("low_low", low_close)

    high_close = high_close.ffill()
    low_close = low_close.ffill()

    features = np.column_stack([
        high_close.fillna(0).values,
        low_close.fillna(0).values,
        high_vol.fillna(0).values,
        low_vol.fillna(0).values,
        (high_close - low_close).fillna(0).values,
        sample_count.fillna(0).values,
        np.zeros(len(df)),
        (high_high - high_low).fillna(0).values,
        (low_high - low_low).fillna(0).values,
        (high_vol + low_vol).fillna(0).values
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def process_sample_sequential(
    item_id: int,
    timestamp: pd.Timestamp,
    df_5min: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    config: DataConfig,
    item_id_to_idx: dict
) -> Optional[dict]:
    """Process a single sample using passed-in dataframes."""
    try:
        # Recent 5-min sequence
        start = timestamp - pd.Timedelta(hours=config.recent_hours)
        item_data = df_5min.loc[item_id]
        df = item_data[(item_data.index > start) & (item_data.index <= timestamp)]
        recent = prepare_5min_features(df, config.recent_len)

        # Medium 1-hour sequence
        start = timestamp - pd.Timedelta(days=config.medium_days)
        hour_ts = timestamp.floor('h')
        item_data_1h = df_1h.loc[item_id]
        df = item_data_1h[(item_data_1h.index > start) & (item_data_1h.index <= hour_ts)]
        medium = prepare_aggregated_features(df, config.medium_len)

        # Long 4-hour sequence
        start = timestamp - pd.Timedelta(days=config.long_days)
        hour = timestamp.hour
        block_hour = hour - (hour % 4)
        block_ts = timestamp.replace(hour=block_hour, minute=0, second=0, microsecond=0)
        item_data_4h = df_4h.loc[item_id]
        df = item_data_4h[(item_data_4h.index > start) & (item_data_4h.index <= block_ts)]
        long_seq = prepare_aggregated_features(df, config.long_len)

        # Current prices
        try:
            row = df_5min.loc[(item_id, timestamp)]
            current_high = float(row["avg_high_price"]) if pd.notna(row["avg_high_price"]) else 0.0
            current_low = float(row["avg_low_price"]) if pd.notna(row["avg_low_price"]) else 0.0
        except (KeyError, TypeError):
            current_high, current_low = 0.0, 0.0

        # Targets
        targets_min_low = []
        targets_max_high = []
        for horizon in config.horizons:
            end_time = timestamp + pd.Timedelta(hours=horizon)
            window = item_data[(item_data.index > timestamp) & (item_data.index <= end_time)]

            if len(window) > 0:
                min_low = window["avg_low_price"].min()
                max_high = window["avg_high_price"].max()
                targets_min_low.append(min_low if pd.notna(min_low) else 0)
                targets_max_high.append(max_high if pd.notna(max_high) else 0)
            else:
                targets_min_low.append(0)
                targets_max_high.append(0)

        return {
            'item_idx': item_id_to_idx[item_id],
            'recent': recent,
            'medium': medium,
            'long': long_seq,
            'current_high': np.float32(current_high),
            'current_low': np.float32(current_low),
            'targets_min_low': np.array(targets_min_low, dtype=np.float32),
            'targets_max_high': np.array(targets_max_high, dtype=np.float32),
        }
    except Exception as e:
        return None


def process_item_batch(args: tuple) -> list[tuple[int, dict]]:
    """Process all samples for a batch of items.

    Each worker handles a subset of items and loads only that subset's data
    using pyarrow predicate pushdown - much more memory efficient.
    """
    (batch_item_ids, item_samples, data_dir, config_dict, item_id_to_idx) = args

    config = DataConfig(**config_dict)
    results = []

    # Load only data for this batch of items using predicate pushdown
    df_5min, df_1h, df_4h = load_item_data_filtered(Path(data_dir), batch_item_ids)

    # Process each sample for these items
    for global_idx, item_id, ts in item_samples:
        result = process_sample_sequential(
            item_id, ts, df_5min, df_1h, df_4h, config, item_id_to_idx
        )
        if result is not None:
            results.append((global_idx, result))

    return results


def precompute_dataset(
    data_dir: Path,
    output_dir: Path,
    train_end: str,
    val_end: str,
    n_workers: int = 200,
    sample_interval_hours: int = 1
):
    """Precompute all training and validation samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data to get metadata
    df_5min, df_1h, df_4h = load_data(data_dir)

    # Load item list
    items_df = pd.read_parquet(data_dir / "training_items.parquet")
    item_ids = items_df['item_id'].tolist()
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    logger.info(f"Processing {len(item_ids)} items with {n_workers} workers")

    # Get date range
    data_start = df_5min.index.get_level_values('timestamp').min()
    train_end_dt = datetime.fromisoformat(train_end).replace(tzinfo=timezone.utc)
    val_end_dt = datetime.fromisoformat(val_end).replace(tzinfo=timezone.utc)

    config = DataConfig()
    config_dict = {
        'recent_hours': config.recent_hours,
        'medium_days': config.medium_days,
        'long_days': config.long_days,
    }

    # Clear loaded data to free memory for workers
    del df_5min, df_1h, df_4h
    import gc
    gc.collect()

    # Process train and val sets
    for split_name, start_date, end_date in [
        ('train', data_start, train_end_dt),
        ('val', train_end_dt - timedelta(days=config.long_days), val_end_dt)
    ]:
        logger.info(f"\n=== Processing {split_name} split ===")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Reload data for sample index building
        df_5min_full, _, _ = load_data(data_dir)

        # Build sample index
        samples = build_sample_index(df_5min_full, item_ids, start_date, end_date, config, sample_interval_hours)
        n_samples = len(samples)
        logger.info(f"Found {n_samples:,} samples")

        # Clear full data again
        del df_5min_full
        gc.collect()

        if n_samples == 0:
            logger.warning(f"No samples for {split_name}, skipping")
            continue

        # Prepare memory-mapped arrays
        recent_shape = (n_samples, config.recent_len, 6)
        medium_shape = (n_samples, config.medium_len, 10)
        long_shape = (n_samples, config.long_len, 10)

        recent_mmap = np.memmap(output_dir / f"{split_name}_recent.npy", dtype=np.float32, mode='w+', shape=recent_shape)
        medium_mmap = np.memmap(output_dir / f"{split_name}_medium.npy", dtype=np.float32, mode='w+', shape=medium_shape)
        long_mmap = np.memmap(output_dir / f"{split_name}_long.npy", dtype=np.float32, mode='w+', shape=long_shape)
        item_idx_arr = np.zeros(n_samples, dtype=np.int32)
        current_high_arr = np.zeros(n_samples, dtype=np.float32)
        current_low_arr = np.zeros(n_samples, dtype=np.float32)
        targets_min_low_arr = np.zeros((n_samples, len(config.horizons)), dtype=np.float32)
        targets_max_high_arr = np.zeros((n_samples, len(config.horizons)), dtype=np.float32)

        # Group samples by item_id
        from collections import defaultdict
        samples_by_item = defaultdict(list)
        for global_idx, (item_id, ts) in enumerate(samples):
            samples_by_item[item_id].append((global_idx, item_id, ts))

        # Create batches with ~2-3 items per batch for 200 workers
        items_per_batch = max(1, len(item_ids) // n_workers)
        batches = []
        current_batch_items = []
        current_batch_samples = []

        for item_id in item_ids:
            current_batch_items.append(item_id)
            current_batch_samples.extend(samples_by_item[item_id])

            if len(current_batch_items) >= items_per_batch:
                batches.append((
                    current_batch_items.copy(),
                    current_batch_samples.copy(),
                    str(data_dir),
                    config_dict,
                    item_id_to_idx
                ))
                current_batch_items = []
                current_batch_samples = []

        # Don't forget the last batch
        if current_batch_items:
            batches.append((
                current_batch_items,
                current_batch_samples,
                str(data_dir),
                config_dict,
                item_id_to_idx
            ))

        logger.info(f"Split into {len(batches)} batches for {n_workers} workers")
        avg_samples_per_batch = n_samples / len(batches)
        logger.info(f"~{avg_samples_per_batch:.0f} samples per batch, ~{items_per_batch} items per batch")

        completed = 0
        failed = 0
        batches_done = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_item_batch, batch): batch for batch in batches}

            for future in as_completed(futures):
                try:
                    results = future.result()
                    for global_idx, result in results:
                        recent_mmap[global_idx] = result['recent']
                        medium_mmap[global_idx] = result['medium']
                        long_mmap[global_idx] = result['long']
                        item_idx_arr[global_idx] = result['item_idx']
                        current_high_arr[global_idx] = result['current_high']
                        current_low_arr[global_idx] = result['current_low']
                        targets_min_low_arr[global_idx] = result['targets_min_low']
                        targets_max_high_arr[global_idx] = result['targets_max_high']
                        completed += 1
                    batches_done += 1
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
                    failed += len(futures[future][1])
                    batches_done += 1

                # Log progress every 10 batches
                if batches_done % 10 == 0 or batches_done == len(batches):
                    logger.info(f"  Batches: {batches_done}/{len(batches)} | Samples: {completed:,}/{n_samples:,} ({100*completed/n_samples:.1f}%)")

        # Flush memory maps
        recent_mmap.flush()
        medium_mmap.flush()
        long_mmap.flush()
        del recent_mmap, medium_mmap, long_mmap

        # Save other arrays
        np.save(output_dir / f"{split_name}_item_idx.npy", item_idx_arr)
        np.save(output_dir / f"{split_name}_current_high.npy", current_high_arr)
        np.save(output_dir / f"{split_name}_current_low.npy", current_low_arr)
        np.save(output_dir / f"{split_name}_targets_min_low.npy", targets_min_low_arr)
        np.save(output_dir / f"{split_name}_targets_max_high.npy", targets_max_high_arr)

        # Save metadata as JSON
        metadata = {
            'n_samples': n_samples,
            'n_items': len(item_ids),
            'item_ids': item_ids,
            'recent_shape': list(recent_shape),
            'medium_shape': list(medium_shape),
            'long_shape': list(long_shape),
            'horizons': list(config.horizons),
            'completed': completed,
            'failed': failed,
        }
        with open(output_dir / f"{split_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Completed {split_name}: {completed:,} success, {failed:,} failed")

    logger.info("\n=== Precomputation complete ===")


def main():
    parser = argparse.ArgumentParser(description='Precompute training features')
    parser.add_argument('--data-dir', type=str, default='/workspace/gept', help='Directory with parquet files')
    parser.add_argument('--output-dir', type=str, default='/workspace/gept/precomputed', help='Output directory')
    parser.add_argument('--workers', type=int, default=200, help='Number of worker processes')
    parser.add_argument('--train-end', type=str, default='2026-01-12', help='End of training period')
    parser.add_argument('--val-end', type=str, default='2026-01-20', help='End of validation period')
    parser.add_argument('--sample-interval', type=int, default=1, help='Sample interval in hours')

    args = parser.parse_args()

    precompute_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        train_end=args.train_end,
        val_end=args.val_end,
        n_workers=args.workers,
        sample_interval_hours=args.sample_interval
    )


if __name__ == '__main__':
    main()
