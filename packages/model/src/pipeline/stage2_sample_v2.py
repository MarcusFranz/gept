"""
Stage 2 v2: Time-Based Sample Index with Horizon Buffers
=========================================================

Generate sample indices with strict time-based splits.
Samples are only emitted if t + max_horizon_bars stays within the split.

Works with pre-split data from prepare_training_splits.py:
    data/splits/
        train/5min/item_*.parquet
        val/5min/item_*.parquet
        test/5min/item_*.parquet

Usage:
    python -m src.pipeline.stage2_sample_v2 \
        --input data/splits \
        --output data/samples \
        --max-horizon 48
"""

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.pipeline.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SPLIT BOUNDARIES (must match prepare_training_splits.py)
# =============================================================================

SPLIT_BOUNDARIES = {
    'train': {
        'start': datetime(2023, 11, 1, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2025, 8, 31, 23, 55, tzinfo=timezone.utc),
    },
    'val': {
        'start': datetime(2025, 9, 3, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2025, 11, 30, 23, 55, tzinfo=timezone.utc),
    },
    'test': {
        'start': datetime(2025, 12, 3, 0, 0, tzinfo=timezone.utc),
        'end': datetime(2026, 1, 31, 23, 55, tzinfo=timezone.utc),
    },
}


def load_item_data(parquet_path: Path) -> pd.DataFrame:
    """Load item data from parquet file."""
    if not parquet_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.sort_values('timestamp')


def find_valid_samples_for_split(
    df: pd.DataFrame,
    split_name: str,
    min_history_hours: int,
    max_horizon_hours: int,
    sample_interval_minutes: int = 5,
) -> pd.DatetimeIndex:
    """
    Find valid sample timestamps within a split.

    A sample at time t is valid if:
    1. t - min_history is within available data
    2. t + max_horizon is within the split boundary

    This ensures targets can be computed without leaking into future splits.
    """
    if df.empty:
        return pd.DatetimeIndex([])

    boundaries = SPLIT_BOUNDARIES[split_name]
    split_end = pd.Timestamp(boundaries['end'])

    timestamps = pd.DatetimeIndex(df['timestamp'])

    # Must have enough history
    min_ts = timestamps.min() + timedelta(hours=min_history_hours)

    # Must have room for max horizon within split
    max_ts = split_end - timedelta(hours=max_horizon_hours)

    # Filter to valid range
    valid = timestamps[(timestamps >= min_ts) & (timestamps <= max_ts)]

    # Subsample if needed
    if sample_interval_minutes > 5:
        step = sample_interval_minutes // 5
        valid = valid[::step]

    return valid


def get_item_ids_from_split(split_dir: Path) -> List[int]:
    """Get list of item IDs from parquet files in a split directory."""
    five_min_dir = split_dir / '5min'
    if not five_min_dir.exists():
        return []

    items = []
    for pq_file in five_min_dir.glob('item_*.parquet'):
        item_id = int(pq_file.stem.replace('item_', ''))
        items.append(item_id)

    return sorted(items)


def build_sample_index(
    splits_dir: Path,
    min_history_hours: int = 720,  # 30 days
    max_horizon_hours: int = 48,
    sample_interval_minutes: int = 5,
    items_file: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build complete sample index for all splits.

    Args:
        splits_dir: Directory containing train/val/test subdirs
        min_history_hours: Minimum history required (default 30 days = 720h)
        max_horizon_hours: Maximum prediction horizon (default 48h)
        sample_interval_minutes: Sample frequency (default 5min)
        items_file: Optional file listing items to include

    Returns:
        DataFrame with columns: item_id, timestamp, split
    """
    # Get items to process
    if items_file and items_file.exists():
        with open(items_file) as f:
            item_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        logger.info(f"Using {len(item_ids)} items from {items_file}")
    else:
        # Get union of items across splits
        all_items = set()
        for split_name in ['train', 'val', 'test']:
            split_items = get_item_ids_from_split(splits_dir / split_name)
            all_items.update(split_items)
        item_ids = sorted(all_items)
        logger.info(f"Found {len(item_ids)} items across splits")

    all_samples = []

    for split_name in ['train', 'val', 'test']:
        split_dir = splits_dir / split_name / '5min'
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        logger.info(f"Processing {split_name} split...")
        split_samples = []

        for i, item_id in enumerate(item_ids):
            parquet_path = split_dir / f'item_{item_id}.parquet'
            df = load_item_data(parquet_path)

            if df.empty:
                continue

            valid_ts = find_valid_samples_for_split(
                df,
                split_name,
                min_history_hours=min_history_hours,
                max_horizon_hours=max_horizon_hours,
                sample_interval_minutes=sample_interval_minutes,
            )

            if len(valid_ts) > 0:
                item_df = pd.DataFrame({
                    'item_id': item_id,
                    'timestamp': valid_ts,
                    'split': split_name,
                })
                split_samples.append(item_df)

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i+1}/{len(item_ids)} items...")

        if split_samples:
            split_df = pd.concat(split_samples, ignore_index=True)
            all_samples.append(split_df)
            logger.info(f"  {split_name}: {len(split_df):,} samples")

    if not all_samples:
        return pd.DataFrame(columns=['item_id', 'timestamp', 'split'])

    result = pd.concat(all_samples, ignore_index=True)
    result = result.sort_values(['split', 'item_id', 'timestamp']).reset_index(drop=True)

    return result


def generate_summary(samples_df: pd.DataFrame) -> dict:
    """Generate summary statistics for the sample index."""
    if len(samples_df) == 0:
        return {"error": "No samples generated"}

    summary = {
        "total_samples": len(samples_df),
        "items": int(samples_df['item_id'].nunique()),
        "splits": {},
    }

    for split_name in ['train', 'val', 'test']:
        split_df = samples_df[samples_df['split'] == split_name]
        if len(split_df) > 0:
            summary['splits'][split_name] = {
                "samples": len(split_df),
                "items": int(split_df['item_id'].nunique()),
                "date_range": [
                    split_df['timestamp'].min().isoformat(),
                    split_df['timestamp'].max().isoformat(),
                ],
                "samples_per_item": {
                    "mean": float(split_df.groupby('item_id').size().mean()),
                    "min": int(split_df.groupby('item_id').size().min()),
                    "max": int(split_df.groupby('item_id').size().max()),
                },
            }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2 v2: Time-based sample index")
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with train/val/test splits')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for samples')
    parser.add_argument('--items', type=str,
                        help='Optional file listing item IDs to include')
    parser.add_argument('--min-history', type=int, default=720,
                        help='Minimum history hours (default: 720 = 30 days)')
    parser.add_argument('--max-horizon', type=int, default=48,
                        help='Maximum horizon hours (default: 48)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Sample interval in minutes (default: 5)')
    args = parser.parse_args()

    splits_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    items_file = Path(args.items) if args.items else None

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 2 v2: TIME-BASED SAMPLE INDEX                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Input:        {str(splits_dir):<48} ║
║  Min history:  {args.min_history}h ({args.min_history // 24}d)                                        ║
║  Max horizon:  {args.max_horizon}h (samples must fit within split)             ║
║  Interval:     {args.interval}min                                            ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Build sample index
    samples_df = build_sample_index(
        splits_dir,
        min_history_hours=args.min_history,
        max_horizon_hours=args.max_horizon,
        sample_interval_minutes=args.interval,
        items_file=items_file,
    )

    if len(samples_df) == 0:
        logger.error("No valid samples generated!")
        return 1

    # Save samples
    samples_path = output_dir / 'samples.csv'
    samples_df.to_csv(samples_path, index=False)
    logger.info(f"Saved {len(samples_df):,} samples to {samples_path}")

    # Generate and save summary
    summary = generate_summary(samples_df)
    summary['config'] = {
        'min_history_hours': args.min_history,
        'max_horizon_hours': args.max_horizon,
        'sample_interval_minutes': args.interval,
        'split_boundaries': {
            k: {'start': v['start'].isoformat(), 'end': v['end'].isoformat()}
            for k, v in SPLIT_BOUNDARIES.items()
        },
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  SAMPLING COMPLETE                                               ║
╠══════════════════════════════════════════════════════════════════╣
║  Total samples: {summary['total_samples']:>10,}                                    ║
║  Unique items:  {summary['items']:>10,}                                    ║
╠══════════════════════════════════════════════════════════════════╣""")

    for split_name in ['train', 'val', 'test']:
        if split_name in summary['splits']:
            s = summary['splits'][split_name]
            print(f"║  {split_name.upper():5}: {s['samples']:>10,} samples, {s['items']:>4} items                   ║")

    print(f"""╠══════════════════════════════════════════════════════════════════╣
║  Output: {str(output_dir):<54} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    return 0


if __name__ == '__main__':
    exit(main())
