"""
Stage 2: Sample & Index
=======================

Generate a manifest of valid (item_id, timestamp) training samples.
Not every timestamp works - we need enough history for lookback features
and enough future data for targets.

Usage:
    python -m src.pipeline.stage2_sample --input data/cleaned/ --output data/samples/
    python -m src.pipeline.stage2_sample --input data/cleaned/ --output data/samples/ --interval 1hour
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

from src.pipeline.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_valid_items(cleaned_dir: Path) -> List[int]:
    """
    Get list of item IDs that passed Stage 1 validation.

    Reads the validation report if available, otherwise lists parquet files.
    """
    validation_path = cleaned_dir / "validation_report.json"

    if validation_path.exists():
        with open(validation_path) as f:
            report = json.load(f)
        # Get items that passed validation
        valid_items = [
            int(item_id) for item_id, data in report.get("items", {}).items()
            if data.get("passed", False)
        ]
        logger.info(f"Found {len(valid_items)} valid items from validation report")
        return valid_items

    # Fallback: list parquet files
    five_min_dir = cleaned_dir / "5min"
    if not five_min_dir.exists():
        raise FileNotFoundError(f"5min directory not found: {five_min_dir}")

    items = []
    for pq_file in five_min_dir.glob("item_*.parquet"):
        item_id = int(pq_file.stem.replace("item_", ""))
        items.append(item_id)

    logger.info(f"Found {len(items)} items from parquet files (no validation report)")
    return sorted(items)


def load_item_timestamps(cleaned_dir: Path, item_id: int) -> pd.DatetimeIndex:
    """Load all timestamps for an item from its 5min parquet file."""
    path = cleaned_dir / "5min" / f"item_{item_id}.parquet"
    if not path.exists():
        return pd.DatetimeIndex([])

    df = pd.read_parquet(path, columns=['timestamp', 'gap'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Exclude gap rows from valid samples
    df = df[~df['gap']]

    return pd.DatetimeIndex(df['timestamp'])


def find_valid_samples(
    timestamps: pd.DatetimeIndex,
    min_history_days: int = 30,
    max_horizon_hours: int = 48,
    sample_interval: str = "5min"
) -> pd.DatetimeIndex:
    """
    Find timestamps that have sufficient history and future data.

    Args:
        timestamps: All available timestamps for an item
        min_history_days: Minimum days of history required (for long encoder)
        max_horizon_hours: Maximum prediction horizon (for targets)
        sample_interval: How often to sample ("5min", "30min", "1hour")

    Returns:
        DatetimeIndex of valid sample timestamps
    """
    if len(timestamps) == 0:
        return pd.DatetimeIndex([])

    timestamps = timestamps.sort_values()

    # Calculate bounds
    min_date = timestamps.min() + timedelta(days=min_history_days)
    max_date = timestamps.max() - timedelta(hours=max_horizon_hours)

    if min_date >= max_date:
        return pd.DatetimeIndex([])

    # Filter to valid range
    valid = timestamps[(timestamps >= min_date) & (timestamps <= max_date)]

    # Subsample based on interval
    if sample_interval == "5min":
        return valid
    elif sample_interval == "30min":
        # Keep every 6th sample (30min / 5min = 6)
        return valid[::6]
    elif sample_interval == "1hour":
        # Keep every 12th sample (60min / 5min = 12)
        return valid[::12]
    else:
        logger.warning(f"Unknown interval {sample_interval}, using 5min")
        return valid


def build_sample_index(
    cleaned_dir: Path,
    item_ids: List[int],
    val_cutoff: datetime,
    sample_interval: str = "5min",
    config: Optional[DataConfig] = None
) -> pd.DataFrame:
    """
    Build complete sample index for all items.

    Args:
        cleaned_dir: Directory containing cleaned parquet files
        item_ids: List of valid item IDs
        val_cutoff: Datetime separating train/val splits
        sample_interval: Sample frequency ("5min", "30min", "1hour")
        config: DataConfig with history/horizon settings

    Returns:
        DataFrame with columns: item_id, timestamp, split
    """
    if config is None:
        config = DataConfig()

    all_samples = []

    for i, item_id in enumerate(item_ids):
        timestamps = load_item_timestamps(cleaned_dir, item_id)

        valid_ts = find_valid_samples(
            timestamps,
            min_history_days=config.min_history_days,
            max_horizon_hours=max(config.horizons),
            sample_interval=sample_interval
        )

        if len(valid_ts) == 0:
            logger.warning(f"Item {item_id}: no valid samples")
            continue

        # Create DataFrame for this item
        item_df = pd.DataFrame({
            'item_id': item_id,
            'timestamp': valid_ts
        })

        # Assign train/val split based on cutoff
        val_cutoff_ts = pd.Timestamp(val_cutoff)
        item_df['split'] = 'train'
        item_df.loc[item_df['timestamp'] >= val_cutoff_ts, 'split'] = 'val'

        all_samples.append(item_df)

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i+1}/{len(item_ids)} items...")

    if not all_samples:
        return pd.DataFrame(columns=['item_id', 'timestamp', 'split'])

    result = pd.concat(all_samples, ignore_index=True)

    # Sort by item_id, then timestamp
    result = result.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

    return result


def generate_summary(samples_df: pd.DataFrame, config: DataConfig) -> dict:
    """Generate summary statistics for the sample index."""
    if len(samples_df) == 0:
        return {"error": "No samples generated"}

    train_df = samples_df[samples_df['split'] == 'train']
    val_df = samples_df[samples_df['split'] == 'val']

    return {
        "total_samples": len(samples_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_pct": len(train_df) / len(samples_df) if len(samples_df) > 0 else 0,
        "items": samples_df['item_id'].nunique(),
        "date_range": [
            samples_df['timestamp'].min().isoformat(),
            samples_df['timestamp'].max().isoformat()
        ],
        "samples_per_item": {
            "mean": samples_df.groupby('item_id').size().mean(),
            "min": samples_df.groupby('item_id').size().min(),
            "max": samples_df.groupby('item_id').size().max()
        },
        "min_history_required": f"{config.min_history_days} days",
        "max_horizon": f"{max(config.horizons)} hours"
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Generate sample index")
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory (Stage 1 output)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for samples')
    parser.add_argument('--val-cutoff', type=str, default='2024-12-01',
                        help='Validation split cutoff date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='5min',
                        choices=['5min', '30min', '1hour'],
                        help='Sample interval (default: 5min)')
    parser.add_argument('--min-history', type=int, default=30,
                        help='Minimum history days required (default: 30)')
    args = parser.parse_args()

    cleaned_dir = Path(args.input)
    output_dir = Path(args.output)

    if not cleaned_dir.exists():
        logger.error(f"Input directory not found: {cleaned_dir}")
        return 1

    # Parse validation cutoff
    val_cutoff = datetime.strptime(args.val_cutoff, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    # Create config
    config = DataConfig(long_days=args.min_history)

    logger.info(f"Input: {cleaned_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Val cutoff: {val_cutoff.date()}")
    logger.info(f"Sample interval: {args.interval}")
    logger.info(f"Min history: {config.min_history_days} days")

    # Get valid items
    item_ids = get_valid_items(cleaned_dir)

    if not item_ids:
        logger.error("No valid items found")
        return 1

    # Build sample index
    logger.info(f"Building sample index for {len(item_ids)} items...")
    samples_df = build_sample_index(
        cleaned_dir,
        item_ids,
        val_cutoff,
        sample_interval=args.interval,
        config=config
    )

    if len(samples_df) == 0:
        logger.error("No valid samples generated")
        return 1

    # Generate summary
    summary = generate_summary(samples_df, config)
    summary['val_cutoff'] = args.val_cutoff
    summary['sample_interval'] = args.interval

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.csv"
    samples_df.to_csv(samples_path, index=False)
    logger.info(f"Wrote {len(samples_df):,} samples to {samples_path}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Wrote summary to {summary_path}")

    # Print summary
    logger.info(f"Total samples: {summary['total_samples']:,}")
    logger.info(f"  Train: {summary['train_samples']:,} ({summary['train_pct']:.1%})")
    logger.info(f"  Val: {summary['val_samples']:,}")
    logger.info(f"Items: {summary['items']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
