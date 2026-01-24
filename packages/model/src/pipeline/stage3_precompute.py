"""
Stage 3: Precompute Features
============================

Turn the sample index into ready-to-train feature arrays.
Processes items sequentially and writes output in chunks to control memory.

Usage:
    python -m src.pipeline.stage3_precompute --samples data/samples/samples.csv --parquet data/cleaned/ --output data/features/
    python -m src.pipeline.stage3_precompute --samples data/samples/samples.csv --parquet data/cleaned/ --output data/features/ --resume
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.config import DataConfig
from src.pipeline.features import compute_sample_features, compute_volume_targets
from src.pipeline.validation import validate_stage3_output
from src.pipeline.volume_stats import compute_volume_stats, normalize_volume, save_volume_stats, load_volume_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkWriter:
    """
    Manages writing precomputed features to chunked .npz files.

    Accumulates samples in memory until chunk_size is reached,
    then writes to disk and clears the buffer.
    """

    def __init__(self, output_dir: Path, split: str, chunk_size: int = 10000):
        """
        Args:
            output_dir: Base output directory (features/)
            split: 'train' or 'val'
            chunk_size: Samples per chunk file
        """
        self.output_dir = output_dir / split
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.chunk_size = chunk_size

        self.chunk_idx = 0
        self.buffer: Dict[str, List] = {
            'recent': [],
            'medium': [],
            'long': [],
            'item_ids': [],
            'targets': [],
            'volume_targets': [],
        }
        self.total_written = 0

    def add_sample(self, item_id: int, features: Dict[str, np.ndarray], volume_targets: Optional[np.ndarray] = None) -> None:
        """Add a single sample to the buffer."""
        self.buffer['recent'].append(features['recent'])
        self.buffer['medium'].append(features['medium'])
        self.buffer['long'].append(features['long'])
        self.buffer['item_ids'].append(item_id)
        self.buffer['targets'].append(features['targets'])
        if volume_targets is not None:
            self.buffer['volume_targets'].append(volume_targets)

        if len(self.buffer['item_ids']) >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        """Write current buffer to disk as a chunk."""
        if len(self.buffer['item_ids']) == 0:
            return

        chunk_path = self.output_dir / f"chunk_{self.chunk_idx:04d}.npz"

        save_dict = {
            'recent': np.array(self.buffer['recent'], dtype=np.float32),
            'medium': np.array(self.buffer['medium'], dtype=np.float32),
            'long': np.array(self.buffer['long'], dtype=np.float32),
            'item_ids': np.array(self.buffer['item_ids'], dtype=np.int32),
            'targets': np.array(self.buffer['targets'], dtype=np.float32),
        }
        # Include volume targets if present
        if self.buffer['volume_targets']:
            save_dict['volume_targets'] = np.array(self.buffer['volume_targets'], dtype=np.float32)

        np.savez_compressed(chunk_path, **save_dict)

        samples_in_chunk = len(self.buffer['item_ids'])
        self.total_written += samples_in_chunk
        logger.info(f"Wrote {self.split} chunk {self.chunk_idx}: {samples_in_chunk} samples")

        # Clear buffer
        self.buffer = {k: [] for k in self.buffer}
        self.chunk_idx += 1

    def finalize(self) -> int:
        """Flush any remaining samples and return total count."""
        self._flush()
        return self.total_written


def collect_volume_stats_pass1(
    samples_df: pd.DataFrame,
    parquet_dir: Path,
    config: DataConfig,
) -> Dict:
    """
    Pass 1: Collect raw volumes from training data to compute per-item statistics.

    Only processes training samples to avoid data leakage from validation set.

    Returns:
        Dict of volume statistics per item
    """
    logger.info("Pass 1: Collecting volume statistics from training data...")

    # Filter to training samples only
    train_samples = samples_df[samples_df['split'] == 'train']
    item_ids = sorted(train_samples['item_id'].unique())

    volumes_by_item: Dict[int, List[float]] = {}

    for i, item_id in enumerate(item_ids):
        item_samples = train_samples[train_samples['item_id'] == item_id]

        try:
            df_5min = pd.read_parquet(parquet_dir / "5min" / f"item_{item_id}.parquet")
            df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'], utc=True)
        except FileNotFoundError:
            continue

        # Collect volumes at sample timestamps
        item_volumes = []
        for _, row in item_samples.iterrows():
            timestamp = pd.Timestamp(row['timestamp'])
            # Get the row at this timestamp
            mask = df_5min['timestamp'] == timestamp
            if mask.sum() > 0:
                row_data = df_5min[mask].iloc[0]
                # Sum buy + sell volume
                total_vol = row_data.get('high_price_volume', 0) + row_data.get('low_price_volume', 0)
                if total_vol > 0:
                    item_volumes.append(float(total_vol))

        if item_volumes:
            volumes_by_item[item_id] = np.array(item_volumes)

        if (i + 1) % 50 == 0:
            logger.info(f"Pass 1 progress: {i+1}/{len(item_ids)} items")

    # Compute statistics
    stats = compute_volume_stats(volumes_by_item, min_samples=100)
    logger.info(f"Pass 1 complete: {stats['_metadata']['total_items']} items, "
                f"{stats['_metadata']['fallback_items']} using fallback stats")

    return stats


def load_item_data(parquet_dir: Path, item_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all resolution data for a single item.

    Returns:
        Tuple of (df_5min, df_1h, df_4h) DataFrames
    """
    df_5min = pd.read_parquet(parquet_dir / "5min" / f"item_{item_id}.parquet")
    df_5min['timestamp'] = pd.to_datetime(df_5min['timestamp'], utc=True)

    df_1h = pd.read_parquet(parquet_dir / "1hour" / f"item_{item_id}.parquet")
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)

    df_4h = pd.read_parquet(parquet_dir / "4hour" / f"item_{item_id}.parquet")
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], utc=True)

    return df_5min, df_1h, df_4h


def save_progress(output_dir: Path, last_item_id: int, samples_processed: int) -> None:
    """Save checkpoint for resume capability."""
    progress = {
        'last_completed_item': int(last_item_id),  # Convert numpy int64 to Python int
        'samples_processed': int(samples_processed),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    with open(output_dir / 'progress.json', 'w') as f:
        json.dump(progress, f, indent=2)


def load_progress(output_dir: Path) -> Optional[Dict]:
    """Load checkpoint if exists."""
    progress_path = output_dir / 'progress.json'
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return None


def process_items(
    samples_df: pd.DataFrame,
    parquet_dir: Path,
    output_dir: Path,
    config: DataConfig,
    chunk_size: int = 10000,
    resume_from: Optional[int] = None,
    volume_stats: Optional[Dict] = None,
    enable_volume: bool = True
) -> Tuple[int, int]:
    """
    Process all items sequentially, writing features to chunks.

    Args:
        samples_df: Sample index with item_id, timestamp, split columns
        parquet_dir: Directory with cleaned parquet files
        output_dir: Output directory for feature chunks
        config: DataConfig with feature settings
        chunk_size: Samples per chunk file
        resume_from: Item ID to resume from (skip items before this)
        volume_stats: Pre-computed volume statistics (from Pass 1)
        enable_volume: Whether to compute volume targets

    Returns:
        Tuple of (train_count, val_count)
    """
    # Create chunk writers
    train_writer = ChunkWriter(output_dir, 'train', chunk_size)
    val_writer = ChunkWriter(output_dir, 'val', chunk_size)

    # Get unique items sorted
    item_ids = sorted(samples_df['item_id'].unique())
    logger.info(f"Processing {len(item_ids)} items...")

    # Skip items if resuming
    if resume_from is not None:
        item_ids = [i for i in item_ids if i > resume_from]
        logger.info(f"Resuming from item {resume_from}, {len(item_ids)} items remaining")

    # Build item_id to sequential index mapping
    all_item_ids = sorted(samples_df['item_id'].unique())
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(all_item_ids)}

    samples_processed = 0
    failed_samples = 0

    # Precompute bars needed for max horizon (48h * 12 bars/hr = 576)
    max_horizon_bars = max(config.horizons) * 12

    for i, item_id in enumerate(item_ids):
        # Get samples for this item
        item_samples = samples_df[samples_df['item_id'] == item_id]

        # Load data once for all samples of this item
        try:
            df_5min, df_1h, df_4h = load_item_data(parquet_dir, item_id)
        except FileNotFoundError as e:
            logger.warning(f"Skipping item {item_id}: {e}")
            continue

        # Prepare volume arrays if needed
        vol_data = None
        if enable_volume:
            vol_data = {
                'high_price_volume': df_5min['high_price_volume'].fillna(0).values.astype(np.float32),
                'low_price_volume': df_5min['low_price_volume'].fillna(0).values.astype(np.float32),
            }
            # Create timestamp-to-index mapping for fast lookup
            ts_to_idx = {ts: idx for idx, ts in enumerate(df_5min['timestamp'])}

        # Process each sample
        for _, row in item_samples.iterrows():
            timestamp = pd.Timestamp(row['timestamp'])
            split = row['split']

            features = compute_sample_features(
                df_5min, df_1h, df_4h, timestamp, config
            )

            if features is None:
                failed_samples += 1
                continue

            # Compute volume targets if enabled
            volume_targets = None
            if enable_volume and vol_data is not None and volume_stats is not None:
                current_idx = ts_to_idx.get(timestamp)
                if current_idx is not None and current_idx + max_horizon_bars < len(vol_data['high_price_volume']):
                    try:
                        # Compute raw volume targets (sums over horizons)
                        raw_vol_targets = compute_volume_targets(
                            vol_data, current_idx, config.horizons, bars_per_hour=12
                        )
                        # Normalize using per-item stats
                        # Shape: (n_horizons, 2) -> normalize each value
                        volume_targets = np.zeros_like(raw_vol_targets)
                        for h_idx in range(raw_vol_targets.shape[0]):
                            for side in range(2):  # buy, sell
                                raw_val = raw_vol_targets[h_idx, side]
                                volume_targets[h_idx, side] = normalize_volume(
                                    np.array([raw_val]), item_id, volume_stats
                                )[0]
                    except (AssertionError, IndexError):
                        # Not enough future data for this sample
                        volume_targets = None

            # Get sequential item index
            item_idx = item_id_to_idx[item_id]

            # Add to appropriate writer
            if split == 'train':
                train_writer.add_sample(item_idx, features, volume_targets)
            else:
                val_writer.add_sample(item_idx, features, volume_targets)

            samples_processed += 1

        # Save progress checkpoint every 10 items
        if (i + 1) % 10 == 0:
            save_progress(output_dir, item_id, samples_processed)
            logger.info(f"Progress: {i+1}/{len(item_ids)} items, "
                        f"{samples_processed} samples processed, {failed_samples} failed")

    # Finalize writers
    train_count = train_writer.finalize()
    val_count = val_writer.finalize()

    # Final progress save
    save_progress(output_dir, item_ids[-1] if item_ids else 0, samples_processed)

    logger.info(f"Completed: {samples_processed} samples ({failed_samples} failed)")
    logger.info(f"Train: {train_count}, Val: {val_count}")

    return train_count, val_count


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Precompute features")
    parser.add_argument('--samples', type=str, help='Path to samples.csv from Stage 2')
    parser.add_argument('--parquet', type=str, help='Path to cleaned parquet directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for features')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Samples per chunk (default: 10000)')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing output')
    parser.add_argument('--validate-samples', type=int, default=10,
                        help='Number of chunks to spot-check (default: 10)')
    parser.add_argument('--enable-volume', action='store_true', default=True,
                        help='Enable volume target computation (default: True)')
    parser.add_argument('--no-volume', action='store_true',
                        help='Disable volume target computation')
    args = parser.parse_args()

    # Handle volume flag
    enable_volume = args.enable_volume and not args.no_volume

    output_dir = Path(args.output)

    # Validate-only mode
    if args.validate_only:
        logger.info(f"Validating Stage 3 output at {output_dir}...")
        try:
            report = validate_stage3_output(output_dir, n_samples=args.validate_samples)
            report.to_json(output_dir / "validation_report.json")
            logger.info(f"Validation complete: {report.passed} passed, {report.failed} failed")
            return 0 if report.failed == 0 else 1
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 1

    # Precompute mode
    if not args.samples or not args.parquet:
        logger.error("--samples and --parquet are required for precompute mode")
        return 1

    samples_path = Path(args.samples)
    parquet_dir = Path(args.parquet)

    if not samples_path.exists():
        logger.error(f"Samples file not found: {samples_path}")
        return 1

    if not parquet_dir.exists():
        logger.error(f"Parquet directory not found: {parquet_dir}")
        return 1

    # Load samples
    logger.info(f"Loading samples from {samples_path}...")
    samples_df = pd.read_csv(samples_path)
    samples_df['timestamp'] = pd.to_datetime(samples_df['timestamp'], utc=True)
    logger.info(f"Loaded {len(samples_df):,} samples")

    # Check for resume
    resume_from = None
    if args.resume:
        progress = load_progress(output_dir)
        if progress:
            resume_from = progress['last_completed_item']
            logger.info(f"Resuming from checkpoint: item {resume_from}")

    # Create config
    config = DataConfig()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: Collect volume statistics (if enabled and not resuming)
    volume_stats = None
    volume_stats_path = output_dir / 'volume_stats.json'

    if enable_volume:
        if volume_stats_path.exists() and args.resume:
            logger.info(f"Loading existing volume stats from {volume_stats_path}")
            volume_stats = load_volume_stats(volume_stats_path)
        else:
            volume_stats = collect_volume_stats_pass1(samples_df, parquet_dir, config)
            save_volume_stats(volume_stats, volume_stats_path)
            logger.info(f"Saved volume stats to {volume_stats_path}")

    # Pass 2: Process all items with feature + volume target computation
    train_count, val_count = process_items(
        samples_df,
        parquet_dir,
        output_dir,
        config,
        chunk_size=args.chunk_size,
        resume_from=resume_from,
        volume_stats=volume_stats,
        enable_volume=enable_volume
    )

    # Write metadata
    metadata = {
        'created': datetime.utcnow().isoformat(),
        'samples_file': str(samples_path),
        'parquet_dir': str(parquet_dir),
        'chunk_size': args.chunk_size,
        'train_samples': train_count,
        'val_samples': val_count,
        'enable_volume': enable_volume,
        'config': {
            'recent_hours': config.recent_hours,
            'medium_days': config.medium_days,
            'long_days': config.long_days,
            'horizons': list(config.horizons),
        }
    }
    if enable_volume and volume_stats:
        metadata['volume_stats'] = {
            'total_items': volume_stats['_metadata']['total_items'],
            'fallback_items': volume_stats['_metadata']['fallback_items'],
            'global_mean': volume_stats['_metadata']['global_mean'],
            'global_std': volume_stats['_metadata']['global_std'],
        }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Run validation
    logger.info("Running validation on precomputed features...")
    report = validate_stage3_output(output_dir, n_samples=args.validate_samples)
    report.to_json(output_dir / "validation_report.json")

    if report.failed > 0:
        logger.warning(f"Validation found {report.failed} invalid chunks")
        return 1

    logger.info("Stage 3 complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
