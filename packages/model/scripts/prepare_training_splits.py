#!/usr/bin/env python3
"""
Prepare Training Data with Time-Based Splits and Item Selection
================================================================

Implements strict time-based splitting with 48h horizon buffers:
- Train: 2023-11-01 → 2025-08-31
- Val:   2025-09-03 → 2025-11-30 (48h after train)
- Test:  2025-12-03 → 2026-01-31 (48h after val)

Item selection computed from train split only (no leakage):
- Full training: items with valid_samples >= 1000 AND coverage >= 0.6
- Optuna subset: top 1000 by volume + stratified tail sample

Usage:
    python scripts/prepare_training_splits.py --db-url postgresql://... --output-dir data/splits/

    # Or with SSH tunnel to Ampere:
    python scripts/prepare_training_splits.py --output-dir data/splits/
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SPLIT CONFIGURATION
# =============================================================================

@dataclass
class SplitConfig:
    """Time-based split boundaries with 48h horizon buffer."""
    train_start: datetime = datetime(2023, 11, 1, tzinfo=None)
    train_end: datetime = datetime(2025, 8, 31, 23, 55, tzinfo=None)

    val_start: datetime = datetime(2025, 9, 3, 0, 0, tzinfo=None)  # 48h after train
    val_end: datetime = datetime(2025, 11, 30, 23, 55, tzinfo=None)

    test_start: datetime = datetime(2025, 12, 3, 0, 0, tzinfo=None)  # 48h after val
    test_end: datetime = datetime(2026, 1, 31, 23, 55, tzinfo=None)

    # Horizon settings
    max_horizon_hours: int = 48
    bars_per_hour: int = 12  # 5-min bars

    @property
    def max_horizon_bars(self) -> int:
        return self.max_horizon_hours * self.bars_per_hour

    @property
    def horizon_buffer(self) -> timedelta:
        return timedelta(hours=self.max_horizon_hours)


@dataclass
class ItemStats:
    """Per-item statistics computed from train split."""
    item_id: int
    valid_samples: int
    coverage: float
    volume_score: float
    mean_price: float


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_db_connection(db_url: Optional[str] = None):
    """Get database connection, using env vars if URL not provided."""
    import psycopg2

    if db_url:
        return psycopg2.connect(db_url)

    # Try environment variables
    db_url = os.environ.get('DB_CONNECTION_STRING')
    if db_url:
        return psycopg2.connect(db_url)

    # Construct from individual vars
    host = os.environ.get('DB_HOST', 'localhost')
    port = os.environ.get('DB_PORT', '5432')
    dbname = os.environ.get('DB_NAME', 'osrs_data')
    user = os.environ.get('DB_USER', 'osrs_user')
    password = os.environ.get('DB_PASS', '')

    return psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password
    )


# =============================================================================
# ITEM STATISTICS (TRAIN ONLY)
# =============================================================================

def compute_item_stats(conn, config: SplitConfig) -> List[ItemStats]:
    """
    Compute per-item statistics from TRAIN split only.

    Returns list of ItemStats with:
    - valid_samples: count of usable 5-min bars
    - coverage: valid_samples / total_possible_bars
    - volume_score: sum(log1p(high_vol + low_vol))
    """
    logger.info("Computing item statistics from train split...")

    # Total possible bars in train period
    train_duration = config.train_end - config.train_start
    total_possible_bars = int(train_duration.total_seconds() / 300)  # 5-min bars

    query = """
    SELECT
        item_id,
        COUNT(*) as valid_samples,
        SUM(LN(1 + COALESCE(high_price_volume, 0) + COALESCE(low_price_volume, 0))) as volume_score,
        AVG(COALESCE(avg_high_price, avg_low_price)) as mean_price
    FROM price_data_5min
    WHERE timestamp >= %s AND timestamp <= %s
    GROUP BY item_id
    ORDER BY volume_score DESC
    """

    df = pd.read_sql(query, conn, params=(config.train_start, config.train_end))

    stats = []
    for _, row in df.iterrows():
        stats.append(ItemStats(
            item_id=int(row['item_id']),
            valid_samples=int(row['valid_samples']),
            coverage=row['valid_samples'] / total_possible_bars,
            volume_score=float(row['volume_score'] or 0),
            mean_price=float(row['mean_price'] or 0),
        ))

    logger.info(f"Computed stats for {len(stats)} items")
    return stats


def select_items(
    stats: List[ItemStats],
    min_samples: int = 1000,
    min_coverage: float = 0.6,
    optuna_top_k: int = 1000,
    optuna_tail_sample: int = 150,
) -> Tuple[List[int], List[int]]:
    """
    Select items for full training and Optuna subset.

    Returns:
        (full_items, optuna_items) - lists of item IDs
    """
    # Full training: filter by quality thresholds
    full_items = [
        s.item_id for s in stats
        if s.valid_samples >= min_samples and s.coverage >= min_coverage
    ]
    logger.info(f"Full training items: {len(full_items)} (samples>={min_samples}, coverage>={min_coverage})")

    # Optuna subset: top K by volume + stratified tail
    # Stats are already sorted by volume_score DESC
    qualified = [s for s in stats if s.valid_samples >= min_samples and s.coverage >= min_coverage]

    # Top K by volume
    top_items = [s.item_id for s in qualified[:optuna_top_k]]

    # Stratified tail: sample from remaining items across volume deciles
    remaining = qualified[optuna_top_k:]
    if remaining and optuna_tail_sample > 0:
        # Split into deciles and sample evenly
        n_deciles = 10
        samples_per_decile = optuna_tail_sample // n_deciles
        tail_items = []

        decile_size = max(1, len(remaining) // n_deciles)
        for i in range(n_deciles):
            decile = remaining[i * decile_size : (i + 1) * decile_size]
            if decile:
                # Sample from this decile
                n_sample = min(samples_per_decile, len(decile))
                indices = np.linspace(0, len(decile) - 1, n_sample, dtype=int)
                tail_items.extend([decile[j].item_id for j in indices])

        optuna_items = top_items + tail_items
    else:
        optuna_items = top_items

    logger.info(f"Optuna items: {len(optuna_items)} (top {len(top_items)} + {len(optuna_items) - len(top_items)} tail)")

    return full_items, optuna_items


# =============================================================================
# DATA EXPORT
# =============================================================================

def export_split_data(
    conn,
    config: SplitConfig,
    item_ids: List[int],
    output_dir: Path,
    resolution: str = '5min',
) -> Dict[str, int]:
    """
    Export data for specified items, organized by split.

    Creates:
        output_dir/
            train/5min/item_{id}.parquet
            val/5min/item_{id}.parquet
            test/5min/item_{id}.parquet
    """
    splits = {
        'train': (config.train_start, config.train_end),
        'val': (config.val_start, config.val_end),
        'test': (config.test_start, config.test_end),
    }

    row_counts = {}

    for split_name, (start, end) in splits.items():
        split_dir = output_dir / split_name / resolution
        split_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting {split_name} split ({start} → {end})...")

        total_rows = 0
        for item_id in tqdm(item_ids, desc=f"{split_name}"):
            query = """
            SELECT timestamp, avg_high_price, avg_low_price,
                   high_price_volume, low_price_volume
            FROM price_data_5min
            WHERE item_id = %s AND timestamp >= %s AND timestamp <= %s
            ORDER BY timestamp
            """

            df = pd.read_sql(query, conn, params=(item_id, start, end))

            if len(df) > 0:
                df['item_id'] = item_id
                df.to_parquet(split_dir / f"item_{item_id}.parquet", index=False)
                total_rows += len(df)

        row_counts[split_name] = total_rows
        logger.info(f"  {split_name}: {total_rows:,} rows")

    return row_counts


def export_aggregated(
    conn,
    config: SplitConfig,
    item_ids: List[int],
    output_dir: Path,
) -> None:
    """Export hourly and 4-hourly aggregated data."""

    for agg_hours, resolution in [(1, '1hour'), (4, '4hour')]:
        logger.info(f"Exporting {resolution} aggregated data...")

        for split_name, (start, end) in [
            ('train', (config.train_start, config.train_end)),
            ('val', (config.val_start, config.val_end)),
            ('test', (config.test_start, config.test_end)),
        ]:
            split_dir = output_dir / split_name / resolution
            split_dir.mkdir(parents=True, exist_ok=True)

            for item_id in tqdm(item_ids, desc=f"{split_name}/{resolution}"):
                if agg_hours == 1:
                    query = """
                    SELECT
                        date_trunc('hour', timestamp) as timestamp,
                        AVG(avg_high_price) as avg_high_price,
                        AVG(avg_low_price) as avg_low_price,
                        SUM(high_price_volume) as high_price_volume,
                        SUM(low_price_volume) as low_price_volume
                    FROM price_data_5min
                    WHERE item_id = %s AND timestamp >= %s AND timestamp <= %s
                    GROUP BY date_trunc('hour', timestamp)
                    ORDER BY timestamp
                    """
                else:
                    query = """
                    SELECT
                        date_trunc('hour', timestamp) -
                            (EXTRACT(hour FROM timestamp)::int %% %s) * INTERVAL '1 hour' as timestamp,
                        AVG(avg_high_price) as avg_high_price,
                        AVG(avg_low_price) as avg_low_price,
                        SUM(high_price_volume) as high_price_volume,
                        SUM(low_price_volume) as low_price_volume
                    FROM price_data_5min
                    WHERE item_id = %%s AND timestamp >= %%s AND timestamp <= %%s
                    GROUP BY date_trunc('hour', timestamp) -
                        (EXTRACT(hour FROM timestamp)::int %% %s) * INTERVAL '1 hour'
                    ORDER BY timestamp
                    """ % (agg_hours, agg_hours)

                if agg_hours == 1:
                    df = pd.read_sql(query, conn, params=(item_id, start, end))
                else:
                    df = pd.read_sql(query, conn, params=(item_id, start, end))

                if len(df) > 0:
                    df['item_id'] = item_id
                    df.to_parquet(split_dir / f"item_{item_id}.parquet", index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare training splits with item selection')
    parser.add_argument('--db-url', help='Database connection URL')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--min-samples', type=int, default=1000, help='Min samples for item inclusion')
    parser.add_argument('--min-coverage', type=float, default=0.6, help='Min coverage for item inclusion')
    parser.add_argument('--optuna-only', action='store_true', help='Export only Optuna subset')
    parser.add_argument('--skip-aggregated', action='store_true', help='Skip 1h/4h exports')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SplitConfig()

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING DATA PREPARATION                                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Train: {config.train_start.date()} → {config.train_end.date()}                        ║
║  Val:   {config.val_start.date()} → {config.val_end.date()}  (48h buffer)             ║
║  Test:  {config.test_start.date()} → {config.test_end.date()}  (48h buffer)             ║
║                                                                  ║
║  Max horizon: {config.max_horizon_hours}h ({config.max_horizon_bars} bars)                              ║
║  Min samples: {args.min_samples}                                              ║
║  Min coverage: {args.min_coverage}                                               ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Connect to database
    logger.info("Connecting to database...")
    conn = get_db_connection(args.db_url)

    try:
        # Step 1: Compute item statistics from train split
        stats = compute_item_stats(conn, config)

        # Step 2: Select items
        full_items, optuna_items = select_items(
            stats,
            min_samples=args.min_samples,
            min_coverage=args.min_coverage,
        )

        # Save item lists
        with open(output_dir / 'full_items.txt', 'w') as f:
            f.write('\n'.join(map(str, full_items)))
        with open(output_dir / 'optuna_items.txt', 'w') as f:
            f.write('\n'.join(map(str, optuna_items)))

        # Save stats
        stats_dict = {s.item_id: asdict(s) for s in stats}
        with open(output_dir / 'item_stats.json', 'w') as f:
            json.dump(stats_dict, f, indent=2)

        # Step 3: Export data
        items_to_export = optuna_items if args.optuna_only else full_items
        logger.info(f"Exporting data for {len(items_to_export)} items...")

        # Export 5-min data
        row_counts = export_split_data(conn, config, items_to_export, output_dir)

        # Export aggregated data
        if not args.skip_aggregated:
            export_aggregated(conn, config, items_to_export, output_dir)

        # Save manifest
        manifest = {
            'created': datetime.now().isoformat(),
            'config': {
                'train_start': config.train_start.isoformat(),
                'train_end': config.train_end.isoformat(),
                'val_start': config.val_start.isoformat(),
                'val_end': config.val_end.isoformat(),
                'test_start': config.test_start.isoformat(),
                'test_end': config.test_end.isoformat(),
                'max_horizon_hours': config.max_horizon_hours,
            },
            'items': {
                'full_count': len(full_items),
                'optuna_count': len(optuna_items),
                'exported': len(items_to_export),
            },
            'row_counts': row_counts,
        }
        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  EXPORT COMPLETE                                                 ║
╠══════════════════════════════════════════════════════════════════╣
║  Items exported: {len(items_to_export):,}                                           ║
║  Train rows: {row_counts.get('train', 0):,}                                      ║
║  Val rows:   {row_counts.get('val', 0):,}                                        ║
║  Test rows:  {row_counts.get('test', 0):,}                                       ║
║                                                                  ║
║  Output: {str(output_dir):<52} ║
╚══════════════════════════════════════════════════════════════════╝
""")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
