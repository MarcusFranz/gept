#!/usr/bin/env python3
"""
Convert bulk CSV exports to per-item parquet files for pipeline processing.

Usage:
    python scripts/convert_csv_to_parquet.py \
        --csv-dir data/hydra_export \
        --output-dir data/cleaned \
        --items configs/top_1000_items.txt
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_item_list(items_file: Optional[str]) -> Optional[List[int]]:
    """Load list of item IDs to process."""
    if not items_file:
        return None

    with open(items_file) as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]


def convert_resolution(
    csv_path: Path,
    output_dir: Path,
    resolution: str,
    item_ids: Optional[List[int]] = None,
    chunk_size: int = 5_000_000
) -> int:
    """Convert a single resolution CSV to per-item parquet files."""

    output_subdir = output_dir / resolution
    output_subdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {csv_path.name} -> {output_subdir}/")

    # Read in chunks to handle large files
    items_written = set()

    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunk_size, parse_dates=['timestamp']),
        desc=f"Reading {resolution}"
    ):
        # Filter to requested items if specified
        if item_ids:
            chunk = chunk[chunk['item_id'].isin(item_ids)]

        if chunk.empty:
            continue

        # Group by item and write/append
        for item_id, group in chunk.groupby('item_id'):
            item_path = output_subdir / f"item_{item_id}.parquet"

            # Sort by timestamp
            group = group.sort_values('timestamp').reset_index(drop=True)

            if item_path.exists():
                # Append to existing
                existing = pd.read_parquet(item_path)
                combined = pd.concat([existing, group], ignore_index=True)
                combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                combined.to_parquet(item_path, index=False)
            else:
                group.to_parquet(item_path, index=False)

            items_written.add(item_id)

    logger.info(f"Wrote {len(items_written)} items for {resolution}")
    return len(items_written)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV exports to parquet')
    parser.add_argument('--csv-dir', required=True, help='Directory with CSV exports')
    parser.add_argument('--output-dir', required=True, help='Output directory for parquet files')
    parser.add_argument('--items', help='Optional file with item IDs to include')
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)

    item_ids = load_item_list(args.items)
    if item_ids:
        logger.info(f"Filtering to {len(item_ids)} items")

    # Map CSV files to resolutions
    resolution_map = {
        'price_data_5min.csv': '5min',
        'price_data_1h.csv': '1hour',
        'price_data_4h.csv': '4hour',
    }

    total_items = 0
    for csv_name, resolution in resolution_map.items():
        csv_path = csv_dir / csv_name
        if csv_path.exists():
            count = convert_resolution(csv_path, output_dir, resolution, item_ids)
            total_items = max(total_items, count)
        else:
            logger.warning(f"CSV not found: {csv_path}")

    logger.info(f"Conversion complete. {total_items} items processed.")
    logger.info(f"Output written to: {output_dir}")
    logger.info(f"\nNext: Run stage2_sample.py and stage3_precompute.py")


if __name__ == '__main__':
    main()
