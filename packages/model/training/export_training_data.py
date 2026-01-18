#!/usr/bin/env python3
"""
Export training data from PostgreSQL to CSV files for transfer to GPU server.
Creates compressed archive ready for upload.

Output structure:
  data/
    items.json          - List of items with IDs and names
    565.csv             - Price data for item 565
    560.csv             - Price data for item 560
    ...
  training_data.tar.gz  - Compressed archive of all data
"""

import os
import sys
import json
import gzip
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Centralized database connection management
from db_utils import CONN_PARAMS as DB_CONFIG

# Date range
DATE_START = "2025-06-15"
DATE_END = "2026-01-06"

# Output directory
OUTPUT_DIR = Path("data")


def get_items_with_sufficient_data(conn) -> list:
    """Get all items with at least 50,000 rows in date range."""
    query = """
        SELECT p.item_id, i.name, COUNT(*) as cnt
        FROM price_data_5min p
        JOIN items i ON p.item_id = i.item_id
        WHERE p.timestamp >= %s AND p.timestamp <= %s
        GROUP BY p.item_id, i.name
        HAVING COUNT(*) >= 50000
        ORDER BY cnt DESC
    """
    cur = conn.cursor()
    cur.execute(query, (DATE_START, DATE_END))
    items = [{"id": row[0], "name": row[1], "rows": row[2]} for row in cur.fetchall()]
    cur.close()
    return items


def export_item_data(conn, item_id: int, output_path: Path):
    """Export price data for a single item to CSV."""
    query = """
        SELECT timestamp, avg_high_price, avg_low_price,
               high_price_volume, low_price_volume
        FROM price_data_5min
        WHERE item_id = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp
    """
    df = pd.read_sql(query, conn, params=(item_id, DATE_START, DATE_END))
    df.to_csv(output_path, index=False)
    return len(df)


def main():
    print("=" * 60)
    print("EXPORT TRAINING DATA")
    print("=" * 60)
    print(f"Date range: {DATE_START} to {DATE_END}")
    print()

    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)

    # Get items
    print("Finding items with sufficient data...")
    items = get_items_with_sufficient_data(conn)
    print(f"Found {len(items)} items")

    total_rows = sum(item['rows'] for item in items)
    print(f"Total rows to export: {total_rows:,}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Save items list
    items_file = OUTPUT_DIR / "items.json"
    with open(items_file, 'w') as f:
        json.dump([{"id": i["id"], "name": i["name"]} for i in items], f, indent=2)
    print(f"Saved items list to {items_file}")

    # Export each item
    print("\nExporting item data...")
    exported = 0
    for i, item in enumerate(items):
        csv_path = OUTPUT_DIR / f"{item['id']}.csv"
        rows = export_item_data(conn, item['id'], csv_path)
        exported += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(items):
            print(f"  Progress: {i+1}/{len(items)} items exported")

    conn.close()

    # Calculate sizes
    total_size = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.csv"))
    total_size += (OUTPUT_DIR / "items.json").stat().st_size
    print(f"\nTotal uncompressed size: {total_size / 1024 / 1024:.1f} MB")

    # Create compressed archive
    print("\nCreating compressed archive...")
    archive_path = Path("training_data.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(OUTPUT_DIR, arcname="data")

    archive_size = archive_path.stat().st_size
    print(f"Archive created: {archive_path} ({archive_size / 1024 / 1024:.1f} MB)")
    print(f"Compression ratio: {total_size / archive_size:.1f}x")

    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Items exported: {exported}")
    print(f"Archive: {archive_path.absolute()}")
    print()
    print("To transfer to GPU server:")
    print(f"  scp {archive_path} user@gpu-server:/path/to/training/")
    print("  ssh user@gpu-server 'cd /path/to/training && tar -xzf training_data.tar.gz'")


if __name__ == "__main__":
    main()
