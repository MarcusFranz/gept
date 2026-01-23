"""
Stage 1: Extract & Clean
========================

Extract price data from PostgreSQL, clean it, and write to parquet files
at multiple resolutions (5min, 1hour, 4hour).

Usage:
    python -m src.pipeline.stage1_extract --items configs/items_test.txt --output data/cleaned/
    python -m src.pipeline.stage1_extract --validate-only --output data/cleaned/
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.db_utils import get_simple_connection
from src.pipeline.validation import validate_stage1_output, ValidationReport

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_item_data(
    item_id: int,
    start_date: datetime,
    end_date: datetime,
    conn
) -> pd.DataFrame:
    """
    Extract 5-minute price data for a single item from PostgreSQL.

    Args:
        item_id: The item ID to extract
        start_date: Start of date range (inclusive)
        end_date: End of date range (exclusive)
        conn: Database connection

    Returns:
        DataFrame with columns: timestamp, avg_high_price, avg_low_price,
                               high_price_volume, low_price_volume
    """
    query = """
        SELECT
            timestamp,
            avg_high_price,
            avg_low_price,
            high_price_volume,
            low_price_volume
        FROM price_data_5min
        WHERE item_id = %s
          AND timestamp >= %s
          AND timestamp < %s
        ORDER BY timestamp ASC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(item_id, start_date, end_date)
    )

    # Ensure timezone-aware timestamps
    if len(df) > 0:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    return df


def clean_5min_data(
    df: pd.DataFrame,
    outlier_threshold: float = 0.5,
    max_fill_gaps: int = 6  # 30 minutes
) -> pd.DataFrame:
    """
    Clean 5-minute price data:
    - Forward-fill small gaps (<=30 min)
    - Mark longer gaps with 'gap' flag
    - Flag outliers (>threshold price change in 5 min)

    Args:
        df: Raw 5-min data with timestamp, prices, volumes
        outlier_threshold: Max allowed 5-min price change (0.5 = 50%)
        max_fill_gaps: Max consecutive missing rows to forward-fill

    Returns:
        Cleaned DataFrame with gap and outlier flag columns
    """
    if len(df) == 0:
        return df

    df = df.copy()
    df = df.set_index('timestamp').sort_index()

    # Drop exact duplicate timestamps (keep first)
    df = df[~df.index.duplicated(keep='first')]

    # Create complete timestamp index (5-min intervals)
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='5min',
        tz=timezone.utc
    )

    # Reindex to complete timeline
    df = df.reindex(full_idx)

    # Identify gaps (missing data before fill)
    was_missing = df['avg_high_price'].isna()

    # Count consecutive missing values
    gap_groups = (~was_missing).cumsum()
    gap_sizes = was_missing.groupby(gap_groups).transform('sum')

    # Mark large gaps (don't fill these)
    large_gaps = was_missing & (gap_sizes > max_fill_gaps)

    # Forward fill small gaps only
    df['avg_high_price'] = df['avg_high_price'].ffill(limit=max_fill_gaps)
    df['avg_low_price'] = df['avg_low_price'].ffill(limit=max_fill_gaps)
    df['high_price_volume'] = df['high_price_volume'].fillna(0)
    df['low_price_volume'] = df['low_price_volume'].fillna(0)

    # Mark gap rows
    df['gap'] = large_gaps | df['avg_high_price'].isna()

    # Drop rows that are still NaN after fill attempts
    df = df.dropna(subset=['avg_high_price', 'avg_low_price'])

    # Detect outliers (>threshold price change from previous)
    mid_price = (df['avg_high_price'] + df['avg_low_price']) / 2
    pct_change = mid_price.pct_change().abs()
    df['outlier'] = pct_change > outlier_threshold

    # Reset index for output
    df = df.reset_index()
    df = df.rename(columns={'index': 'timestamp'})

    return df


def aggregate_to_1hour(df_5min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 5-minute data to 1-hour OHLC bars.

    Creates Open/High/Low/Close prices and sums volumes.
    Tracks sample count and staleness per bar.
    """
    if len(df_5min) == 0:
        return pd.DataFrame()

    df = df_5min.copy()
    df = df.set_index('timestamp')

    # Create mid price for OHLC
    df['mid_high'] = df['avg_high_price']
    df['mid_low'] = df['avg_low_price']

    # Resample to 1-hour bars
    agg = df.resample('1h').agg({
        'mid_high': ['first', 'max', 'min', 'last'],
        'mid_low': ['first', 'max', 'min', 'last'],
        'high_price_volume': 'sum',
        'low_price_volume': 'sum',
        'gap': 'sum',
        'outlier': 'sum'
    })

    # Flatten column names
    agg.columns = [
        'high_open', 'high_high', 'high_low', 'high_close',
        'low_open', 'low_high', 'low_low', 'low_close',
        'high_volume', 'low_volume',
        'gap_count', 'outlier_count'
    ]

    # Sample count (12 expected per hour)
    agg['sample_count'] = df.resample('1h').size()
    agg['staleness'] = 12 - agg['sample_count']  # Missing samples

    # Drop bars with no data
    agg = agg.dropna(subset=['high_close'])

    return agg.reset_index()


def aggregate_to_4hour(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-hour data to 4-hour OHLC bars.
    """
    if len(df_1h) == 0:
        return pd.DataFrame()

    df = df_1h.copy()
    df = df.set_index('timestamp')

    # Resample to 4-hour bars
    agg = df.resample('4h').agg({
        'high_open': 'first',
        'high_high': 'max',
        'high_low': 'min',
        'high_close': 'last',
        'low_open': 'first',
        'low_high': 'max',
        'low_low': 'min',
        'low_close': 'last',
        'high_volume': 'sum',
        'low_volume': 'sum',
        'gap_count': 'sum',
        'outlier_count': 'sum',
        'sample_count': 'sum',
        'staleness': 'sum'
    })

    # Drop bars with no data
    agg = agg.dropna(subset=['high_close'])

    return agg.reset_index()


def process_item(
    item_id: int,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    conn,
    outlier_threshold: float = 0.5
) -> Tuple[bool, Optional[str], dict]:
    """
    Process a single item: extract, clean, aggregate, write parquet files.

    Returns:
        Tuple of (success, error_reason, stats_dict)
    """
    logger.info(f"Processing item {item_id}...")
    stats = {'item_id': item_id}

    try:
        # Extract raw data
        df_raw = extract_item_data(item_id, start_date, end_date, conn)
        stats['raw_rows'] = len(df_raw)

        if len(df_raw) == 0:
            return False, "No data in date range", stats

        # Check for negative prices
        if (df_raw['avg_high_price'] < 0).any() or (df_raw['avg_low_price'] < 0).any():
            return False, "Negative prices found", stats

        # Clean 5-min data
        df_5min = clean_5min_data(df_raw, outlier_threshold=outlier_threshold)
        stats['rows_5min'] = len(df_5min)
        stats['gap_pct'] = df_5min['gap'].sum() / len(df_5min) if len(df_5min) > 0 else 0
        stats['outlier_count'] = df_5min['outlier'].sum()

        # Aggregate to 1-hour
        df_1h = aggregate_to_1hour(df_5min)
        stats['rows_1h'] = len(df_1h)

        # Aggregate to 4-hour
        df_4h = aggregate_to_4hour(df_1h)
        stats['rows_4h'] = len(df_4h)

        # Write parquet files
        (output_dir / "5min").mkdir(parents=True, exist_ok=True)
        (output_dir / "1hour").mkdir(parents=True, exist_ok=True)
        (output_dir / "4hour").mkdir(parents=True, exist_ok=True)

        df_5min.to_parquet(output_dir / "5min" / f"item_{item_id}.parquet", index=False)
        df_1h.to_parquet(output_dir / "1hour" / f"item_{item_id}.parquet", index=False)
        df_4h.to_parquet(output_dir / "4hour" / f"item_{item_id}.parquet", index=False)

        logger.info(f"  Item {item_id}: {stats['rows_5min']} 5min rows, "
                    f"{stats['gap_pct']:.1%} gaps, {stats['outlier_count']} outliers")

        return True, None, stats

    except Exception as e:
        logger.error(f"  Item {item_id} failed: {e}")
        return False, str(e), stats


def load_item_ids(path: Path) -> list[int]:
    """Load item IDs from a text file (one per line)."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Extract and clean price data")
    parser.add_argument('--items', type=str, help='Path to items file (one ID per line)')
    parser.add_argument('--start', type=str, default='2024-06-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-01-15', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--outlier-threshold', type=float, default=0.5,
                        help='Outlier detection threshold (default: 0.5 = 50%%)')
    parser.add_argument('--max-gap-pct', type=float, default=0.05,
                        help='Max allowed gap percentage (default: 0.05 = 5%%)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing output, do not extract')
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Validate-only mode
    if args.validate_only:
        logger.info(f"Validating Stage 1 output at {output_dir}...")
        try:
            report = validate_stage1_output(output_dir, max_gap_pct=args.max_gap_pct)
            report.to_json(output_dir / "validation_report.json")
            logger.info(f"Validation complete: {report.passed} passed, {report.failed} failed")
            if report.failed > 0:
                logger.warning(f"Failed items: {[v.item_id for v in report.items.values() if not v.passed]}")
            return 0 if report.failed == 0 else 1
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 1

    # Extraction mode
    if not args.items:
        logger.error("--items is required for extraction mode")
        return 1

    item_ids = load_item_ids(Path(args.items))
    logger.info(f"Loaded {len(item_ids)} item IDs from {args.items}")

    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Output: {output_dir}")

    # Connect to database
    conn = get_simple_connection()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        passed = failed = 0

        for i, item_id in enumerate(item_ids):
            success, reason, stats = process_item(
                item_id, start_date, end_date, output_dir, conn,
                outlier_threshold=args.outlier_threshold
            )
            results.append({'item_id': item_id, 'success': success, 'reason': reason, **stats})
            if success:
                passed += 1
            else:
                failed += 1

            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(item_ids)} items ({passed} passed, {failed} failed)")

        # Write manifest
        manifest = {
            'extraction_date': datetime.utcnow().isoformat(),
            'date_range': [args.start, args.end],
            'items_requested': len(item_ids),
            'items_passed': passed,
            'items_failed': failed,
            'outlier_threshold': args.outlier_threshold,
            'results': results
        }

        with open(output_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2, default=str)

        # Run validation
        logger.info("Running validation on extracted data...")
        report = validate_stage1_output(output_dir, max_gap_pct=args.max_gap_pct)
        report.to_json(output_dir / "validation_report.json")

        logger.info(f"Extraction complete: {passed} passed, {failed} failed")
        logger.info(f"Validation: {report.passed} items valid, {report.failed} items invalid")

        return 0 if failed == 0 and report.failed == 0 else 1

    finally:
        conn.close()


if __name__ == '__main__':
    sys.exit(main())
