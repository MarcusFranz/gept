#!/usr/bin/env python3
"""
RunPod Data Preparation Script

Pulls raw price data from PostgreSQL and computes features locally on RunPod.
This is faster than using GCS since RunPod has beefier CPU/RAM.

Prerequisites:
    1. SSH tunnel to PostgreSQL:
       ssh -i /root/.ssh/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128 -N &

    2. Or set DATABASE_URL environment variable for direct connection

Usage:
    # Pull data for all items (creates parquet files locally)
    python prepare_runpod_data.py --output-dir /root/data/prepared

    # Pull data for specific items
    python prepare_runpod_data.py --output-dir /root/data/prepared --items 2,10006,10008

    # With custom date range
    python prepare_runpod_data.py --output-dir /root/data/prepared --months 6
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Optional, Tuple

import pandas as pd
import psycopg2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from feature_engine import FeatureEngine, Granularity
    HAS_FEATURE_ENGINE = True
except ImportError:
    HAS_FEATURE_ENGINE = False
    print("WARNING: feature_engine not found. Copy src/feature_engine.py to this directory.")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# =============================================================================
# Configuration
# =============================================================================

# Default connection (via SSH tunnel)
# Note: This script uses its own connection params instead of src/db_utils.py
# because it's designed for standalone use on remote GPU servers (RunPod, WSL)
# that may not have the full repository cloned.
DEFAULT_CONN_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'osrs_data',
    'user': 'osrs_user',
    'password': os.environ.get('DB_PASS')  # Required: set DB_PASS env var
}

# Override with DATABASE_URL if set
DATABASE_URL = os.environ.get('DATABASE_URL')

DEFAULT_OUTPUT_DIR = Path('/root/data/prepared')
DEFAULT_MONTHS = 6
MIN_ROWS = 5000  # Minimum rows required per item (lowered from 10K)
MAX_ITEMS = 400  # Target ~300-500 high-quality items

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Credential Validation
# =============================================================================

def validate_database_credentials() -> None:
    """
    Validate database credentials at startup before any data operations.

    Raises:
        SystemExit: If credentials are missing or connection fails.
    """
    errors = []

    # Check for DATABASE_URL or DB_PASS
    if not DATABASE_URL:
        db_pass = os.environ.get('DB_PASS')
        if not db_pass:
            errors.append("DB_PASS environment variable not set (required when DATABASE_URL is not set)")
        elif len(db_pass.strip()) == 0:
            errors.append("DB_PASS is set but empty")

    if errors:
        for error in errors:
            logger.error(f"Database credential validation failed: {error}")
        logger.error("Please set DB_PASS or DATABASE_URL environment variable")
        sys.exit(1)

    # Test actual database connection
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        logger.info("Database credentials validated successfully")
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        if "password authentication failed" in str(e):
            logger.error("Check that DB_PASS is correct")
        elif "could not connect to server" in str(e):
            logger.error("Check that SSH tunnel is running or DATABASE_URL is correctly configured")
        else:
            logger.error("Ensure database server is reachable and credentials are correct")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        sys.exit(1)


# =============================================================================
# Database Functions
# =============================================================================

def get_connection():
    """Get database connection."""
    if DATABASE_URL:
        return psycopg2.connect(DATABASE_URL)
    return psycopg2.connect(**DEFAULT_CONN_PARAMS)


def get_training_date_range(months: int = DEFAULT_MONTHS) -> Tuple[str, str]:
    """Get dynamic training date range based on latest data in database."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT MAX(timestamp) FROM price_data_5min")
    result = cur.fetchone()
    conn.close()

    if result[0] is None:
        raise ValueError("No price data found in database")

    end_date = result[0]
    start_date = end_date - relativedelta(months=months)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d %H:%M:%S')


def load_items_from_db(start_date: str, end_date: str, min_rows: int = MIN_ROWS, max_items: int = MAX_ITEMS) -> List[Dict]:
    """Load list of items with sufficient data, ranked by trade volume."""
    conn = get_connection()
    cur = conn.cursor()

    # Query ranks items by total trade volume (high_volume + low_volume)
    # This prioritizes actively traded items over dead content
    cur.execute("""
        SELECT p.item_id, i.name, COUNT(*) as row_count,
               COALESCE(SUM(p.high_price_volume), 0) + COALESCE(SUM(p.low_price_volume), 0) as total_volume
        FROM price_data_5min p
        JOIN items i ON p.item_id = i.item_id
        WHERE p.timestamp >= %s AND p.timestamp <= %s
        GROUP BY p.item_id, i.name
        HAVING COUNT(*) > %s
        ORDER BY total_volume DESC
        LIMIT %s
    """, (start_date, end_date, min_rows, max_items))

    items = [{'item_id': r[0], 'name': r[1], 'row_count': r[2], 'total_volume': r[3]} for r in cur.fetchall()]
    conn.close()

    return items


def load_price_data(item_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data for a single item."""
    conn = get_connection()

    query = """
        SELECT timestamp, avg_high_price, avg_low_price,
               high_price_volume, low_price_volume
        FROM price_data_5min
        WHERE item_id = %s
          AND timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp
    """

    df = pd.read_sql(query, conn, params=[item_id, start_date, end_date])
    conn.close()

    return df


# =============================================================================
# Feature Computation
# =============================================================================

def compute_features_for_item(
    item: Dict,
    start_date: str,
    end_date: str,
    feature_engine: Optional['FeatureEngine'] = None
) -> Dict:
    """
    Load price data and compute features for a single item.

    Returns:
        Dict with prepared data or error status
    """
    item_id = item['item_id']
    item_name = item['name']

    try:
        # Load raw price data
        df = load_price_data(item_id, start_date, end_date)

        if len(df) < 1000:
            return {
                'item_id': item_id,
                'item_name': item_name,
                'status': 'insufficient_data',
                'rows': len(df)
            }

        # Compute features
        if feature_engine is not None:
            df_features = feature_engine.compute_features(df)
        else:
            # Minimal feature computation if feature_engine not available
            df_features = compute_minimal_features(df)

        # Get feature columns from FeatureEngine (authoritative source)
        if feature_engine is not None:
            canonical_features = set(feature_engine.get_feature_columns())
            feature_cols = [c for c in df_features.columns if c in canonical_features]
        else:
            # Fallback for environments without FeatureEngine - use complete prefixes
            feature_prefixes = [
                'return_', 'volatility_', 'volume_', 'ma_', 'spread',
                'rsi_', 'hour', 'momentum_', 'log_', 'high_', 'low_',
                # Critical additions to match FeatureEngine output:
                'range_', 'dist_', 'parkinson_', 'dow_', 'is_', 'mid_'
            ]
            feature_cols = [c for c in df_features.columns
                            if any(c.startswith(p) for p in feature_prefixes)]

        # Keep essential columns for target computation
        essential_cols = ['high', 'low', 'timestamp']
        keep_cols = list(set(feature_cols + [c for c in essential_cols if c in df_features.columns]))

        # Remove duplicate columns from DataFrame before selecting
        df_features = df_features.loc[:, ~df_features.columns.duplicated()]

        # Filter keep_cols to only those that exist after deduplication
        keep_cols = [c for c in keep_cols if c in df_features.columns]

        # SMART NaN HANDLING:
        # 1. Skip warmup rows where rolling windows haven't filled (max window = 288)
        # 2. Only require non-NaN for essential columns used in target computation
        # 3. Forward-fill sporadic NaN values in feature columns

        WARMUP_ROWS = 300  # Slightly more than max rolling window (288)

        df_subset = df_features[keep_cols].iloc[WARMUP_ROWS:].copy()

        # Essential columns must be non-NaN (needed for target computation)
        essential_for_targets = ['high', 'low']
        essential_present = [c for c in essential_for_targets if c in df_subset.columns]

        if essential_present:
            df_subset = df_subset.dropna(subset=essential_present)

        # Forward-fill sporadic NaN values in feature columns (common for volume gaps)
        feature_cols_present = [c for c in keep_cols if c not in essential_for_targets + ['timestamp']]
        if feature_cols_present:
            df_subset[feature_cols_present] = df_subset[feature_cols_present].ffill().bfill()

        # Final check: drop any remaining rows with NaN (should be minimal now)
        df_clean = df_subset.dropna()

        if len(df_clean) < 1000:
            return {
                'item_id': item_id,
                'item_name': item_name,
                'status': 'insufficient_clean_data',
                'rows': len(df_clean)
            }

        return {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'success',
            'data': df_clean,
            'feature_cols': [c for c in feature_cols if c in df_clean.columns],
            'rows': len(df_clean)
        }

    except Exception as e:
        import traceback
        return {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def compute_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute minimal features when feature_engine is not available.

    This provides basic functionality but the full feature_engine is preferred.
    """
    df = df.copy()

    # Rename columns
    if 'avg_high_price' in df.columns:
        df['high'] = df['avg_high_price']
        df['low'] = df['avg_low_price']

    # Basic features
    df['mid'] = (df['high'] + df['low']) / 2

    # Returns
    for periods in [3, 12, 24, 48, 96]:  # 15min, 1h, 2h, 4h, 8h
        df[f'return_{periods}p'] = df['mid'].pct_change(periods)

    # Volatility
    for window in [12, 48, 144]:  # 1h, 4h, 12h
        df[f'volatility_{window}p'] = df['mid'].pct_change().rolling(window).std()

    # Volume features
    if 'high_price_volume' in df.columns:
        df['volume_total'] = df['high_price_volume'].fillna(0) + df['low_price_volume'].fillna(0)
        for window in [12, 48]:
            df[f'volume_ma_{window}p'] = df['volume_total'].rolling(window).mean()

    # Moving averages
    for window in [12, 48, 144, 288]:
        df[f'ma_{window}p'] = df['mid'].rolling(window).mean()
        df[f'ma_{window}p_diff'] = (df['mid'] - df[f'ma_{window}p']) / df[f'ma_{window}p']

    # Spread
    df['spread'] = (df['high'] - df['low']) / df['mid']

    # Time features
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek

    # High/low normalized
    df['high_low_range'] = df['high'] - df['low']
    for window in [12, 48]:
        rolling_high = df['high'].rolling(window).max()
        rolling_low = df['low'].rolling(window).min()
        df[f'high_N_{window}p'] = (df['high'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        df[f'low_N_{window}p'] = (df['low'] - rolling_low) / (rolling_high - rolling_low + 1e-8)

    return df


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare training data on RunPod')

    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--items', type=str, help='Comma-separated item IDs (default: all)')
    parser.add_argument('--months', type=int, default=DEFAULT_MONTHS,
                        help=f'Months of historical data (default: {DEFAULT_MONTHS})')
    parser.add_argument('--min-rows', type=int, default=MIN_ROWS,
                        help=f'Minimum rows required (default: {MIN_ROWS})')
    parser.add_argument('--max-items', type=int, default=MAX_ITEMS,
                        help=f'Maximum items to process, ranked by volume (default: {MAX_ITEMS})')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID (default: YYYYMMDD_HHMMSS)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 70)
    logger.info("GePT RunPod Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Months: {args.months}")

    start_time = time.time()

    # Validate database credentials at startup
    logger.info("\nValidating database credentials...")
    validate_database_credentials()

    # Test database connection and show table stats
    logger.info("\nTesting database connection...")
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM price_data_5min")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        logger.info(f"  ✓ Connected. price_data_5min has {count:,} rows")
    except Exception as e:
        logger.error(f"  ✗ Database connection failed: {e}")
        logger.error("  Make sure SSH tunnel is running or DATABASE_URL is set")
        sys.exit(1)

    # Get date range
    logger.info(f"\nComputing date range ({args.months} months from latest)...")
    start_date, end_date = get_training_date_range(args.months)
    logger.info(f"  Start: {start_date}")
    logger.info(f"  End:   {end_date}")

    # Load items (ranked by volume, limited to max_items)
    logger.info(f"\nLoading top {args.max_items} items by trade volume...")
    all_items = load_items_from_db(start_date, end_date, args.min_rows, args.max_items)
    logger.info(f"  Found {len(all_items)} items with >= {args.min_rows} rows (top by volume)")

    # Filter items if specified
    if args.items:
        item_ids = set(int(x.strip()) for x in args.items.split(','))
        items = [i for i in all_items if i['item_id'] in item_ids]
        logger.info(f"  Filtered to {len(items)} specified items")
    else:
        items = all_items

    # Create output directory
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize feature engine
    feature_engine = None
    if HAS_FEATURE_ENGINE:
        feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        logger.info("\n  Using full FeatureEngine (102 features)")
    else:
        logger.warning("\n  Using minimal features (feature_engine not available)")

    # Process items
    logger.info(f"\nProcessing {len(items)} items...")
    results = []
    errors = []

    if HAS_TQDM:
        items_iter = tqdm(items, desc="Preparing data")
    else:
        items_iter = items

    for i, item in enumerate(items_iter):
        item_id = item['item_id']
        item_name = item['name']

        # Skip if parquet already exists (resume support)
        parquet_path = output_dir / f'{item_id}.parquet'
        if parquet_path.exists():
            results.append({
                'item_id': item_id,
                'item_name': item_name,
                'rows': -1,  # Unknown, already processed
                'feature_cols': []
            })
            continue

        if not HAS_TQDM and (i + 1) % 25 == 0:
            logger.info(f"  Progress: {i+1}/{len(items)}")

        result = compute_features_for_item(item, start_date, end_date, feature_engine)

        if result['status'] == 'success':
            # Save parquet
            df = result.pop('data')
            parquet_path = output_dir / f'{item_id}.parquet'
            df.to_parquet(parquet_path, compression='snappy')

            results.append({
                'item_id': item_id,
                'item_name': item_name,
                'rows': result['rows'],
                'feature_cols': result['feature_cols']
            })
        else:
            errors.append(result)

    # Get canonical feature columns for global config
    canonical_feature_cols = None
    if feature_engine is not None:
        canonical_feature_cols = feature_engine.get_feature_columns()
        logger.info(f"Canonical feature columns: {len(canonical_feature_cols)} features")

    # Save config
    config = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(),
        'start_date': start_date,
        'end_date': end_date,
        'months': args.months,
        'feature_cols': canonical_feature_cols,  # Global canonical feature list
        'items': results,
    }

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Successful: {len(results)}/{len(items)}")
    logger.info(f"Errors: {len(errors)}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Run ID: {run_id}")

    if errors:
        errors_path = output_dir / 'errors.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"Errors saved to: {errors_path}")

    logger.info("\nTo train models, run:")
    logger.info(f"  python train_runpod_multitarget.py --run-id {run_id} --all")
    logger.info(f"  (with --cache-dir {output_dir.parent})")


if __name__ == '__main__':
    main()
