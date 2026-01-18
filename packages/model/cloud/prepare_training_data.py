#!/usr/bin/env python3
"""
Step 1: Prepare training data locally and upload to GCS.

Run this on the Ampere server (or anywhere with DB access) before triggering Cloud Run.

Usage:
    python prepare_training_data.py --bucket osrs-models

This script:
1. Loads price data for all items from local DB
2. Computes features and targets
3. Uploads precomputed data to GCS as parquet files
4. Triggers Cloud Run job for parallel training
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
import psycopg2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engine import FeatureEngine, Granularity
from target_engine import DiscreteHourTargetEngine, DiscreteHourConfig

# Config
CONN_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'osrs_data',
    'user': 'osrs_user',
    'password': os.environ['DB_PASS']  # Required - no default for security
}

# Training parameters
DEFAULT_TRAINING_MONTHS = 6
TARGET_OFFSETS = [0.02, 0.025]
DISCRETE_HOURS = list(range(1, 25))


def get_training_date_range(months: int = DEFAULT_TRAINING_MONTHS) -> tuple:
    """
    Get dynamic training date range based on latest data in database.

    Args:
        months: Number of months of historical data to use (default: 6)

    Returns:
        Tuple of (start_date, end_date) as strings in 'YYYY-MM-DD' format
    """
    conn = psycopg2.connect(**CONN_PARAMS)
    cur = conn.cursor()

    # Get the latest timestamp in the price data
    cur.execute("SELECT MAX(timestamp) FROM price_data_5min")
    result = cur.fetchone()
    conn.close()

    if result[0] is None:
        raise ValueError("No price data found in database")

    end_date = result[0]
    start_date = end_date - relativedelta(months=months)

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d %H:%M:%S')


def load_items_from_db(start_date: str, end_date: str, min_rows: int = 10000) -> list:
    """
    Load list of items with sufficient data for training.

    Args:
        start_date: Start of training period
        end_date: End of training period
        min_rows: Minimum number of rows required (default: 10000)

    Returns:
        List of dicts with item_id and name
    """
    conn = psycopg2.connect(**CONN_PARAMS)
    cur = conn.cursor()

    cur.execute("""
        SELECT p.item_id, i.name, COUNT(*) as row_count
        FROM price_data_5min p
        JOIN items i ON p.item_id = i.item_id
        WHERE p.timestamp >= %s AND p.timestamp <= %s
        GROUP BY p.item_id, i.name
        HAVING COUNT(*) > %s
        ORDER BY row_count DESC
    """, (start_date, end_date, min_rows))

    items = [{'item_id': r[0], 'name': r[1]} for r in cur.fetchall()]
    conn.close()

    return items


def load_price_data(item_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load price data for a single item within date range.

    Args:
        item_id: OSRS item ID
        start_date: Start of date range
        end_date: End of date range

    Returns:
        DataFrame with price data
    """
    conn = psycopg2.connect(**CONN_PARAMS)

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


def prepare_item_data(item: dict, feature_engine: FeatureEngine,
                      target_engine: DiscreteHourTargetEngine,
                      start_date: str, end_date: str) -> dict:
    """
    Prepare features and targets for a single item.

    Args:
        item: Dict with item_id and name
        feature_engine: FeatureEngine instance
        target_engine: DiscreteHourTargetEngine instance
        start_date: Start of training period
        end_date: End of training period

    Returns:
        Dict with prepared data or error status
    """
    item_id = item['item_id']
    item_name = item['name']

    try:
        # Load raw data
        df = load_price_data(item_id, start_date, end_date)
        if len(df) < 1000:
            return {'item_id': item_id, 'status': 'insufficient_data'}

        # Compute features
        df_features = feature_engine.compute_features(df)

        # Compute targets
        df_targets = target_engine.compute_targets(df_features)

        # Get feature columns
        feature_cols = [c for c in df_features.columns if any(c.startswith(p) for p in
                        ['return_', 'volatility_', 'volume_', 'ma_', 'spread', 'rsi_', 'hour', 'dayofweek',
                         'high_', 'low_', 'momentum_', 'log_'])]

        # Get target columns
        target_cols = [c for c in df_targets.columns if c.startswith('roundtrip_')]

        # Combine and clean
        all_cols = feature_cols + target_cols
        df_clean = df_targets[all_cols].dropna()

        if len(df_clean) < 1000:
            return {'item_id': item_id, 'status': 'insufficient_clean_data'}

        return {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'success',
            'data': df_clean,
            'feature_cols': feature_cols,
            'target_cols': target_cols,
            'rows': len(df_clean)
        }

    except Exception as e:
        return {'item_id': item_id, 'status': 'error', 'error': str(e)}


def prepare_and_upload_item(item: dict, feature_engine: FeatureEngine,
                            target_engine: DiscreteHourTargetEngine,
                            start_date: str, end_date: str,
                            bucket, run_id: str) -> dict:
    """
    Prepare features and targets for a single item, then upload immediately.

    This streaming approach frees memory after each item, preventing OOM crashes.

    Args:
        item: Dict with item_id and name
        feature_engine: FeatureEngine instance
        target_engine: DiscreteHourTargetEngine instance
        start_date: Start of training period
        end_date: End of training period
        bucket: GCS bucket object
        run_id: Run identifier

    Returns:
        Dict with metadata only (no data - already uploaded)
    """
    result = prepare_item_data(item, feature_engine, target_engine, start_date, end_date)

    if result['status'] != 'success':
        return result

    # Upload immediately and free memory
    df = result.pop('data')  # Remove from result dict

    # Convert to parquet bytes and upload
    table = pa.Table.from_pandas(df)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')

    blob = bucket.blob(f'runs/{run_id}/data/{result["item_id"]}.parquet')
    blob.upload_from_string(buffer.getvalue().to_pybytes())

    # Explicitly free memory
    del df
    del table
    del buffer

    return result


def upload_to_gcs(bucket_name: str, item_results: list, run_id: str,
                  start_date: str, end_date: str):
    """
    Upload prepared data to GCS.

    Args:
        bucket_name: GCS bucket name
        item_results: List of prepared item data
        run_id: Unique identifier for this training run
        start_date: Training data start date
        end_date: Training data end date
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Upload items config
    items_config = []
    for result in item_results:
        if result['status'] == 'success':
            items_config.append({
                'item_id': result['item_id'],
                'item_name': result['item_name'],
                'feature_cols': result['feature_cols'],
                'target_cols': result['target_cols'],
                'rows': result['rows']
            })

    config_blob = bucket.blob(f'runs/{run_id}/config.json')
    config_blob.upload_from_string(json.dumps({
        'run_id': run_id,
        'created_at': datetime.now().isoformat(),
        'start_date': start_date,
        'end_date': end_date,
        'items': items_config,
        'target_offsets': TARGET_OFFSETS,
        'discrete_hours': DISCRETE_HOURS
    }, indent=2))

    # Upload data for each item
    for result in item_results:
        if result['status'] != 'success':
            continue

        item_id = result['item_id']
        df = result['data']

        # Convert to parquet bytes
        table = pa.Table.from_pandas(df)
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer, compression='snappy')

        # Upload
        blob = bucket.blob(f'runs/{run_id}/data/{item_id}.parquet')
        blob.upload_from_string(buffer.getvalue().to_pybytes())

    print(f"Uploaded {len(items_config)} items to gs://{bucket_name}/runs/{run_id}/")
    return len(items_config)


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for GCP')
    parser.add_argument('--bucket', type=str, required=True,
                        help='GCS bucket name')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for data prep')
    parser.add_argument('--months', type=int, default=DEFAULT_TRAINING_MONTHS,
                        help=f'Months of historical data to use (default: {DEFAULT_TRAINING_MONTHS})')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run ID (default: YYYYMMDD_HHMMSS)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Prepare data but do not upload')
    parser.add_argument('--streaming', action='store_true', default=True,
                        help='Use streaming uploads (memory efficient, default: True)')
    parser.add_argument('--no-streaming', action='store_false', dest='streaming',
                        help='Use batch uploads (faster but uses more memory)')
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Preparing training data (run_id: {run_id})")
    print("=" * 60)

    start_time = time.time()

    # Get dynamic date range
    print(f"Computing date range ({args.months} months from latest data)...")
    start_date, end_date = get_training_date_range(months=args.months)
    print(f"  Start: {start_date}")
    print(f"  End:   {end_date}")

    # Load items
    print("\nLoading items list...")
    items = load_items_from_db(start_date, end_date)
    print(f"Found {len(items)} items with sufficient data")

    # Initialize engines
    feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
    target_config = DiscreteHourConfig(
        offsets=TARGET_OFFSETS,
        discrete_hours=DISCRETE_HOURS
    )
    target_engine = DiscreteHourTargetEngine(target_config)

    # Initialize GCS client for streaming mode
    bucket = None
    if not args.dry_run and args.streaming:
        storage_client = storage.Client()
        bucket = storage_client.bucket(args.bucket)
        print(f"\nStreaming mode: uploading to gs://{args.bucket}/ as items complete")

    # Process items
    print(f"\nPreparing data with {args.workers} workers...")
    results = []

    if args.streaming and not args.dry_run:
        # STREAMING MODE: Process and upload items one at a time to conserve memory
        # Uses sequential processing to avoid memory buildup from parallel futures
        for i, item in enumerate(items):
            result = prepare_and_upload_item(
                item, feature_engine, target_engine,
                start_date, end_date, bucket, run_id
            )
            results.append(result)

            status = result['status']
            if status == 'success':
                print(f"[{i+1}/{len(items)}] {item['name']}: {result['rows']} rows ✓")
            else:
                print(f"[{i+1}/{len(items)}] {item['name']}: {status}")

            # Periodic progress for long runs
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(items) - i - 1) / rate / 60
                print(f"    Progress: {i+1}/{len(items)} ({elapsed/60:.1f}m elapsed, ~{remaining:.0f}m remaining)")

    else:
        # BATCH MODE: Process in parallel (faster but uses more memory)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(prepare_item_data, item, feature_engine, target_engine,
                                start_date, end_date): item
                for item in items
            }

            for i, future in enumerate(as_completed(futures)):
                item = futures[future]
                result = future.result()
                results.append(result)

                status = result['status']
                if status == 'success':
                    print(f"[{i+1}/{len(items)}] {item['name']}: {result['rows']} rows")
                else:
                    print(f"[{i+1}/{len(items)}] {item['name']}: {status}")

    # Summary
    successful = [r for r in results if r['status'] == 'success']
    prep_time = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("Data preparation complete")
    print(f"  Items: {len(successful)}/{len(items)} successful")
    print(f"  Time: {prep_time:.1f}s ({prep_time/60:.1f}m)")

    if not args.dry_run:
        if args.streaming:
            # Data already uploaded in streaming mode, just write config
            print(f"\nWriting config to gs://{args.bucket}/runs/{run_id}/config.json...")
            items_config = []
            for result in results:
                if result['status'] == 'success':
                    items_config.append({
                        'item_id': result['item_id'],
                        'item_name': result['item_name'],
                        'feature_cols': result['feature_cols'],
                        'target_cols': result['target_cols'],
                        'rows': result['rows']
                    })

            config_blob = bucket.blob(f'runs/{run_id}/config.json')
            config_blob.upload_from_string(json.dumps({
                'run_id': run_id,
                'created_at': datetime.now().isoformat(),
                'start_date': start_date,
                'end_date': end_date,
                'items': items_config,
                'target_offsets': TARGET_OFFSETS,
                'discrete_hours': DISCRETE_HOURS
            }, indent=2))
            uploaded = len(items_config)
        else:
            # Batch upload in batch mode
            print(f"\nUploading to gs://{args.bucket}/runs/{run_id}/...")
            upload_start = time.time()
            uploaded = upload_to_gcs(args.bucket, results, run_id, start_date, end_date)
            upload_time = time.time() - upload_start
            print(f"  Uploaded {uploaded} items in {upload_time:.1f}s")

        total_time = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Run ID: {run_id}")
        print("\nTo train models, run:")
        print("  gcloud run jobs execute gept-daily-train --region us-central1 \\")
        print(f"    --tasks {uploaded} --set-env-vars RUN_ID={run_id},GCS_BUCKET={args.bucket}")
    else:
        print("\n[DRY RUN] Skipping upload")
        print("\nDate range that would be used:")
        print(f"  Start: {start_date}")
        print(f"  End:   {end_date}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
