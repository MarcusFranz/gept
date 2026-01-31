#!/usr/bin/env python3
"""
Inference Runner - 5-minute refresh cycle

Generates predictions for all items and writes to TimescaleDB.
Uses optimized inference order:
  1. Most volatile items first (fresher data for fast-moving items)
  2. Short-term predictions first (hours 1-4 before hours 17-24)

Model Type:
  - Multi-target: CatBoost MultiLogloss models with 108 targets per item

Optimizations:
  - Single forward pass for all 108 targets (much faster)
  - Connection pooling for DB connections
  - COPY protocol for 2.7x faster bulk writes

Run via cron every 5 minutes:
    */5 * * * * cd /path/to/GePT_Model && python run_inference.py >> logs/inference.log 2>&1
"""

import sys
import os
import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

# Import probability thresholds from centralized config (see calibration_analysis.md)
from inference_config import MAX_PROBABILITY, MIN_PROBABILITY
from db_utils import validate_db_config


def main():
    # Fail fast if required database env vars are missing
    validate_db_config()

    parser = argparse.ArgumentParser(description='Run inference cycle')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Directory containing trained models (auto-detects latest if not specified)')
    parser.add_argument('--hours', type=int, default=72,
                        help='Hours of price history to load')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate predictions but do not save to DB')
    parser.add_argument('--no-priority', action='store_true',
                        help='Disable prioritized inference (batch all at once)')
    parser.add_argument('--staging', action='store_true',
                        help='Write to predictions_staging table instead of predictions (safe testing)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Local Parquet cache directory (loads from cache instead of DB)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers for feature computation (default: 8)')
    parser.add_argument('--predict-workers', type=int, default=16,
                        help='Number of parallel workers for predictions (default: 16)')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"INFERENCE CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    start = time.time()

    # Multi-target predictor (production)
    from batch_predictor_multitarget import MultiTargetBatchPredictor as BatchPredictor
    print("Using multi-target predictor (108 targets per item)")
    models_dir = args.models_dir  # None = auto-detect latest
    if args.cache_dir:
        print(f"WARNING: --cache-dir is not supported for multi-target predictor, ignoring")

    # Initialize predictor (loads all models)
    print("Loading models...")
    load_start = time.time()
    try:
        predictor = BatchPredictor(models_dir=models_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("No multi-target models found. Run train_runpod_multitarget.py first.")
        return 1

    print(f"Loaded in {time.time()-load_start:.1f}s")

    if args.dry_run:
        # Dry run: generate but don't save
        print("\n[DRY RUN MODE]")
        price_data = predictor.load_recent_prices(hours=args.hours)
        print(f"Loaded {len(price_data)} items")

        features_cache = predictor.compute_all_features(price_data)
        print(f"Computed features for {len(features_cache)} items")

        predictions = predictor.predict_all(price_data) if hasattr(predictor, 'predict_all') else []
        print(f"Generated {len(predictions)} predictions (not saved)")

        # Show top opportunities
        if predictions:
            print(f"\nTop 5 opportunities by EV:")
            sorted_preds = sorted(predictions, key=lambda x: x['expected_value'], reverse=True)
            for p in sorted_preds[:5]:
                print(f"  {p['item_name']:20s} Hour {p['hour_offset']:2d} ({p['offset_pct']*100:.1f}%): "
                      f"P={p['fill_probability']:.1%}, EV={p['expected_value']*100:.2f}%")
    else:
        # Determine target table
        table_name = 'predictions_staging' if args.staging else 'predictions'
        if args.staging:
            print(f"\n[STAGING MODE] Writing to '{table_name}' table (production unaffected)")

        # Full inference cycle with DB writes
        # Note: --workers and --predict-workers flags are ignored by multi-target predictor
        try:
            results = predictor.run_inference_cycle(
                hours_history=args.hours,
                use_copy=True,
                table_name=table_name
            )
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            if args.staging:
                print("Make sure the staging table exists. Run: psql -f scripts/setup_predictions_staging.sql")
            else:
                print("Make sure TimescaleDB is set up. Run: psql -f scripts/setup_predictions_table.sql")
            return 1

    # Summary
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")

    if not args.dry_run:
        print(f"Items: {results.get('items', 'N/A')}")
        print(f"Predictions: {results.get('predictions', results.get('total_predictions', 'N/A'))}")
        print(f"Load time: {results.get('load_time', 0):.1f}s")
        print(f"Feature time: {results.get('feature_time', 0):.1f}s")
        print(f"Total time: {results.get('total_time', total_time):.1f}s")

        # Report clipped predictions (Issue #34: enforce probability caps)
        total_clipped = results.get('total_clipped', 0)
        if total_clipped > 0:
            print(f"\nClipped predictions: {total_clipped} (capped to [{MIN_PROBABILITY}, {MAX_PROBABILITY}])")

        # Report missing/invalid features (Issue #78: data quality visibility)
        missing = results.get('total_missing_features', 0)
        invalid = results.get('total_invalid_features', 0)
        if missing > 0 or invalid > 0:
            print(f"Feature quality: {missing} missing, {invalid} invalid (replaced with 0.0)")

        if 'tiers' in results:
            print(f"\nTier breakdown:")
            for tier in results['tiers']:
                clipped_str = f", clipped={tier.get('clipped', 0)}" if tier.get('clipped', 0) > 0 else ""
                print(f"  {tier['tier']:10s} (hours {tier['hours']:>5s}): "
                      f"{tier['predictions']:4d} predictions, "
                      f"infer={tier['predict_time']:.2f}s, save={tier['save_time']:.2f}s{clipped_str}")

    print(f"\nWall clock: {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
