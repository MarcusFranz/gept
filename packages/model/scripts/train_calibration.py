#!/usr/bin/env python3
"""
Train Calibration Layer for Multi-Target Models
================================================

Fits isotonic regression calibrators per hour-group using holdout validation data.
Saves calibration.json alongside each model in the models directory.

Usage:
    # Train calibration for latest model run
    python scripts/train_calibration.py

    # Train for a specific model run
    python scripts/train_calibration.py --models-dir models/20260111_142024

    # Train using a specific date range for validation data
    python scripts/train_calibration.py --start-date 2025-12-01 --end-date 2026-01-01
"""

import os
import sys
import json
import argparse
import re
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from catboost import CatBoostClassifier
from feature_engine import FeatureEngine, Granularity
from calibration import (
    CalibrationConfig,
    CalibrationManager,
    IsotonicCalibrator,
    CALIBRATED_MAX,
    CALIBRATED_MIN,
    compute_brier_score
)
from db_utils import get_db_connection


def find_latest_model_run(models_base: str = 'models') -> str:
    """Find the latest model run directory by timestamp."""
    models_path = Path(models_base)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_base}")

    run_pattern = re.compile(r'^\d{8}_\d{6}$')
    runs = [
        d for d in models_path.iterdir()
        if d.is_dir() and run_pattern.match(d.name)
    ]

    if not runs:
        raise FileNotFoundError(f"No model runs found in {models_base}")

    latest = sorted(runs, key=lambda x: x.name)[-1]
    return str(latest)


def parse_target_name(target_name: str) -> Tuple[int, float]:
    """Parse target name to extract hour and offset."""
    match = re.match(r'seq_(\d+)h_(\d+\.?\d*)pct', target_name)
    if match:
        hour = int(match.group(1))
        offset = float(match.group(2)) / 100
        return hour, offset
    return None, None


def load_validation_data(
    item_id: int,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Load price data for validation period."""
    with get_db_connection() as conn:
        query = """
            SELECT item_id, timestamp, avg_high_price, avg_low_price,
                   high_price_volume, low_price_volume
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp
        """
        df = pd.read_sql(query, conn, params=[item_id, start_date, end_date])
    return df


def compute_sequential_fill_targets(
    df: pd.DataFrame,
    hours: List[int],
    offsets: List[float]
) -> pd.DataFrame:
    """
    Compute sequential fill targets: buy must fill before sell.

    This matches the training target computation from train_runpod_multitarget.py.
    """
    df = df.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Number of 5-min periods per hour
    periods_per_hour = 12

    for hour in hours:
        lookforward = hour * periods_per_hour

        for offset in offsets:
            col_name = f'seq_{hour}h_{offset*100:.2f}'.rstrip('0').rstrip('.') + 'pct'

            targets = []
            for i in range(len(df)):
                if i + lookforward >= len(df):
                    targets.append(np.nan)
                    continue

                future_slice = df.iloc[i:i + lookforward + 1]
                current_low = df.iloc[i]['avg_low_price']
                current_high = df.iloc[i]['avg_high_price']

                if pd.isna(current_low) or pd.isna(current_high):
                    targets.append(np.nan)
                    continue

                buy_target = current_low * (1 - offset)
                sell_target = current_high * (1 + offset)

                # Find buy fill time (when low price drops to buy target)
                buy_filled = False
                buy_fill_idx = None

                for j, row in enumerate(future_slice.itertuples()):
                    if pd.notna(row.avg_low_price) and row.avg_low_price <= buy_target:
                        buy_filled = True
                        buy_fill_idx = j
                        break

                if not buy_filled:
                    targets.append(0)
                    continue

                # Find sell fill after buy (when high price rises to sell target)
                sell_slice = future_slice.iloc[buy_fill_idx:]
                sell_filled = False

                for row in sell_slice.itertuples():
                    if pd.notna(row.avg_high_price) and row.avg_high_price >= sell_target:
                        sell_filled = True
                        break

                targets.append(1 if sell_filled else 0)

            df[col_name] = targets

    return df


def generate_predictions_for_item(
    model: CatBoostClassifier,
    meta: Dict,
    features_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate raw predictions for all targets on validation data."""
    feature_cols = meta.get('feature_cols', [])
    target_names = meta.get('target_names', [])

    if not feature_cols or not target_names:
        return pd.DataFrame()

    # Build feature matrix
    X = []
    valid_indices = []

    for idx in range(len(features_df)):
        row = features_df.iloc[idx]
        feature_vec = []
        valid = True

        for col in feature_cols:
            if col in row.index:
                val = row[col]
                if pd.isna(val) or np.isinf(val):
                    feature_vec.append(0.0)
                else:
                    feature_vec.append(float(val))
            else:
                feature_vec.append(0.0)

        X.append(feature_vec)
        valid_indices.append(idx)

    if not X:
        return pd.DataFrame()

    X = np.array(X)

    # Get predictions (single forward pass for all targets)
    proba = model.predict_proba(X)

    # Extract positive class probabilities
    # MultiLogloss format: [neg0, pos0, neg1, pos1, ...]
    n_targets = len(target_names)

    if proba.shape[1] == 2 * n_targets:
        predictions = {
            target_names[i]: proba[:, 2 * i + 1]
            for i in range(n_targets)
        }
    else:
        predictions = {
            target_names[i]: proba[:, i]
            for i in range(n_targets)
        }

    # Build result DataFrame
    result = pd.DataFrame(predictions, index=valid_indices)
    result['timestamp'] = features_df.iloc[valid_indices]['timestamp'].values

    return result


def fit_calibration_for_item(
    predictions_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    item_id: int
) -> CalibrationManager:
    """
    Fit isotonic calibrators for all hour groups.

    Groups predictions by hour bucket and fits one calibrator per group.
    """
    calibrators = {}
    all_hours = CalibrationConfig.get_all_groups()

    # Collect predictions and actuals by hour group
    group_data: Dict[str, Tuple[List[float], List[int]]] = {
        group: ([], []) for group in all_hours
    }

    # Iterate through all target columns
    for col in predictions_df.columns:
        if col == 'timestamp':
            continue

        hour, offset = parse_target_name(col)
        if hour is None:
            continue

        group_name = CalibrationConfig.get_hour_group(hour)

        # Find matching target column
        target_col = col
        if target_col not in targets_df.columns:
            continue

        # Align predictions and actuals
        for idx in predictions_df.index:
            if idx >= len(targets_df):
                continue

            pred = predictions_df.loc[idx, col]
            actual = targets_df.iloc[idx].get(target_col)

            if pd.notna(pred) and pd.notna(actual):
                group_data[group_name][0].append(float(pred))
                group_data[group_name][1].append(int(actual))

    # Fit calibrator for each group
    global_brier_before = []
    global_brier_after = []
    total_samples = 0

    for group_name in all_hours:
        preds, actuals = group_data[group_name]

        if len(preds) < 100:  # Need minimum samples for reliable fit
            print(f"    {group_name}: skipped (only {len(preds)} samples)")
            continue

        preds = np.array(preds)
        actuals = np.array(actuals)

        calibrator = IsotonicCalibrator.fit(
            preds,
            actuals,
            y_min=CALIBRATED_MIN,
            y_max=CALIBRATED_MAX
        )

        if calibrator.is_fitted:
            calibrators[group_name] = calibrator
            total_samples += calibrator.n_samples

            if calibrator.brier_before is not None:
                global_brier_before.append(calibrator.brier_before)
                global_brier_after.append(calibrator.brier_after)

            improvement = 0
            if calibrator.brier_before and calibrator.brier_before > 0:
                improvement = (1 - calibrator.brier_after / calibrator.brier_before) * 100

            print(f"    {group_name}: {len(preds)} samples, "
                  f"Brier {calibrator.brier_before:.4f} -> {calibrator.brier_after:.4f} "
                  f"({improvement:+.1f}%)")

    # Compute global metrics
    global_metrics = {}
    if global_brier_before:
        avg_before = np.mean(global_brier_before)
        avg_after = np.mean(global_brier_after)
        global_metrics = {
            'avg_brier_before': float(avg_before),
            'avg_brier_after': float(avg_after),
            'brier_improvement_pct': float((1 - avg_after / avg_before) * 100) if avg_before > 0 else 0,
            'total_samples': total_samples,
            'n_groups_fitted': len(calibrators)
        }

    return CalibrationManager(
        calibrators=calibrators,
        version='1.0',
        fitted_at=datetime.utcnow().isoformat() + 'Z',
        item_id=item_id,
        global_metrics=global_metrics
    )


def train_calibration(
    models_dir: str,
    start_date: str,
    end_date: str,
    min_data_rows: int = 1000
) -> Dict:
    """
    Train calibration for all models in a run.

    Args:
        models_dir: Path to model run directory
        start_date: Start date for validation data
        end_date: End date for validation data
        min_data_rows: Minimum rows required for calibration

    Returns:
        Summary statistics
    """
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)

    # Define target hours and offsets
    hours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 20, 24, 32, 40, 48]
    offsets = [0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025]

    stats = {
        'models_dir': str(models_dir),
        'validation_period': f'{start_date} to {end_date}',
        'items_processed': 0,
        'items_skipped': 0,
        'avg_brier_improvement': 0,
        'items': []
    }

    brier_improvements = []

    print(f"\nTraining calibration for models in: {models_dir}")
    print(f"Validation period: {start_date} to {end_date}")
    print("-" * 60)

    for item_dir in sorted(models_path.iterdir()):
        if not item_dir.is_dir():
            continue

        try:
            item_id = int(item_dir.name)
        except ValueError:
            continue

        model_path = item_dir / 'model.cbm'
        meta_path = item_dir / 'meta.json'

        if not model_path.exists() or not meta_path.exists():
            continue

        print(f"\nItem {item_id}:")

        # Load model and metadata
        with open(meta_path) as f:
            meta = json.load(f)

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        # Load validation data
        print("  Loading validation data...")
        df = load_validation_data(item_id, start_date, end_date)

        if len(df) < min_data_rows:
            print(f"  Skipped: only {len(df)} rows (need {min_data_rows})")
            stats['items_skipped'] += 1
            continue

        # Compute features
        print(f"  Computing features ({len(df)} rows)...")
        features_df = feature_engine.compute_features(df.copy())

        # Compute targets
        print("  Computing sequential fill targets...")
        targets_df = compute_sequential_fill_targets(df, hours, offsets)

        # Generate predictions
        print("  Generating predictions...")
        predictions_df = generate_predictions_for_item(model, meta, features_df)

        if predictions_df.empty:
            print("  Skipped: no valid predictions")
            stats['items_skipped'] += 1
            continue

        # Fit calibration
        print("  Fitting calibrators:")
        calibration_manager = fit_calibration_for_item(
            predictions_df,
            targets_df,
            item_id
        )

        # Save calibration
        calib_path = item_dir / 'calibration.json'
        calibration_manager.save(str(calib_path))
        print(f"  Saved to: {calib_path}")

        # Track statistics
        global_metrics = calibration_manager.global_metrics
        if global_metrics.get('brier_improvement_pct'):
            brier_improvements.append(global_metrics['brier_improvement_pct'])

        stats['items_processed'] += 1
        stats['items'].append({
            'item_id': item_id,
            'n_groups_fitted': global_metrics.get('n_groups_fitted', 0),
            'total_samples': global_metrics.get('total_samples', 0),
            'brier_improvement_pct': global_metrics.get('brier_improvement_pct', 0)
        })

    # Compute aggregate stats
    if brier_improvements:
        stats['avg_brier_improvement'] = float(np.mean(brier_improvements))

    return stats


def main():
    parser = argparse.ArgumentParser(description='Train calibration for multi-target models')
    parser.add_argument('--models-dir', type=str, default=None,
                        help='Path to models directory (default: latest run)')
    parser.add_argument('--start-date', type=str, default='2025-12-01',
                        help='Start date for validation data')
    parser.add_argument('--end-date', type=str, default='2026-01-01',
                        help='End date for validation data')
    parser.add_argument('--min-rows', type=int, default=1000,
                        help='Minimum data rows required for calibration')

    args = parser.parse_args()

    # Find models directory
    if args.models_dir is None:
        args.models_dir = find_latest_model_run()
        print(f"Using latest model run: {args.models_dir}")

    # Run calibration training
    stats = train_calibration(
        models_dir=args.models_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        min_data_rows=args.min_rows
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Items processed: {stats['items_processed']}")
    print(f"Items skipped: {stats['items_skipped']}")
    print(f"Average Brier improvement: {stats['avg_brier_improvement']:.1f}%")

    # Save summary
    summary_path = Path(args.models_dir) / 'calibration_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()
