#!/usr/bin/env python3
"""
Overnight Evaluation Script
Run inference on all models and compare predictions to actual price movements.
"""

import os
import sys
import json
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

warnings.filterwarnings('ignore')

MODELS_DIR = Path(__file__).parent / 'models_downloaded'
RESULTS_FILE = Path(__file__).parent / 'overnight_evaluation_results.json'

CONN_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'database': 'osrs_data',
    'user': 'osrs_user',
    'password': os.environ['DB_PASS']
}

sys.path.insert(0, str(Path(__file__).parent / 'src'))


class ModelEvaluator:
    """Evaluates model predictions against actual price data."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.meta = {}
        self.items = []
        self._load_models()

    def _load_models(self):
        """Load all models into memory."""
        print(f"Loading models from {self.models_dir}...")

        for item_dir in sorted(self.models_dir.iterdir()):
            if not item_dir.is_dir():
                continue

            try:
                item_id = int(item_dir.name)
            except ValueError:
                continue

            registry_path = item_dir / 'registry.json'
            if not registry_path.exists():
                continue

            with open(registry_path) as f:
                registry = json.load(f)

            if not registry.get('models'):
                continue

            item_name = registry.get('item_name', f'Item-{item_id}')
            self.items.append({'item_id': item_id, 'item_name': item_name})
            self.models[item_id] = {}
            self.scalers[item_id] = {}
            self.meta[item_id] = {}

            for target_name in registry.get('models', {}):
                model_path = item_dir / f'{target_name}_model.pkl'
                scaler_path = item_dir / f'{target_name}_scaler.pkl'
                meta_path = item_dir / f'{target_name}_meta.json'

                if model_path.exists() and scaler_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[item_id][target_name] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[item_id][target_name] = pickle.load(f)

                    if meta_path.exists():
                        with open(meta_path) as f:
                            self.meta[item_id][target_name] = json.load(f)

        total_models = sum(len(m) for m in self.models.values())
        print(f"Loaded {len(self.items)} items with {total_models} models")

    def load_price_data(self, hours_back: int = 96) -> Dict[int, pd.DataFrame]:
        """Load historical price data for evaluation."""
        conn = psycopg2.connect(**CONN_PARAMS)

        item_ids = [item['item_id'] for item in self.items]

        query = """
            SELECT item_id, timestamp, avg_high_price as high, avg_low_price as low,
                   high_price_volume, low_price_volume
            FROM price_data_5min
            WHERE item_id = ANY(%s)
              AND timestamp >= NOW() - make_interval(hours => %s)
            ORDER BY item_id, timestamp
        """

        df = pd.read_sql(query, conn, params=[item_ids, hours_back])
        conn.close()

        price_data = {}
        for item_id in item_ids:
            item_df = df[df['item_id'] == item_id].copy()
            if len(item_df) > 100:
                price_data[item_id] = item_df.reset_index(drop=True)

        return price_data

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a single item."""
        df = df.copy()

        # Basic price columns
        df['mid'] = (df['high'] + df['low']) / 2
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['mid']

        # Returns
        for period in [1, 3, 6, 12, 24, 48]:
            df[f'return_{period}'] = df['mid'].pct_change(period)

        # Volatility
        for window in [12, 24, 48, 96]:
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()

        # Moving averages
        for window in [6, 12, 24, 48]:
            df[f'ma_{window}'] = df['mid'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['mid'] / df[f'ma_{window}']

        # RSI
        delta = df['mid'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Volume features
        df['volume_total'] = df['high_price_volume'].fillna(0) + df['low_price_volume'].fillna(0)
        for window in [12, 24, 48]:
            df[f'volume_ma_{window}'] = df['volume_total'].rolling(window).mean()

        # High/Low momentum
        for period in [12, 24, 48]:
            df[f'high_momentum_{period}'] = df['high'].pct_change(period)
            df[f'low_momentum_{period}'] = df['low'].pct_change(period)

        # Fill NaN
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        return df

    def check_fill(self, df: pd.DataFrame, start_idx: int, offset: float,
                   hour: int, direction: str = 'buy') -> Tuple[bool, Optional[int]]:
        """
        Check if a limit order would have filled within the hour window.

        For buy orders: Check if low price went below buy_target
        For sell orders: Check if high price went above sell_target
        """
        periods_per_hour = 12  # 5-min intervals

        # Calculate target price
        if direction == 'buy':
            target_price = df.iloc[start_idx]['low'] * (1 - offset)
            look_end = min(start_idx + hour * periods_per_hour, len(df))
            look_start = start_idx + (hour - 1) * periods_per_hour if hour > 1 else start_idx

            # Check if any low price hit our target
            for i in range(look_start, look_end):
                if df.iloc[i]['low'] <= target_price:
                    return True, i
        else:
            target_price = df.iloc[start_idx]['high'] * (1 + offset)
            look_end = min(start_idx + hour * periods_per_hour, len(df))
            look_start = start_idx + (hour - 1) * periods_per_hour if hour > 1 else start_idx

            for i in range(look_start, look_end):
                if df.iloc[i]['high'] >= target_price:
                    return True, i

        return False, None

    def evaluate_item(self, item_id: int, df: pd.DataFrame) -> Dict:
        """Evaluate all models for a single item."""
        if item_id not in self.models:
            return {}

        results = {
            'item_id': item_id,
            'item_name': next((i['item_name'] for i in self.items if i['item_id'] == item_id), ''),
            'models': {}
        }

        # Compute features
        features_df = self.compute_features(df)

        # Get feature columns from first model
        first_target = next(iter(self.meta[item_id].keys()), None)
        if not first_target:
            return results

        feature_cols = self.meta[item_id][first_target].get('feature_columns', [])

        # Evaluate each model at multiple time points
        for target_name, model in self.models[item_id].items():
            scaler = self.scalers[item_id].get(target_name)
            meta = self.meta[item_id].get(target_name, {})

            if scaler is None:
                continue

            # Parse target
            parts = target_name.replace('roundtrip_', '').split('_hour')
            if len(parts) != 2:
                continue

            offset_str = parts[0]
            hour = int(parts[1])

            try:
                if '_' in offset_str:
                    offset = float(offset_str.replace('pct', '').replace('_', '.')) / 100
                else:
                    offset = float(offset_str.replace('pct', '')) / 100
            except:
                offset = 0.02

            # Evaluate at multiple historical points (leave room for checking outcome)
            predictions = []
            actuals = []

            # Start evaluation from 24 hours ago to leave room for checking outcomes
            eval_points = range(len(features_df) - hour * 12 - 24, len(features_df) - hour * 12, 12)

            for idx in eval_points:
                if idx < 100:  # Need enough history for features
                    continue

                row = features_df.iloc[idx]

                # Build feature vector
                X = []
                for col in feature_cols:
                    if col in row.index:
                        val = row[col]
                        X.append(0.0 if pd.isna(val) or np.isinf(val) else float(val))
                    else:
                        X.append(0.0)

                X = np.array(X).reshape(1, -1)

                try:
                    X_scaled = scaler.transform(X)
                    prob = model.predict_proba(X_scaled)[0, 1]
                except:
                    continue

                # Check actual outcome
                filled, _ = self.check_fill(df, idx, offset, hour, 'buy')

                predictions.append(prob)
                actuals.append(1 if filled else 0)

            if len(predictions) < 10:
                continue

            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)

            # Calibration: binned predicted vs actual
            bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
            calibration = {}
            for i in range(len(bins) - 1):
                mask = (predictions >= bins[i]) & (predictions < bins[i+1])
                if mask.sum() > 0:
                    calibration[f'{bins[i]:.0%}-{bins[i+1]:.0%}'] = {
                        'count': int(mask.sum()),
                        'predicted_avg': float(predictions[mask].mean()),
                        'actual_rate': float(actuals[mask].mean())
                    }

            results['models'][target_name] = {
                'offset': offset,
                'hour': hour,
                'n_predictions': len(predictions),
                'mean_predicted': float(predictions.mean()),
                'actual_fill_rate': float(actuals.mean()),
                'calibration_error': float(abs(predictions.mean() - actuals.mean())),
                'calibration': calibration,
                'training_auc': meta.get('metrics', {}).get('auc', 0)
            }

        return results

    def run_evaluation(self) -> Dict:
        """Run full evaluation across all items."""
        print("\nLoading price data...")
        start = time.time()
        price_data = self.load_price_data(hours_back=96)
        print(f"Loaded {len(price_data)} items in {time.time()-start:.1f}s")

        print("\nEvaluating models...")
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'items_evaluated': 0,
            'models_evaluated': 0,
            'items': []
        }

        for i, item in enumerate(self.items):
            item_id = item['item_id']
            if item_id not in price_data:
                continue

            result = self.evaluate_item(item_id, price_data[item_id])
            if result.get('models'):
                all_results['items'].append(result)
                all_results['items_evaluated'] += 1
                all_results['models_evaluated'] += len(result['models'])

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(self.items)} items...")

        # Aggregate statistics
        all_predictions = []
        all_actuals = []
        calibration_errors = []

        for item in all_results['items']:
            for target, metrics in item['models'].items():
                calibration_errors.append(metrics['calibration_error'])

        all_results['summary'] = {
            'total_items': all_results['items_evaluated'],
            'total_models': all_results['models_evaluated'],
            'avg_calibration_error': float(np.mean(calibration_errors)) if calibration_errors else 0,
            'median_calibration_error': float(np.median(calibration_errors)) if calibration_errors else 0
        }

        return all_results


def generate_predictions(evaluator: ModelEvaluator, price_data: Dict[int, pd.DataFrame]) -> List[Dict]:
    """Generate current predictions for all models."""
    predictions = []
    prediction_time = datetime.now()

    for item in evaluator.items:
        item_id = item['item_id']
        if item_id not in price_data:
            continue

        if item_id not in evaluator.models:
            continue

        df = price_data[item_id]
        features_df = evaluator.compute_features(df)

        first_target = next(iter(evaluator.meta[item_id].keys()), None)
        if not first_target:
            continue

        feature_cols = evaluator.meta[item_id][first_target].get('feature_columns', [])
        latest = features_df.iloc[-1]

        X = []
        for col in feature_cols:
            if col in latest.index:
                val = latest[col]
                X.append(0.0 if pd.isna(val) or np.isinf(val) else float(val))
            else:
                X.append(0.0)
        X = np.array(X).reshape(1, -1)

        current_high = float(latest.get('high', 0))
        current_low = float(latest.get('low', 0))

        for target_name, model in evaluator.models[item_id].items():
            scaler = evaluator.scalers[item_id].get(target_name)
            if scaler is None:
                continue

            parts = target_name.replace('roundtrip_', '').split('_hour')
            if len(parts) != 2:
                continue

            offset_str = parts[0]
            hour = int(parts[1])

            try:
                if '_' in offset_str:
                    offset = float(offset_str.replace('pct', '').replace('_', '.')) / 100
                else:
                    offset = float(offset_str.replace('pct', '')) / 100
            except:
                offset = 0.02

            try:
                X_scaled = scaler.transform(X)
                prob = float(model.predict_proba(X_scaled)[0, 1])
            except:
                continue

            # Calculate expected value
            ev = prob * offset - (1 - prob) * offset  # Simplified EV

            predictions.append({
                'time': prediction_time.isoformat(),
                'item_id': item_id,
                'item_name': item['item_name'],
                'target': target_name,
                'hour': hour,
                'offset': offset,
                'probability': prob,
                'expected_value': ev,
                'buy_price': current_low * (1 - offset),
                'sell_price': current_high * (1 + offset),
                'current_high': current_high,
                'current_low': current_low
            })

    return predictions


def main():
    print("=" * 70)
    print("OVERNIGHT MODEL EVALUATION")
    print("=" * 70)

    evaluator = ModelEvaluator()

    # Run full evaluation
    results = evaluator.run_evaluation()

    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Items Evaluated:          {results['summary']['total_items']}")
    print(f"Models Evaluated:         {results['summary']['total_models']}")
    print(f"Avg Calibration Error:    {results['summary']['avg_calibration_error']:.4f}")
    print(f"Median Calibration Error: {results['summary']['median_calibration_error']:.4f}")

    # Show top calibrated items
    print("\nTop 10 Best Calibrated Items:")
    items_by_cal = sorted(results['items'],
                          key=lambda x: np.mean([m['calibration_error'] for m in x['models'].values()]))
    for item in items_by_cal[:10]:
        avg_cal = np.mean([m['calibration_error'] for m in item['models'].values()])
        print(f"  {item['item_name']} (ID:{item['item_id']}): {avg_cal:.4f}")

    # Generate current predictions
    print("\n" + "=" * 70)
    print("GENERATING CURRENT PREDICTIONS")
    print("=" * 70)

    print("Loading recent price data...")
    price_data = evaluator.load_price_data(hours_back=72)

    predictions = generate_predictions(evaluator, price_data)
    print(f"Generated {len(predictions)} predictions")

    # Save predictions
    predictions_file = Path(__file__).parent / 'current_predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {predictions_file}")

    # Show top predictions by probability
    print("\nTop 10 Highest Probability Predictions:")
    sorted_by_prob = sorted(predictions, key=lambda x: x['probability'], reverse=True)
    for p in sorted_by_prob[:10]:
        print(f"  {p['item_name']} Hour {p['hour']} ({p['offset']*100:.1f}%): "
              f"P={p['probability']:.2%}, EV={p['expected_value']*100:.2f}%")

    return results


if __name__ == '__main__':
    main()
