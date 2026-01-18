"""
Backtest Validator for GE Flipping Predictions

Validates model predictions against actual historical price movements.
This is the critical test: Do model probabilities match actual fill rates?

Key Questions:
1. When model says 20% probability, do ~20% of those actually fill?
2. Are high-probability predictions (>10%) actually reliable?
3. How many actionable predictions do we get per day?
"""

import os
import json
import logging
import joblib
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engine import FeatureEngine, Granularity
from target_engine import DiscreteHourTargetEngine, compute_expected_value
from db_utils import CONN_PARAMS


class BacktestValidator:
    """
    Validates model predictions against actual historical outcomes.

    Approach:
    1. Load historical price data (last 30-60 days)
    2. At each 5-min interval, generate predictions using the trained models
    3. Check if those predictions would have filled in the actual data
    4. Compare predicted probabilities to actual fill rates
    """

    def __init__(self, models_dir: str = 'models_downloaded'):
        self.models_dir = Path(models_dir)
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self.target_engine = DiscreteHourTargetEngine(granularity='5m')

        self.models = {}
        self.scalers = {}
        self.meta = {}
        self.items = []

        self._load_all_models()

    def _load_all_models(self):
        """Load all models into memory."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        for item_dir in self.models_dir.iterdir():
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

            item_name = registry.get('item_name', f'Item-{item_id}')

            # Check if item has valid models
            models = registry.get('models', {})
            valid_models = [k for k, v in models.items() if v.get('is_valid', False)]

            if not valid_models:
                continue

            self.items.append({'item_id': item_id, 'item_name': item_name})
            self.models[item_id] = {}
            self.scalers[item_id] = {}
            self.meta[item_id] = {}

            for target_name, model_info in models.items():
                model_path = item_dir / f'{target_name}_model.pkl'
                scaler_path = item_dir / f'{target_name}_scaler.pkl'
                meta_path = item_dir / f'{target_name}_meta.json'

                if model_path.exists() and scaler_path.exists():
                    self.models[item_id][target_name] = joblib.load(model_path)
                    self.scalers[item_id][target_name] = joblib.load(scaler_path)

                    if meta_path.exists():
                        with open(meta_path) as f:
                            self.meta[item_id][target_name] = json.load(f)

        print(f"Loaded {len(self.items)} items with valid models")
        total_models = sum(len(m) for m in self.models.values())
        print(f"Total models: {total_models}")

    def load_historical_data(self, item_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical price data for an item."""
        conn = psycopg2.connect(**CONN_PARAMS)

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
        conn.close()

        return df

    def compute_actual_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute actual fill outcomes for the data."""
        df = self.target_engine.compute_targets(df, copy=True)
        return df

    def predict_at_point(self, item_id: int, features_row: pd.Series) -> List[Dict]:
        """Generate all predictions for a single point in time."""
        predictions = []

        if item_id not in self.models:
            return predictions

        # Get feature columns from first model's metadata
        first_target = next(iter(self.meta[item_id].keys()), None)
        if not first_target:
            return predictions

        feature_cols = self.meta[item_id][first_target].get('feature_columns', [])

        # Build feature vector
        X = []
        for col in feature_cols:
            if col in features_row.index:
                val = features_row[col]
                if pd.isna(val) or np.isinf(val):
                    X.append(0.0)
                else:
                    X.append(float(val))
            else:
                X.append(0.0)

        X = np.array(X).reshape(1, -1)

        # Generate predictions for each model
        for target_name, model in self.models[item_id].items():
            scaler = self.scalers[item_id].get(target_name)
            meta = self.meta[item_id].get(target_name, {})

            if scaler is None:
                continue

            # Parse target name
            parts = target_name.replace('roundtrip_', '').split('_hour')
            if len(parts) != 2:
                continue

            offset_str = parts[0]
            hour = int(parts[1])

            # Convert offset
            try:
                if '_' in offset_str:
                    offset = float(offset_str.replace('pct', '').replace('_', '.')) / 100
                else:
                    offset = float(offset_str.replace('pct', '')) / 100
            except ValueError:
                offset = 0.02

            # Skip invalid models
            if not meta.get('metrics', {}).get('is_valid', False):
                continue

            # Predict
            try:
                X_scaled = scaler.transform(X)
                prob = model.predict_proba(X_scaled)[0, 1]
            except (ValueError, IndexError) as e:
                logger.debug(f"Prediction failed for {target_name}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected prediction error for {target_name}: {e}")
                continue

            predictions.append({
                'target_name': target_name,
                'hour': hour,
                'offset': offset,
                'probability': float(prob),
                'ev': compute_expected_value(prob, offset)
            })

        return predictions

    def backtest_item(self, item_id: int, start_date: str, end_date: str,
                      sample_every_n: int = 12) -> Dict:
        """
        Backtest predictions for a single item.

        Args:
            item_id: Item to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            sample_every_n: Sample every N rows to speed up (12 = hourly)

        Returns:
            Dict with backtest results
        """
        # Load data
        df = self.load_historical_data(item_id, start_date, end_date)

        if len(df) < 500:  # Need enough data
            return {'error': 'Insufficient data', 'rows': len(df)}

        # Compute features
        features_df = self.feature_engine.compute_features(df, copy=True)

        # Compute actual outcomes
        outcomes_df = self.compute_actual_outcomes(df)

        # Merge features and outcomes
        merged = features_df.copy()
        for col in outcomes_df.columns:
            if col.startswith('roundtrip_'):
                merged[col] = outcomes_df[col]

        # Sample points for backtesting (need 24h+ of future data)
        # Exclude last 24 hours * 12 periods = 288 periods
        valid_indices = range(self.feature_engine.min_history,
                              len(merged) - 288,
                              sample_every_n)

        results = []

        for idx in valid_indices:
            row = merged.iloc[idx]

            # Generate predictions
            predictions = self.predict_at_point(item_id, row)

            for pred in predictions:
                target_col = pred['target_name']

                # Get actual outcome
                if target_col in merged.columns:
                    actual = merged.iloc[idx][target_col]

                    if not pd.isna(actual):
                        results.append({
                            'timestamp': row.get('timestamp'),
                            'target': target_col,
                            'hour': pred['hour'],
                            'offset': pred['offset'],
                            'predicted_prob': pred['probability'],
                            'actual_fill': int(actual),
                            'ev': pred['ev']
                        })

        return {
            'item_id': item_id,
            'item_name': self._get_item_name(item_id),
            'total_predictions': len(results),
            'results': results
        }

    def _get_item_name(self, item_id: int) -> str:
        for item in self.items:
            if item['item_id'] == item_id:
                return item['item_name']
        return f'Item-{item_id}'

    def run_full_backtest(self, start_date: str, end_date: str,
                          n_items: int = 20, sample_every_n: int = 12) -> Dict:
        """
        Run backtest across multiple items.

        Args:
            start_date: Start date
            end_date: End date
            n_items: Number of items to backtest (random sample if < total)
            sample_every_n: Sample interval

        Returns:
            Full backtest results
        """
        import random

        # Select items (take random sample if needed)
        items_to_test = (self.items[:n_items] if n_items >= len(self.items)
                         else random.sample(self.items, n_items))

        all_results = []
        item_stats = []

        print(f"Running backtest on {len(items_to_test)} items...")
        print(f"Date range: {start_date} to {end_date}")

        for i, item in enumerate(items_to_test):
            item_id = item['item_id']
            print(f"  [{i+1}/{len(items_to_test)}] {item['item_name']}...", end=' ')

            try:
                result = self.backtest_item(item_id, start_date, end_date, sample_every_n)

                if 'error' in result:
                    print(f"Error: {result['error']}")
                    continue

                all_results.extend(result['results'])
                item_stats.append({
                    'item_id': item_id,
                    'item_name': item['item_name'],
                    'predictions': result['total_predictions']
                })

                print(f"{result['total_predictions']} predictions")
            except Exception as e:
                print(f"Error: {e}")
                continue

        return {
            'start_date': start_date,
            'end_date': end_date,
            'items_tested': len(item_stats),
            'total_predictions': len(all_results),
            'item_stats': item_stats,
            'results': all_results
        }

    def compute_calibration(self, results: List[Dict],
                            buckets: List[float] = None) -> Dict:
        """
        Compute calibration metrics from backtest results.

        Args:
            results: List of prediction results
            buckets: Probability bucket boundaries

        Returns:
            Calibration analysis
        """
        if buckets is None:
            buckets = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]

        df = pd.DataFrame(results)

        if len(df) == 0:
            return {'error': 'No results to analyze'}

        # Overall stats
        overall = {
            'total_predictions': len(df),
            'actual_fill_rate': df['actual_fill'].mean(),
            'avg_predicted_prob': df['predicted_prob'].mean(),
            'fill_count': df['actual_fill'].sum()
        }

        # Bucket analysis
        bucket_stats = []

        for i in range(len(buckets) - 1):
            low, high = buckets[i], buckets[i+1]

            mask = (df['predicted_prob'] >= low) & (df['predicted_prob'] < high)
            bucket_df = df[mask]

            if len(bucket_df) > 0:
                predicted_rate = bucket_df['predicted_prob'].mean()
                actual_rate = bucket_df['actual_fill'].mean()
                fill_count = bucket_df['actual_fill'].sum()

                # Calibration error
                calibration_error = actual_rate - predicted_rate

                bucket_stats.append({
                    'bucket': f'{low:.0%}-{high:.0%}',
                    'low': low,
                    'high': high,
                    'count': len(bucket_df),
                    'predicted_rate': predicted_rate,
                    'actual_rate': actual_rate,
                    'fill_count': int(fill_count),
                    'calibration_error': calibration_error,
                    'abs_error': abs(calibration_error)
                })

        # By hour analysis
        hour_stats = []
        for hour in sorted(df['hour'].unique()):
            hour_df = df[df['hour'] == hour]
            hour_stats.append({
                'hour': int(hour),
                'count': len(hour_df),
                'predicted_rate': hour_df['predicted_prob'].mean(),
                'actual_rate': hour_df['actual_fill'].mean(),
                'fill_count': int(hour_df['actual_fill'].sum())
            })

        # By offset analysis
        offset_stats = []
        for offset in sorted(df['offset'].unique()):
            offset_df = df[df['offset'] == offset]
            offset_stats.append({
                'offset': float(offset),
                'count': len(offset_df),
                'predicted_rate': offset_df['predicted_prob'].mean(),
                'actual_rate': offset_df['actual_fill'].mean(),
                'fill_count': int(offset_df['actual_fill'].sum())
            })

        # High-probability predictions (actionable)
        high_prob = df[df['predicted_prob'] >= 0.10]
        very_high_prob = df[df['predicted_prob'] >= 0.20]

        actionable = {
            'above_10pct': {
                'count': len(high_prob),
                'predicted_rate': high_prob['predicted_prob'].mean() if len(high_prob) > 0 else 0,
                'actual_rate': high_prob['actual_fill'].mean() if len(high_prob) > 0 else 0,
                'fill_count': int(high_prob['actual_fill'].sum()) if len(high_prob) > 0 else 0
            },
            'above_20pct': {
                'count': len(very_high_prob),
                'predicted_rate': very_high_prob['predicted_prob'].mean() if len(very_high_prob) > 0 else 0,
                'actual_rate': very_high_prob['actual_fill'].mean() if len(very_high_prob) > 0 else 0,
                'fill_count': int(very_high_prob['actual_fill'].sum()) if len(very_high_prob) > 0 else 0
            }
        }

        # Expected value analysis
        df['actual_profit'] = df['actual_fill'] * (2 * df['offset'] - 0.02)
        ev_stats = {
            'avg_predicted_ev': df['ev'].mean(),
            'avg_actual_profit': df['actual_profit'].mean(),
            'total_predicted_ev': df['ev'].sum(),
            'total_actual_profit': df['actual_profit'].sum()
        }

        return {
            'overall': overall,
            'by_bucket': bucket_stats,
            'by_hour': hour_stats,
            'by_offset': offset_stats,
            'actionable': actionable,
            'ev_analysis': ev_stats
        }

    def compute_top_predictions_accuracy(self, results: List[Dict], top_n: int = 50) -> Dict:
        """
        Analyze accuracy of top N predictions by expected value.

        This answers: "If we only traded the best opportunities, would they work?"
        """
        df = pd.DataFrame(results)

        if len(df) < top_n:
            return {'error': f'Only {len(df)} predictions available'}

        # Sort by EV and take top N
        top_df = df.nlargest(top_n, 'ev')

        return {
            'top_n': top_n,
            'avg_predicted_prob': top_df['predicted_prob'].mean(),
            'actual_fill_rate': top_df['actual_fill'].mean(),
            'fill_count': int(top_df['actual_fill'].sum()),
            'avg_ev': top_df['ev'].mean(),
            'total_ev': top_df['ev'].sum(),
            'actual_total_profit': (top_df['actual_fill'] * (2 * top_df['offset'] - 0.02)).sum(),
            'sample_predictions': top_df.head(10).to_dict('records')
        }


def analyze_base_rates():
    """Analyze base fill rates in the training data."""
    print("\n" + "="*60)
    print("BASE RATE ANALYSIS")
    print("="*60)

    conn = psycopg2.connect(**CONN_PARAMS)

    # Sample a few items
    sample_items = [565, 1391, 453, 560, 2]  # Blood rune, Rune sword, etc.

    target_engine = DiscreteHourTargetEngine()

    for item_id in sample_items:
        query = """
            SELECT item_id, timestamp, avg_high_price, avg_low_price,
                   high_price_volume, low_price_volume
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= '2025-06-15'
              AND timestamp <= '2026-01-06'
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=[item_id])

        if len(df) < 1000:
            continue

        # Compute targets
        targets_df = target_engine.compute_targets(df)

        print(f"\n{item_id}:")
        print(f"  Rows: {len(df):,}")

        # Analyze base rates
        for offset in [0.02, 0.025]:
            offset_str = (f"{int(offset*100)}pct" if offset*100 == int(offset*100)
                          else f"{offset*100:.1f}pct".replace('.', '_'))

            rates = []
            for hour in [1, 4, 8, 12, 24]:
                col = f'roundtrip_{offset_str}_hour{hour}'
                if col in targets_df.columns:
                    rate = targets_df[col].dropna().mean()
                    rates.append(f"h{hour}={rate:.3%}")

            if rates:
                print(f"  {offset_str}: {', '.join(rates)}")

    conn.close()


def main():
    """Run full validation."""
    print("="*60)
    print("BACKTEST VALIDATOR")
    print("="*60)

    # Initialize validator
    validator = BacktestValidator()

    # Run backtest on last 30 days of data (before training cutoff)
    # Training ended at Jan 6, so backtest on Nov 15 - Dec 15
    start_date = '2025-11-15'
    end_date = '2025-12-15'

    print(f"\nRunning backtest: {start_date} to {end_date}")

    # Full backtest
    backtest_results = validator.run_full_backtest(
        start_date=start_date,
        end_date=end_date,
        n_items=20,  # Start with 20 items
        sample_every_n=12  # Sample hourly
    )

    print(f"\nTotal predictions: {backtest_results['total_predictions']:,}")

    # Compute calibration
    calibration = validator.compute_calibration(backtest_results['results'])

    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)

    print("\nOverall:")
    print(f"  Predictions: {calibration['overall']['total_predictions']:,}")
    print(f"  Avg predicted prob: {calibration['overall']['avg_predicted_prob']:.2%}")
    print(f"  Actual fill rate: {calibration['overall']['actual_fill_rate']:.2%}")
    print(f"  Total fills: {calibration['overall']['fill_count']:,}")

    print("\nBy Probability Bucket:")
    print(f"  {'Bucket':<12} {'Count':>8} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
    print("  " + "-"*52)
    for b in calibration['by_bucket']:
        print(f"  {b['bucket']:<12} {b['count']:>8,} {b['predicted_rate']:>10.2%} "
              f"{b['actual_rate']:>10.2%} {b['calibration_error']:>+10.2%}")

    print("\nBy Hour:")
    print(f"  {'Hour':>4} {'Count':>8} {'Predicted':>10} {'Actual':>10} {'Fills':>6}")
    print("  " + "-"*44)
    for h in calibration['by_hour'][:12]:  # First 12 hours
        print(f"  {h['hour']:>4} {h['count']:>8,} {h['predicted_rate']:>10.2%} "
              f"{h['actual_rate']:>10.2%} {h['fill_count']:>6,}")

    print("\nActionable Predictions (>10% prob):")
    act = calibration['actionable']['above_10pct']
    print(f"  Count: {act['count']:,}")
    print(f"  Predicted rate: {act['predicted_rate']:.2%}")
    print(f"  Actual rate: {act['actual_rate']:.2%}")
    print(f"  Fills: {act['fill_count']:,}")

    print("\nExpected Value Analysis:")
    ev = calibration['ev_analysis']
    print(f"  Avg predicted EV: {ev['avg_predicted_ev']*100:.4f}%")
    print(f"  Avg actual profit: {ev['avg_actual_profit']*100:.4f}%")

    # Top predictions analysis
    top_analysis = validator.compute_top_predictions_accuracy(
        backtest_results['results'], top_n=50
    )

    print("\n" + "="*60)
    print("TOP 50 PREDICTIONS BY EV")
    print("="*60)
    print(f"  Avg predicted prob: {top_analysis['avg_predicted_prob']:.2%}")
    print(f"  Actual fill rate: {top_analysis['actual_fill_rate']:.2%}")
    print(f"  Fills: {top_analysis['fill_count']}")
    print(f"  Avg predicted EV: {top_analysis['avg_ev']*100:.3f}%")
    print(f"  Actual total profit: {top_analysis['actual_total_profit']*100:.3f}%")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'backtest_period': {
            'start': start_date,
            'end': end_date
        },
        'calibration': calibration,
        'top_predictions': top_analysis,
        'items_tested': backtest_results['items_tested']
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\nResults saved to backtest_results.json")

    # Also run base rate analysis
    analyze_base_rates()


if __name__ == "__main__":
    main()
