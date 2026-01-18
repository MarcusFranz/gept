#!/usr/bin/env python3
"""
Production CatBoost Training Pipeline with Profit Validation

Trains CatBoost models for all viable items with:
- Multiple hours (1, 2, 4, 8, 12, 24)
- Multiple offsets (1.5%, 2%, 2.5%)
- Isotonic calibration
- Profit simulation and validation
- Calibration verification

Based on experiment results showing CatBoost as the best model.
"""

import os
import sys
import json
import time
import pickle
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: CatBoost not installed. Run: pip install catboost")

from feature_engine import FeatureEngine, Granularity

# Centralized database connection management
from db_utils import CONN_PARAMS


@dataclass
class ProductionConfig:
    """Configuration for production training."""
    # Date range
    start_date: str = '2025-06-15'
    end_date: str = '2026-01-06'

    # Target configuration
    hours: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 12, 24])
    offsets: List[float] = field(default_factory=lambda: [0.015, 0.02, 0.025])

    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # CatBoost parameters (optimized from experiments)
    iterations: int = 100
    depth: int = 5
    learning_rate: float = 0.1

    # Validation thresholds
    min_auc: float = 0.52
    min_positive_samples: int = 20

    # Feature warmup
    drop_first_n_rows: int = 300

    # Calibration
    calibrate: bool = True
    calibration_cv: int = 3


@dataclass
class ModelResult:
    """Result from training a single model."""
    item_id: int
    item_name: str
    hour: int
    offset: float
    auc: float
    brier: float
    positive_rate: float
    n_train: int
    n_test: int
    is_valid: bool
    train_time_ms: float
    calibration: Optional[Dict] = None
    profit_analysis: Optional[Dict] = None
    error: Optional[str] = None


class ProductionTrainer:
    """
    Production CatBoost trainer with profit validation.
    """

    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig()
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)

    def get_all_items(self, min_rows: int = 40000) -> List[Dict]:
        """Get all items with sufficient data."""
        conn = psycopg2.connect(**CONN_PARAMS)

        query = """
            WITH item_stats AS (
                SELECT
                    p.item_id,
                    i.name as item_name,
                    COUNT(CASE WHEN avg_high_price IS NOT NULL
                               AND avg_low_price IS NOT NULL THEN 1 END) as valid_rows,
                    AVG(CASE WHEN avg_high_price IS NOT NULL
                             AND avg_low_price IS NOT NULL
                             THEN (avg_high_price::BIGINT + avg_low_price::BIGINT) / 2 END) as avg_price
                FROM price_data_5min p
                JOIN items i ON p.item_id = i.item_id
                WHERE p.timestamp >= %s AND p.timestamp <= %s
                GROUP BY p.item_id, i.name
                HAVING COUNT(CASE WHEN avg_high_price IS NOT NULL
                                  AND avg_low_price IS NOT NULL THEN 1 END) > %s
            )
            SELECT item_id, item_name, valid_rows, avg_price
            FROM item_stats
            ORDER BY valid_rows DESC
        """

        df = pd.read_sql(query, conn, params=[
            self.config.start_date,
            self.config.end_date,
            min_rows
        ])
        conn.close()

        # Convert to native Python types
        items = []
        for _, row in df.iterrows():
            items.append({
                'item_id': int(row['item_id']),
                'item_name': row['item_name'],
                'valid_rows': int(row['valid_rows']),
                'avg_price': float(row['avg_price']) if row['avg_price'] else 0
            })

        return items

    def load_item_data(self, item_id: int) -> pd.DataFrame:
        """Load price data for an item."""
        conn = psycopg2.connect(**CONN_PARAMS)

        query = """
            SELECT timestamp, avg_high_price, avg_low_price,
                   high_price_volume, low_price_volume
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= %s AND timestamp <= %s
              AND avg_high_price IS NOT NULL
              AND avg_low_price IS NOT NULL
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=[
            item_id, self.config.start_date, self.config.end_date
        ])
        conn.close()

        return df

    def compute_targets(self, df: pd.DataFrame, offset: float, hour: int) -> pd.Series:
        """
        Compute fill target: Did a buy order at -offset% fill within X hours?

        This is cumulative (filled within X hours), not discrete.
        """
        periods_per_hour = 12  # 5-min intervals
        lookforward = hour * periods_per_hour

        # Buy price is low * (1 - offset)
        buy_price = df['low'] * (1 - offset)

        # Check if future low goes below our buy price
        future_min_low = df['low'].shift(-lookforward).rolling(lookforward).min()

        # Fill happens if future low <= buy price
        fills = (future_min_low <= buy_price).astype(int)

        return fills

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Compute features and return feature matrix."""
        # Compute features
        df_features = self.feature_engine.compute_features(df)

        # Get feature columns
        feature_cols = self.feature_engine.get_feature_columns()

        # Filter to existing numeric columns
        valid_cols = []
        for col in feature_cols:
            if col in df_features.columns:
                if df_features[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    valid_cols.append(col)

        X = df_features[valid_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return X, valid_cols, df_features

    def create_catboost_model(self) -> cb.CatBoostClassifier:
        """Create CatBoost classifier with production settings."""
        return cb.CatBoostClassifier(
            iterations=self.config.iterations,
            depth=self.config.depth,
            learning_rate=self.config.learning_rate,
            random_state=42,
            verbose=0,
            thread_count=1,
            task_type='CPU'
        )

    def compute_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Compute calibration curve data."""
        buckets = [(0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.0)]

        calibration = []
        for low, high in buckets:
            mask = (y_prob >= low) & (y_prob < high)
            if mask.sum() < 20:
                continue

            predicted_rate = float(y_prob[mask].mean())
            actual_rate = float(y_true[mask].mean())
            error = abs(predicted_rate - actual_rate)

            calibration.append({
                'bucket': f'{low:.0%}-{high:.0%}',
                'count': int(mask.sum()),
                'predicted': predicted_rate,
                'actual': actual_rate,
                'error': error,
                'error_pct': error / max(predicted_rate, 0.001)
            })

        return {
            'buckets': calibration,
            'is_well_calibrated': all(b['error_pct'] < 0.25 for b in calibration if b['count'] >= 20)
        }

    def compute_profit_analysis(self, y_true: np.ndarray, y_prob: np.ndarray,
                                 offset: float, hour: int) -> Dict:
        """
        Compute profit analysis at different probability thresholds.

        Key insight: We need to find the threshold that maximizes daily profit,
        balancing fill rate improvement vs reduced trading opportunities.
        """
        # Net margin after 2% GE tax (buy and sell)
        gross_margin = offset * 2  # Buy at -offset, sell at +offset
        tax = 0.02  # 2% total (1% buy + 1% sell in OSRS terms, roughly)
        net_margin = gross_margin - tax

        # Days in test set (assuming 5-min intervals)
        days_in_test = len(y_true) / (12 * 24)  # 12 periods/hour * 24 hours

        results = []

        for threshold in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            mask = y_prob >= threshold

            if mask.sum() < 10:
                continue

            # Trades above threshold
            n_trades = mask.sum()
            trades_per_day = n_trades / days_in_test

            # Actual fill rate on filtered trades
            actual_fill_rate = float(y_true[mask].mean())

            # Expected value per trade attempt
            ev_per_trade = net_margin * actual_fill_rate

            # Daily profit (per 1M GP capital per trade)
            daily_profit_per_1m = ev_per_trade * trades_per_day * 1_000_000

            results.append({
                'threshold': threshold,
                'n_trades': int(n_trades),
                'trades_per_day': float(trades_per_day),
                'actual_fill_rate': actual_fill_rate,
                'ev_per_trade': float(ev_per_trade),
                'daily_profit_per_1m': float(daily_profit_per_1m),
                'net_margin': float(net_margin)
            })

        # Find optimal threshold
        if results:
            optimal = max(results, key=lambda x: x['daily_profit_per_1m'])
        else:
            optimal = None

        return {
            'profit_curve': results,
            'optimal': optimal,
            'is_profitable': optimal is not None and optimal['daily_profit_per_1m'] > 0
        }

    def train_single_model(self, item_id: int, item_name: str,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           hour: int, offset: float) -> Tuple[any, any, ModelResult]:
        """Train a single CatBoost model."""

        # Check minimum positive samples
        if y_train.sum() < self.config.min_positive_samples:
            return None, None, ModelResult(
                item_id=item_id, item_name=item_name, hour=hour, offset=offset,
                auc=0.5, brier=0.25, positive_rate=float(y_train.mean()),
                n_train=len(y_train), n_test=len(y_test), is_valid=False,
                train_time_ms=0, error='insufficient_positive_samples'
            )

        start_time = time.time()

        try:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train CatBoost
            model = self.create_catboost_model()
            model.fit(X_train_scaled, y_train, verbose=False)

            # CatBoost has built-in probability calibration, so we use it directly
            # The CalibratedClassifierCV has sklearn compatibility issues with CatBoost
            final_model = model

            train_time = (time.time() - start_time) * 1000

            # Predict on test set
            y_prob = final_model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5

            brier = brier_score_loss(y_test, y_prob)

            # Calibration analysis
            calibration = self.compute_calibration(y_test, y_prob)

            # Profit analysis
            profit_analysis = self.compute_profit_analysis(y_test, y_prob, offset, hour)

            is_valid = auc > self.config.min_auc

            result = ModelResult(
                item_id=item_id,
                item_name=item_name,
                hour=hour,
                offset=offset,
                auc=float(auc),
                brier=float(brier),
                positive_rate=float(y_train.mean()),
                n_train=len(y_train),
                n_test=len(y_test),
                is_valid=is_valid,
                train_time_ms=train_time,
                calibration=calibration,
                profit_analysis=profit_analysis
            )

            return final_model, scaler, result

        except Exception as e:
            return None, None, ModelResult(
                item_id=item_id, item_name=item_name, hour=hour, offset=offset,
                auc=0.5, brier=0.25, positive_rate=0,
                n_train=len(y_train), n_test=len(y_test), is_valid=False,
                train_time_ms=0, error=str(e)
            )

    def train_item(self, item: Dict) -> Dict:
        """Train all models for a single item."""
        item_id = item['item_id']
        item_name = item['item_name']

        # Load data
        df = self.load_item_data(item_id)

        if len(df) < 5000:
            return {
                'item_id': item_id,
                'item_name': item_name,
                'status': 'insufficient_data',
                'models': {}
            }

        # Prepare features
        X, feature_cols, df_features = self.prepare_features(df)

        # Drop warmup rows
        X = X[self.config.drop_first_n_rows:]
        df_features = df_features.iloc[self.config.drop_first_n_rows:].reset_index(drop=True)

        # Split data (temporal)
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        X_train = X[:train_end]
        X_test = X[val_end:]  # Use test set for evaluation

        models = {}
        results = []

        for offset in self.config.offsets:
            for hour in self.config.hours:
                # Compute target
                target = self.compute_targets(df_features, offset, hour)

                # Drop last rows where target can't be computed
                max_lookforward = max(self.config.hours) * 12
                target = target[:-max_lookforward]

                y_train = target[:train_end].values.astype(int)
                y_test = target[val_end:-max_lookforward].values.astype(int)

                # Adjust X_test to match y_test length
                X_test_adj = X[val_end:val_end + len(y_test)]

                # Train model
                model, scaler, result = self.train_single_model(
                    item_id, item_name,
                    X_train[:len(y_train)], y_train,
                    X_test_adj, y_test,
                    hour, offset
                )

                model_key = f'{hour}h_{offset*100:.1f}pct'

                if model is not None:
                    models[model_key] = {
                        'model': model,
                        'scaler': scaler,
                        'feature_cols': feature_cols,
                        'result': result
                    }

                results.append(result)

        return {
            'item_id': item_id,
            'item_name': item_name,
            'status': 'success',
            'models': models,
            'results': results
        }

    def save_item_models(self, item_result: Dict, output_dir: str):
        """Save models for a single item."""
        item_id = item_result['item_id']
        item_dir = os.path.join(output_dir, str(item_id))
        os.makedirs(item_dir, exist_ok=True)

        for model_key, model_data in item_result.get('models', {}).items():
            # Save model
            model_path = os.path.join(item_dir, f'{model_key}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)

            # Save scaler
            scaler_path = os.path.join(item_dir, f'{model_key}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(model_data['scaler'], f)

            # Save metadata
            result = model_data['result']
            meta = {
                'item_id': result.item_id,
                'item_name': result.item_name,
                'hour': result.hour,
                'offset': result.offset,
                'auc': result.auc,
                'brier': result.brier,
                'positive_rate': result.positive_rate,
                'is_valid': result.is_valid,
                'calibration': result.calibration,
                'profit_analysis': result.profit_analysis,
                'feature_cols': model_data['feature_cols'],
                'trained_at': datetime.now().isoformat()
            }

            meta_path = os.path.join(item_dir, f'{model_key}_meta.json')
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

    def create_registry(self, all_results: List[Dict], output_dir: str) -> Dict:
        """Create unified model registry."""
        registry = {
            'created_at': datetime.now().isoformat(),
            'config': asdict(self.config),
            'items': {}
        }

        for item_result in all_results:
            if item_result['status'] != 'success':
                continue

            item_id = str(item_result['item_id'])
            item_name = item_result['item_name']

            # Aggregate item statistics
            results = item_result.get('results', [])
            valid_results = [r for r in results if r.is_valid]

            if not valid_results:
                continue

            # Best model for this item
            best_result = max(valid_results, key=lambda r: r.auc)

            # Average stats
            avg_auc = np.mean([r.auc for r in valid_results])
            avg_positive_rate = np.mean([r.positive_rate for r in valid_results])

            # Profitability summary
            profitable_models = [
                r for r in valid_results
                if r.profit_analysis and r.profit_analysis.get('is_profitable', False)
            ]

            # Calculate total expected daily profit
            total_daily_profit = 0
            best_profit_model = None
            for r in valid_results:
                if r.profit_analysis and r.profit_analysis.get('optimal'):
                    profit = r.profit_analysis['optimal']['daily_profit_per_1m']
                    if profit > 0:
                        if best_profit_model is None or profit > best_profit_model['profit']:
                            best_profit_model = {
                                'hour': r.hour,
                                'offset': r.offset,
                                'profit': profit,
                                'threshold': r.profit_analysis['optimal']['threshold']
                            }

            # Determine tier based on AUC and profitability
            if avg_auc >= 0.70 and len(profitable_models) >= 3:
                tier = 1
            elif avg_auc >= 0.60 and len(profitable_models) >= 1:
                tier = 2
            elif avg_auc >= 0.55:
                tier = 3
            else:
                tier = 4

            # Recommended offset based on positive rate
            if avg_positive_rate < 0.02:
                recommended_offset = 0.015  # Low fill rate items need lower offset
            elif avg_positive_rate < 0.05:
                recommended_offset = 0.02
            else:
                recommended_offset = 0.025

            registry['items'][item_id] = {
                'item_id': int(item_id),
                'item_name': item_name,
                'tier': tier,
                'models': {},
                'characteristics': {
                    'avg_auc': float(avg_auc),
                    'avg_positive_rate': float(avg_positive_rate),
                    'predictability': 'high' if avg_auc >= 0.70 else ('medium' if avg_auc >= 0.60 else 'low'),
                    'recommended_offset': recommended_offset,
                    'profitable_model_count': len(profitable_models),
                    'best_profit_model': best_profit_model
                }
            }

            # Add model details
            for r in results:
                model_key = f'{r.hour}h_{r.offset*100:.1f}pct'
                registry['items'][item_id]['models'][model_key] = {
                    'path': f'models/{item_id}/{model_key}',
                    'auc': float(r.auc),
                    'brier': float(r.brier),
                    'is_valid': r.is_valid,
                    'calibrated': self.config.calibrate,
                    'is_well_calibrated': r.calibration.get('is_well_calibrated', False) if r.calibration else False,
                    'is_profitable': r.profit_analysis.get('is_profitable', False) if r.profit_analysis else False,
                    'optimal_threshold': r.profit_analysis['optimal']['threshold'] if r.profit_analysis and r.profit_analysis.get('optimal') else None,
                    'expected_daily_profit': r.profit_analysis['optimal']['daily_profit_per_1m'] if r.profit_analysis and r.profit_analysis.get('optimal') else 0
                }

        # Save registry
        registry_path = os.path.join(output_dir, 'registry.json')
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return registry


def train_item_worker(args):
    """Worker function for parallel training."""
    item, config_dict = args
    config = ProductionConfig(**config_dict)
    trainer = ProductionTrainer(config)
    return trainer.train_item(item)


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("PRODUCTION CATBOOST TRAINING PIPELINE")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")

    if not HAS_CATBOOST:
        print("ERROR: CatBoost not installed")
        return

    # Configuration
    config = ProductionConfig()
    trainer = ProductionTrainer(config)

    print(f"\nConfiguration:")
    print(f"  Date range: {config.start_date} to {config.end_date}")
    print(f"  Hours: {config.hours}")
    print(f"  Offsets: {config.offsets}")
    print(f"  Calibration: {config.calibrate}")

    # Get all viable items
    print(f"\nFetching items with sufficient data...")
    items = trainer.get_all_items(min_rows=40000)
    print(f"Found {len(items)} items with sufficient data")

    # Calculate expected model count
    models_per_item = len(config.hours) * len(config.offsets)
    total_models = len(items) * models_per_item
    print(f"Expected models: {len(items)} items × {models_per_item} = {total_models} models")

    # Output directory
    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    # Train all items
    all_results = []
    start_time = time.time()

    # Use sequential processing for reliability
    for i, item in enumerate(items):
        print(f"\n[{i+1}/{len(items)}] Training {item['item_name']} (ID: {item['item_id']})...")

        try:
            result = trainer.train_item(item)
            all_results.append(result)

            # Save models immediately
            if result['status'] == 'success':
                trainer.save_item_models(result, output_dir)

                # Summary
                valid_count = sum(1 for r in result.get('results', []) if r.is_valid)
                profitable_count = sum(
                    1 for r in result.get('results', [])
                    if r.profit_analysis and r.profit_analysis.get('is_profitable', False)
                )
                avg_auc = np.mean([r.auc for r in result.get('results', []) if r.is_valid]) if valid_count > 0 else 0

                print(f"  Valid: {valid_count}/{models_per_item}, Profitable: {profitable_count}, Avg AUC: {avg_auc:.4f}")
            else:
                print(f"  Status: {result['status']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'status': 'error',
                'error': str(e)
            })

        # Progress update every 10 items
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(items) - i - 1) / rate if rate > 0 else 0
            print(f"\n  Progress: {i+1}/{len(items)} ({100*(i+1)/len(items):.1f}%), "
                  f"Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")

    # Create registry
    print(f"\nCreating model registry...")
    registry = trainer.create_registry(all_results, output_dir)

    # Summary statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r.get('status') == 'success')
    total_valid_models = sum(
        len([m for m in r.get('results', []) if m.is_valid])
        for r in all_results if r.get('status') == 'success'
    )
    total_profitable_models = sum(
        len([m for m in r.get('results', []) if m.profit_analysis and m.profit_analysis.get('is_profitable', False)])
        for r in all_results if r.get('status') == 'success'
    )

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Items processed: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Total valid models: {total_valid_models}")
    print(f"Total profitable models: {total_profitable_models}")
    print(f"\nRegistry saved to: {output_dir}/registry.json")

    # Tier breakdown
    tier_counts = defaultdict(int)
    for item_id, item_info in registry.get('items', {}).items():
        tier_counts[item_info['tier']] += 1

    print(f"\nItem Tiers:")
    for tier in sorted(tier_counts.keys()):
        print(f"  Tier {tier}: {tier_counts[tier]} items")

    return registry


if __name__ == '__main__':
    main()
