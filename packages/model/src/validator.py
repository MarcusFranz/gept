"""
Model Validation and Backtesting for GE Flipping

Validates model predictions against historical data and computes
realistic profit expectations.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from feature_engine import FeatureEngine, Granularity
from target_engine import TargetEngine
from db_utils import get_simple_connection


@dataclass
class ValidationResult:
    """Results from validating a model."""
    item_id: int
    item_name: str
    target: str
    is_valid: bool
    metrics: Dict
    issues: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Results from backtesting a strategy."""
    item_id: int
    item_name: str
    offset_pct: float
    window_hours: int
    n_trades: int
    n_wins: int
    win_rate: float
    total_profit_pct: float
    avg_profit_per_trade: float
    expected_value: float
    actual_vs_expected: float  # Ratio of actual to expected profit


class ModelValidator:
    """Validates trained models for quality and reliability."""

    def __init__(self, min_accuracy: float = 0.52,
                 min_roc_auc: float = 0.50,
                 max_brier: float = 0.30):
        """
        Initialize validator with thresholds.

        Args:
            min_accuracy: Minimum accuracy (must beat random)
            min_roc_auc: Minimum ROC AUC (0.5 = random)
            max_brier: Maximum Brier score (lower is better calibration)
        """
        self.min_accuracy = min_accuracy
        self.min_roc_auc = min_roc_auc
        self.max_brier = max_brier

    def validate_metrics(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Validate model metrics against thresholds.

        Returns (is_valid, list of issues).
        """
        issues = []

        accuracy = metrics.get('accuracy', 0)
        roc_auc = metrics.get('roc_auc', 0.5)
        brier = metrics.get('brier_score', 1.0)
        base_rate = metrics.get('base_rate', 0.5)

        # Check accuracy beats baseline
        if accuracy < self.min_accuracy:
            issues.append(f"Accuracy {accuracy:.3f} below threshold {self.min_accuracy}")

        # Check accuracy is significantly above base rate
        if accuracy < base_rate + 0.02:
            issues.append(f"Accuracy {accuracy:.3f} not significantly above base rate {base_rate:.3f}")

        # Check ROC AUC
        if roc_auc < self.min_roc_auc:
            issues.append(f"ROC AUC {roc_auc:.3f} below threshold {self.min_roc_auc}")

        # Check calibration
        if brier > self.max_brier:
            issues.append(f"Brier score {brier:.3f} above threshold {self.max_brier}")

        return len(issues) == 0, issues

    def validate_calibration_curve(self, calibration: Dict) -> Tuple[bool, List[str]]:
        """
        Validate that predicted probabilities match actual frequencies.

        A well-calibrated model should have calibration curve close to diagonal.
        """
        issues = []

        bin_centers = calibration.get('bin_centers', [])
        actual_rates = calibration.get('actual_rates', [])

        if not bin_centers or not actual_rates:
            issues.append("No calibration data available")
            return False, issues

        # Check calibration error
        errors = [abs(pred - actual) for pred, actual in zip(bin_centers, actual_rates)]
        avg_error = np.mean(errors) if errors else 1.0

        if avg_error > 0.15:
            issues.append(f"Average calibration error {avg_error:.3f} is high")

        # Check for systematic bias (all predictions too high or too low)
        biases = [pred - actual for pred, actual in zip(bin_centers, actual_rates)]
        avg_bias = np.mean(biases) if biases else 0

        if abs(avg_bias) > 0.10:
            direction = "overconfident" if avg_bias > 0 else "underconfident"
            issues.append(f"Model is {direction}: avg bias {avg_bias:.3f}")

        return len(issues) == 0, issues


class Backtester:
    """
    Backtests trading strategies on historical data.

    Simulates what would happen if we traded based on model predictions.
    """

    def __init__(self, registry_path: str = 'models'):
        self.registry_path = registry_path
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self.target_engine = TargetEngine(granularity='5m')

    def load_model(self, item_id: int, target: str) -> Tuple[any, any, Dict]:
        """Load model, scaler, and metadata."""
        item_dir = f'{self.registry_path}/{item_id}'
        model = joblib.load(f'{item_dir}/{target}_model.pkl')
        scaler = joblib.load(f'{item_dir}/{target}_scaler.pkl')
        with open(f'{item_dir}/{target}_meta.json') as f:
            meta = json.load(f)
        return model, scaler, meta

    def load_test_data(self, item_id: int, test_months: int = 3) -> pd.DataFrame:
        """Load recent data for backtesting."""
        conn = get_simple_connection()

        query = """
            SELECT
                timestamp,
                avg_high_price,
                avg_low_price,
                high_price_volume,
                low_price_volume
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= NOW() - make_interval(months => %s)
            ORDER BY timestamp
        """

        try:
            df = pd.read_sql(query, conn, params=[item_id, test_months])
        finally:
            conn.close()
        return df

    def run_backtest(self, item_id: int, target: str,
                     probability_threshold: float = 0.5,
                     test_months: int = 3) -> Optional[BacktestResult]:
        """
        Run backtest for a single item and target.

        Args:
            item_id: OSRS item ID
            target: Target name (e.g., "roundtrip_2pct_24h")
            probability_threshold: Minimum probability to trade
            test_months: Months of data to backtest

        Returns:
            BacktestResult or None if insufficient data
        """
        try:
            # Load model
            model, scaler, meta = self.load_model(item_id, target)
            feature_cols = meta['feature_columns']
            item_name = meta.get('item_name', f'Item-{item_id}')

            # Parse target
            parts = target.split('_')
            offset = int(parts[1].replace('pct', '')) / 100
            window_hours = int(parts[2].replace('h', ''))
            window_periods = window_hours * 12  # 5min periods

            # Load and prepare data
            df = self.load_test_data(item_id, test_months)
            if len(df) < 1000:
                return None

            df = self.feature_engine.compute_features(df)
            df = self.target_engine.compute_targets(df)

            # Drop warmup rows
            df = df.iloc[300:-window_periods].reset_index(drop=True)

            if len(df) < 500:
                return None

            # Get predictions
            X = df[feature_cols].values
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            X_scaled = scaler.transform(X)

            probs = model.predict_proba(X_scaled)[:, 1]
            actuals = df[target].values

            # Simulate trades
            trades = probs >= probability_threshold
            n_trades = trades.sum()

            if n_trades == 0:
                return BacktestResult(
                    item_id=item_id,
                    item_name=item_name,
                    offset_pct=offset * 100,
                    window_hours=window_hours,
                    n_trades=0,
                    n_wins=0,
                    win_rate=0,
                    total_profit_pct=0,
                    avg_profit_per_trade=0,
                    expected_value=0,
                    actual_vs_expected=0
                )

            # Calculate results
            wins = actuals[trades].sum()
            win_rate = wins / n_trades

            net_profit_per_win = 2 * offset - 0.02  # 2*offset gross - 2% tax
            total_profit = wins * net_profit_per_win
            avg_profit = total_profit / n_trades

            # Expected value from predictions
            expected_profit = (probs[trades] * net_profit_per_win).sum()
            expected_per_trade = expected_profit / n_trades

            actual_vs_expected = total_profit / expected_profit if expected_profit > 0 else 0

            return BacktestResult(
                item_id=item_id,
                item_name=item_name,
                offset_pct=offset * 100,
                window_hours=window_hours,
                n_trades=int(n_trades),
                n_wins=int(wins),
                win_rate=float(win_rate),
                total_profit_pct=float(total_profit * 100),
                avg_profit_per_trade=float(avg_profit * 100),
                expected_value=float(expected_per_trade * 100),
                actual_vs_expected=float(actual_vs_expected)
            )

        except Exception as e:
            print(f"Backtest error for {item_id}/{target}: {e}")
            return None

    def run_full_backtest(self, test_months: int = 3) -> List[BacktestResult]:
        """Run backtest for all available models."""
        results = []

        # Load registry
        registry_file = f'{self.registry_path}/registry.json'
        if not os.path.exists(registry_file):
            print("No model registry found")
            return results

        with open(registry_file) as f:
            registry = json.load(f)

        for item_id_str, item_data in registry.get('items', {}).items():
            item_id = int(item_id_str)

            for target in item_data.get('models', {}).keys():
                result = self.run_backtest(item_id, target, test_months=test_months)
                if result:
                    results.append(result)

        return results


def validate_all_models(registry_path: str = 'models') -> Dict[str, ValidationResult]:
    """Validate all models in registry."""
    validator = ModelValidator()
    results = {}

    registry_file = f'{registry_path}/registry.json'
    if not os.path.exists(registry_file):
        print("No model registry found")
        return results

    with open(registry_file) as f:
        registry = json.load(f)

    for item_id_str, item_data in registry.get('items', {}).items():
        item_id = int(item_id_str)

        for target, model_info in item_data.get('models', {}).items():
            # Load metadata
            with open(model_info['meta_path']) as f:
                meta = json.load(f)

            test_metrics = meta.get('metrics', {}).get('test', {})
            calibration = meta.get('metrics', {}).get('calibration', {})

            # Validate
            metrics_valid, metrics_issues = validator.validate_metrics(test_metrics)
            calib_valid, calib_issues = validator.validate_calibration_curve(calibration)

            all_issues = metrics_issues + calib_issues
            is_valid = metrics_valid and calib_valid

            result = ValidationResult(
                item_id=item_id,
                item_name=meta.get('item_name', f'Item-{item_id}'),
                target=target,
                is_valid=is_valid,
                metrics=test_metrics,
                issues=all_issues
            )

            results[f"{item_id}_{target}"] = result

    return results


def main():
    """Run validation and backtesting."""
    print("="*70)
    print("MODEL VALIDATION AND BACKTESTING")
    print("="*70)

    # Check if models exist
    if not os.path.exists('models/registry.json'):
        print("No models found. Run trainer.py first.")
        return

    # Validate all models
    print("\n1. Validating all models...")
    validation_results = validate_all_models()

    valid_count = sum(1 for r in validation_results.values() if r.is_valid)
    print(f"   {valid_count}/{len(validation_results)} models passed validation")

    # Show issues
    for key, result in validation_results.items():
        if result.issues:
            print(f"\n   {result.item_name} - {result.target}:")
            for issue in result.issues:
                print(f"     - {issue}")

    # Run backtests
    print("\n2. Running backtests...")
    backtester = Backtester()
    backtest_results = backtester.run_full_backtest(test_months=3)

    print(f"   Completed {len(backtest_results)} backtests")

    # Show best results
    profitable = [r for r in backtest_results if r.total_profit_pct > 0]
    profitable.sort(key=lambda x: x.avg_profit_per_trade, reverse=True)

    if profitable:
        print("\n" + "="*70)
        print("TOP 10 PROFITABLE STRATEGIES (by avg profit per trade)")
        print("="*70)
        print(f"{'Item':<25} {'Offset':>8} {'Window':>8} {'Trades':>8} {'WinRate':>10} {'AvgProfit':>10}")
        print("-"*70)

        for result in profitable[:10]:
            print(f"{result.item_name[:25]:<25} {result.offset_pct:>7.1f}% "
                  f"{result.window_hours:>6}h {result.n_trades:>8} "
                  f"{result.win_rate*100:>9.1f}% {result.avg_profit_per_trade:>9.2f}%")

    # Summary statistics
    if backtest_results:
        print("\n" + "="*70)
        print("BACKTEST SUMMARY")
        print("="*70)
        total_trades = sum(r.n_trades for r in backtest_results)
        total_wins = sum(r.n_wins for r in backtest_results)
        avg_win_rate = total_wins / total_trades if total_trades > 0 else 0

        print(f"Total trades simulated: {total_trades:,}")
        print(f"Overall win rate: {avg_win_rate:.1%}")

        profitable_strategies = [r for r in backtest_results if r.avg_profit_per_trade > 0]
        print(f"Profitable strategies: {len(profitable_strategies)}/{len(backtest_results)}")

        if profitable_strategies:
            avg_profit = np.mean([r.avg_profit_per_trade for r in profitable_strategies])
            print(f"Average profit per trade (profitable only): {avg_profit:.2f}%")


if __name__ == "__main__":
    main()
