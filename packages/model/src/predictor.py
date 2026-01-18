"""
Inference Pipeline for GE Flipping Predictions

Loads trained models and generates predictions for real-time trading decisions.
"""

import os
import json
import joblib
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from .feature_engine import FeatureEngine, Granularity
    from .db_utils import CONN_PARAMS
    from .inference_config import get_confidence_tier_with_brier
except ImportError:
    from feature_engine import FeatureEngine, Granularity
    from db_utils import CONN_PARAMS
    from inference_config import get_confidence_tier_with_brier


@dataclass
class Prediction:
    """Container for a single prediction."""
    item_id: int
    item_name: str
    timestamp: datetime
    target: str
    offset_pct: float
    window_hours: int
    fill_probability: float
    expected_value: float
    current_high: float
    current_low: float
    buy_target_price: float
    sell_target_price: float
    confidence_tier: str
    model_metrics: Dict


class ModelRegistry:
    """Manages loading and caching of trained models."""

    def __init__(self, registry_path: str = 'models'):
        self.registry_path = registry_path
        self.models = {}
        self.metadata = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry index."""
        registry_file = f'{self.registry_path}/registry.json'
        if os.path.exists(registry_file):
            with open(registry_file) as f:
                self.registry_data = json.load(f)
        else:
            self.registry_data = {'items': {}}

    def get_available_items(self) -> List[int]:
        """Get list of items with trained models."""
        return [int(k) for k in self.registry_data.get('items', {}).keys()]

    def get_model_targets(self, item_id: int) -> List[str]:
        """Get list of available targets for an item."""
        item_data = self.registry_data.get('items', {}).get(str(item_id), {})
        return list(item_data.get('models', {}).keys())

    def load_model(self, item_id: int, target: str) -> Tuple[any, any, Dict]:
        """
        Load a model, scaler, and metadata for an item+target.

        Returns tuple of (model, scaler, metadata).
        """
        cache_key = f"{item_id}_{target}"
        if cache_key in self.models:
            return self.models[cache_key]

        item_data = self.registry_data.get('items', {}).get(str(item_id), {})
        model_info = item_data.get('models', {}).get(target)

        if not model_info:
            raise ValueError(f"No model found for item {item_id}, target {target}")

        model = joblib.load(model_info['model_path'])
        scaler = joblib.load(model_info['scaler_path'])

        with open(model_info['meta_path']) as f:
            metadata = json.load(f)

        self.models[cache_key] = (model, scaler, metadata)
        return model, scaler, metadata


class GEPTPredictor:
    """
    Main prediction class for GE flipping.

    Loads models and generates predictions for items.
    """

    def __init__(self, registry_path: str = 'models', use_5min: bool = True):
        """
        Initialize predictor.

        Args:
            registry_path: Path to model registry
            use_5min: If True, use 5min data; if False, use 1min (prices_latest)
        """
        self.registry = ModelRegistry(registry_path)
        self.use_5min = use_5min

        # Initialize feature engine (use 5min for now - can adapt to 1min)
        granularity = Granularity.FIVE_MIN if use_5min else Granularity.ONE_MIN
        self.feature_engine = FeatureEngine(granularity=granularity)

    def load_recent_data(self, item_id: int, hours: int = 72) -> pd.DataFrame:
        """Load recent price data for an item."""
        conn = psycopg2.connect(**CONN_PARAMS)

        if self.use_5min:
            query = """
                SELECT
                    timestamp,
                    avg_high_price,
                    avg_low_price,
                    high_price_volume,
                    low_price_volume
                FROM price_data_5min
                WHERE item_id = %s
                  AND timestamp >= NOW() - make_interval(hours => %s)
                ORDER BY timestamp
            """
        else:
            query = """
                SELECT
                    timestamp,
                    high_price as avg_high_price,
                    low_price as avg_low_price,
                    1 as high_price_volume,
                    1 as low_price_volume
                FROM prices_latest
                WHERE item_id = %s
                  AND timestamp >= NOW() - make_interval(hours => %s)
                ORDER BY timestamp
            """

        df = pd.read_sql(query, conn, params=[item_id, hours])
        conn.close()
        return df

    def predict_item(self, item_id: int, targets: Optional[List[str]] = None,
                     hours_history: int = 72) -> List[Prediction]:
        """
        Generate predictions for a single item.

        Args:
            item_id: OSRS item ID
            targets: List of targets to predict (None = all available)
            hours_history: Hours of history to load

        Returns:
            List of Prediction objects
        """
        # Load recent data
        df = self.load_recent_data(item_id, hours_history)

        if len(df) < 100:
            return []

        # Compute features
        df = self.feature_engine.compute_features(df)

        # Get latest features
        latest = df.iloc[-1]

        # Get available targets
        if targets is None:
            targets = self.registry.get_model_targets(item_id)

        predictions = []
        for target in targets:
            try:
                pred = self._predict_single_target(item_id, target, df, latest)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                print(f"Error predicting {target} for item {item_id}: {e}")

        return predictions

    def _predict_single_target(self, item_id: int, target: str,
                               df: pd.DataFrame, latest: pd.Series) -> Optional[Prediction]:
        """Generate prediction for a single target."""
        # Load model
        model, scaler, metadata = self.registry.load_model(item_id, target)

        # Get feature columns - only use columns that exist
        feature_cols = metadata['feature_columns']
        available_cols = [c for c in feature_cols if c in latest.index]

        if len(available_cols) < len(feature_cols) * 0.8:
            print(f"Warning: Only {len(available_cols)}/{len(feature_cols)} features available")
            return None

        # Build feature vector with NaN handling
        X = []
        for col in feature_cols:
            if col in latest.index:
                val = latest[col]
                if pd.isna(val) or np.isinf(val):
                    X.append(0.0)
                else:
                    X.append(float(val))
            else:
                X.append(0.0)  # Missing column

        X = np.array(X).reshape(1, -1)

        # Scale and predict
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]

        # Parse target name (e.g., "roundtrip_2pct_24h")
        parts = target.split('_')
        offset_str = parts[1]  # "2pct"
        window_str = parts[2]  # "24h"

        offset = int(offset_str.replace('pct', '')) / 100
        window_hours = int(window_str.replace('h', ''))

        # Calculate expected value
        gross_profit = 2 * offset
        net_profit = gross_profit - 0.02  # 2% tax
        expected_value = prob * net_profit

        # Get current prices
        current_high = latest['high'] if 'high' in latest else latest.get('avg_high_price', 0)
        current_low = latest['low'] if 'low' in latest else latest.get('avg_low_price', 0)

        # Calculate target prices
        buy_target = current_low * (1 - offset)
        sell_target = current_high * (1 + offset)

        # Assign confidence tier based on model metrics
        test_metrics = metadata.get('metrics', {}).get('test', {})
        confidence = self._assign_confidence_tier(test_metrics)

        return Prediction(
            item_id=item_id,
            item_name=metadata.get('item_name', f'Item-{item_id}'),
            timestamp=datetime.now(),
            target=target,
            offset_pct=offset * 100,
            window_hours=window_hours,
            fill_probability=float(prob),
            expected_value=float(expected_value),
            current_high=float(current_high),
            current_low=float(current_low),
            buy_target_price=float(buy_target),
            sell_target_price=float(sell_target),
            confidence_tier=confidence,
            model_metrics=test_metrics
        )

    def _assign_confidence_tier(self, metrics: Dict) -> str:
        """Assign confidence tier based on model metrics (Issue #70)."""
        roc_auc = metrics.get('roc_auc', 0.5)
        brier = metrics.get('brier_score', 1.0)
        return get_confidence_tier_with_brier(roc_auc, brier)

    def predict_all(self, targets: Optional[List[str]] = None) -> Dict[int, List[Prediction]]:
        """
        Generate predictions for all items with models.

        Returns dict mapping item_id to list of predictions.
        """
        all_predictions = {}

        for item_id in self.registry.get_available_items():
            predictions = self.predict_item(item_id, targets)
            if predictions:
                all_predictions[item_id] = predictions

        return all_predictions

    def get_best_opportunities(self, min_ev: float = 0.005,
                               min_fill_prob: float = 0.1) -> List[Prediction]:
        """
        Get the best trading opportunities across all items.

        Args:
            min_ev: Minimum expected value (0.005 = 0.5%)
            min_fill_prob: Minimum fill probability

        Returns:
            List of predictions sorted by expected value
        """
        all_preds = self.predict_all()

        opportunities = []
        for item_id, predictions in all_preds.items():
            for pred in predictions:
                if pred.expected_value >= min_ev and pred.fill_probability >= min_fill_prob:
                    opportunities.append(pred)

        # Sort by expected value
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        return opportunities

    def format_prediction(self, pred: Prediction) -> str:
        """Format a prediction for display."""
        return (
            f"{pred.item_name} (ID: {pred.item_id})\n"
            f"  Target: {pred.offset_pct:.1f}% offset, {pred.window_hours}h window\n"
            f"  Fill Probability: {pred.fill_probability:.1%}\n"
            f"  Expected Value: {pred.expected_value*100:.2f}%\n"
            f"  Buy Target: {pred.buy_target_price:,.0f} gp\n"
            f"  Sell Target: {pred.sell_target_price:,.0f} gp\n"
            f"  Confidence: {pred.confidence_tier}\n"
        )


def main():
    """Test the predictor."""
    print("Testing GEPTPredictor...")

    # Check if models exist
    if not os.path.exists('models/registry.json'):
        print("No models found. Run trainer.py first.")
        return

    predictor = GEPTPredictor()

    # Get available items
    items = predictor.registry.get_available_items()
    print(f"Found {len(items)} items with models")

    if items:
        # Predict for first item
        item_id = items[0]
        print(f"\nPredicting for item {item_id}...")

        predictions = predictor.predict_item(item_id)

        print(f"\nGenerated {len(predictions)} predictions:")
        for pred in predictions:
            print(predictor.format_prediction(pred))


if __name__ == "__main__":
    main()
