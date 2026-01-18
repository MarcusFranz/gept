#!/usr/bin/env python3
"""
Production Inference Pipeline for GE Flipping Predictions

Provides:
- Unified predictor for all items
- Real-time price fetching
- Expected value calculations
- Trading recommendations
"""

import os
import json
import pickle
import psycopg2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from feature_engine import FeatureEngine, Granularity
from target_engine import compute_expected_value_pct
from db_utils import CONN_PARAMS


@dataclass
class Prediction:
    """Single prediction result."""
    item_id: int
    item_name: str
    hour: int
    offset: float
    probability: float
    expected_value: float
    is_profitable: bool
    buy_price: int
    sell_price: int
    threshold: float
    confidence: str


@dataclass
class ItemRecommendation:
    """Trading recommendation for an item."""
    item_id: int
    item_name: str
    tier: int
    current_high: float
    current_low: float
    predictions: Dict[str, Prediction]
    best_config: str
    best_probability: float
    best_expected_value: float
    suggested_buy: int
    suggested_sell: int
    action: str  # 'buy', 'wait', 'skip'


class GEPTPredictor:
    """
    Production inference class for GE flipping predictions.

    Loads all trained models and provides predictions for all items.
    """

    def __init__(self, registry_path: str = 'models/registry.json'):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.models = {}
        self.scalers = {}
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self._load_all_models()

    def _load_registry(self) -> Dict:
        """Load model registry."""
        with open(self.registry_path) as f:
            return json.load(f)

    def _load_all_models(self):
        """Load all models into memory."""
        models_dir = os.path.dirname(self.registry_path)

        for item_id, item_info in self.registry.get('items', {}).items():
            self.models[item_id] = {}
            self.scalers[item_id] = {}

            for model_key, model_info in item_info.get('models', {}).items():
                if not model_info.get('is_valid', False):
                    continue

                model_path = os.path.join(models_dir, str(item_id), f'{model_key}_model.pkl')
                scaler_path = os.path.join(models_dir, str(item_id), f'{model_key}_scaler.pkl')

                try:
                    with open(model_path, 'rb') as f:
                        self.models[item_id][model_key] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[item_id][model_key] = pickle.load(f)
                except FileNotFoundError:
                    continue

        print(f"Loaded models for {len(self.models)} items")

    def fetch_recent_prices(self, item_id: int, hours: int = 72) -> pd.DataFrame:
        """Fetch recent price data for an item."""
        conn = psycopg2.connect(**CONN_PARAMS)

        query = """
            SELECT timestamp, avg_high_price, avg_low_price,
                   high_price_volume, low_price_volume
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= NOW() - make_interval(hours => %s)
              AND avg_high_price IS NOT NULL
              AND avg_low_price IS NOT NULL
            ORDER BY timestamp
        """

        df = pd.read_sql(query, conn, params=[item_id, hours])
        conn.close()

        return df

    def fetch_all_recent_prices(self, hours: int = 72) -> Dict[int, pd.DataFrame]:
        """Fetch recent prices for all items in registry."""
        prices = {}

        for item_id in self.registry.get('items', {}).keys():
            try:
                df = self.fetch_recent_prices(int(item_id), hours)
                if len(df) > 0:
                    prices[int(item_id)] = df
            except Exception:
                continue

        return prices

    def compute_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute features from price data."""
        df_features = self.feature_engine.compute_features(df)

        feature_cols = self.feature_engine.get_feature_columns()
        valid_cols = [c for c in feature_cols if c in df_features.columns]

        X = df_features[valid_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return X[-1:], df_features  # Return last row only

    def predict_item(self, item_id: int, price_data: Optional[pd.DataFrame] = None) -> ItemRecommendation:
        """Generate all predictions for a single item."""
        item_id_str = str(item_id)

        if item_id_str not in self.registry.get('items', {}):
            raise ValueError(f"Item {item_id} not in registry")

        item_info = self.registry['items'][item_id_str]
        item_name = item_info['item_name']
        tier = item_info['tier']

        # Fetch price data if not provided
        if price_data is None:
            price_data = self.fetch_recent_prices(item_id)

        if len(price_data) < 300:
            raise ValueError(f"Insufficient price data for item {item_id}")

        # Compute features
        X, df_features = self.compute_features(price_data)

        # Current prices
        current_high = float(df_features['high'].iloc[-1])
        current_low = float(df_features['low'].iloc[-1])

        # Generate predictions for all models
        predictions = {}
        best_ev = -float('inf')
        best_config = None

        for model_key, model in self.models.get(item_id_str, {}).items():
            model_info = item_info['models'].get(model_key, {})

            # Parse hour and offset from key
            parts = model_key.split('_')
            hour = int(parts[0].replace('h', ''))
            offset = float(parts[1].replace('pct', '')) / 100

            # Scale and predict
            scaler = self.scalers[item_id_str].get(model_key)
            if scaler is None:
                continue

            X_scaled = scaler.transform(X)
            prob = float(model.predict_proba(X_scaled)[0, 1])

            # Calculate suggested prices
            buy_price = int(current_low * (1 - offset))
            sell_price = int(current_high * (1 + offset))

            # Calculate EV with accurate GP-based tax
            ev = compute_expected_value_pct(prob, buy_price, sell_price)

            # Get optimal threshold from training
            threshold = model_info.get('optimal_threshold', 0.10)

            # Determine confidence
            if prob >= threshold * 2:
                confidence = 'high'
            elif prob >= threshold:
                confidence = 'medium'
            else:
                confidence = 'low'

            is_profitable = ev > 0.001 and prob >= threshold

            pred = Prediction(
                item_id=item_id,
                item_name=item_name,
                hour=hour,
                offset=offset,
                probability=prob,
                expected_value=ev,
                is_profitable=is_profitable,
                buy_price=buy_price,
                sell_price=sell_price,
                threshold=threshold,
                confidence=confidence
            )

            predictions[model_key] = pred

            if ev > best_ev and is_profitable:
                best_ev = ev
                best_config = model_key

        # Determine action
        if best_config and best_ev > 0.001:
            action = 'buy'
        elif best_ev > 0:
            action = 'wait'
        else:
            action = 'skip'

        # Get best prediction
        best_pred = predictions.get(best_config) if best_config else None

        return ItemRecommendation(
            item_id=item_id,
            item_name=item_name,
            tier=tier,
            current_high=current_high,
            current_low=current_low,
            predictions=predictions,
            best_config=best_config or 'none',
            best_probability=best_pred.probability if best_pred else 0,
            best_expected_value=best_ev if best_ev > -float('inf') else 0,
            suggested_buy=best_pred.buy_price if best_pred else 0,
            suggested_sell=best_pred.sell_price if best_pred else 0,
            action=action
        )

    def predict_all(self, price_data: Optional[Dict[int, pd.DataFrame]] = None) -> List[ItemRecommendation]:
        """Generate predictions for all items in registry."""
        if price_data is None:
            price_data = self.fetch_all_recent_prices()

        results = []

        for item_id_str in self.registry.get('items', {}).keys():
            item_id = int(item_id_str)

            if item_id not in price_data:
                continue

            try:
                rec = self.predict_item(item_id, price_data[item_id])
                results.append(rec)
            except Exception:
                continue

        return results

    def get_actionable_predictions(self, min_ev: float = 0.001) -> List[ItemRecommendation]:
        """Get predictions that meet minimum EV threshold."""
        all_predictions = self.predict_all()

        actionable = [
            p for p in all_predictions
            if p.action == 'buy' and p.best_expected_value > min_ev
        ]

        # Sort by expected value
        actionable.sort(key=lambda x: x.best_expected_value, reverse=True)

        return actionable

    def get_top_opportunities(self, n: int = 10) -> List[ItemRecommendation]:
        """Get top N trading opportunities by expected value."""
        actionable = self.get_actionable_predictions()
        return actionable[:n]


def format_prediction_table(predictions: List[ItemRecommendation]) -> str:
    """Format predictions as a table."""
    if not predictions:
        return "No actionable predictions"

    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Item':<25} {'Tier':>4} {'Config':<12} {'Prob':>6} {'EV':>8} {'Buy':>10} {'Sell':>10} {'Action':<6}")
    lines.append("-" * 100)

    for p in predictions:
        lines.append(
            f"{p.item_name[:24]:<25} {p.tier:>4} {p.best_config:<12} "
            f"{p.best_probability:>5.1%} {p.best_expected_value:>7.3%} "
            f"{p.suggested_buy:>10,} {p.suggested_sell:>10,} {p.action:<6}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    """Test the predictor."""
    print("Loading GEPTPredictor...")
    predictor = GEPTPredictor('models/registry.json')

    print("\nFetching top opportunities...")
    top = predictor.get_top_opportunities(20)

    print(format_prediction_table(top))

    # Summary
    all_preds = predictor.predict_all()
    buy_count = sum(1 for p in all_preds if p.action == 'buy')
    wait_count = sum(1 for p in all_preds if p.action == 'wait')
    skip_count = sum(1 for p in all_preds if p.action == 'skip')

    print("\nSummary:")
    print(f"  Total items: {len(all_preds)}")
    print(f"  Buy: {buy_count}")
    print(f"  Wait: {wait_count}")
    print(f"  Skip: {skip_count}")

    if buy_count > 0:
        total_ev = sum(p.best_expected_value for p in all_preds if p.action == 'buy')
        avg_ev = total_ev / buy_count
        print(f"  Average EV (buy): {avg_ev:.3%}")


if __name__ == '__main__':
    main()
