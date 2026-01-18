#!/usr/bin/env python3
"""
Per-Target Batch Predictor

Loads and runs inference on models saved in the new per-target format:
    models/<run_id>/item_<item_id>/<target_name>.cbm

This is the format produced by continuous_scheduler.py.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from catboost import CatBoostClassifier
from psycopg2.extras import execute_values

from feature_engine import FeatureEngine, Granularity
from db_utils import get_db_connection, get_db_cursor
from inference_config import clip_probability, get_confidence_tier, DEFAULT_AUC

logger = logging.getLogger(__name__)


def find_latest_model_run(models_dir: str = "models") -> Optional[str]:
    """Find the most recent model run directory."""
    if not os.path.exists(models_dir):
        return None

    runs = []
    for d in os.listdir(models_dir):
        path = os.path.join(models_dir, d)
        if os.path.isdir(path) and d.startswith("20"):
            runs.append(d)

    if not runs:
        return None

    runs.sort(reverse=True)
    return os.path.join(models_dir, runs[0])


class PerTargetBatchPredictor:
    """
    Batch predictor for per-target CatBoost models.

    Expected model structure:
        models/<run_id>/item_<item_id>/<target_name>.cbm

    Where target_name is like: buy_fills_1pct_4h, sell_fills_2pct_12h, etc.
    """

    # Target configurations
    OFFSETS = [1, 2, 3]  # 1%, 2%, 3%
    WINDOWS = ['4h', '8h', '12h', '24h', '48h']
    DIRECTIONS = ['buy_fills', 'sell_fills', 'roundtrip']

    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize predictor.

        Args:
            models_dir: Path to model run directory. Auto-detects latest if None.
        """
        if models_dir is None:
            models_dir = find_latest_model_run()
            if models_dir is None:
                raise FileNotFoundError("No model runs found in models/")
            logger.info(f"Using latest model run: {models_dir}")

        self.models_dir = Path(models_dir)
        self.models: Dict[int, Dict[str, CatBoostClassifier]] = {}
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self.run_id = self.models_dir.name

        self._load_all_models()

    def _load_all_models(self):
        """Load all models from the run directory."""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        loaded = 0
        skipped = 0

        for item_dir in os.listdir(self.models_dir):
            if not item_dir.startswith("item_"):
                continue

            item_path = self.models_dir / item_dir
            if not item_path.is_dir():
                continue

            try:
                item_id = int(item_dir.replace("item_", ""))
            except ValueError:
                continue

            # Load all target models for this item
            item_models = {}
            for model_file in item_path.glob("*.cbm"):
                target_name = model_file.stem  # e.g., buy_fills_1pct_4h
                try:
                    model = CatBoostClassifier()
                    model.load_model(str(model_file))
                    item_models[target_name] = model
                except Exception as e:
                    logger.warning(f"Failed to load {model_file}: {e}")

            if item_models:
                self.models[item_id] = item_models
                loaded += 1
            else:
                skipped += 1

        logger.info(f"Loaded models for {loaded} items ({skipped} skipped)")

    def load_recent_prices(self, hours: int = 72) -> Dict[int, pd.DataFrame]:
        """Load recent price data for all items with models."""
        price_data = {}

        with get_db_connection() as conn:
            for item_id in self.models.keys():
                query = """
                    SELECT timestamp,
                           avg_high_price,
                           avg_low_price,
                           high_price_volume,
                           low_price_volume
                    FROM price_data_5min
                    WHERE item_id = %s
                      AND timestamp > NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp
                """
                df = pd.read_sql(query, conn, params=[item_id, hours])
                if len(df) >= 12:  # Need at least 1 hour of data
                    price_data[item_id] = df

        return price_data

    def compute_features(self, item_id: int, price_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute features for a single item."""
        try:
            features = self.feature_engine.compute_features(price_df)
            if features is not None and len(features) > 0:
                # Deduplicate columns (feature_engine bug produces duplicates)
                features = features.loc[:, ~features.columns.duplicated()]
                # Drop non-numeric columns (timestamp, etc.)
                features = features.select_dtypes(include=[np.number])
                return features.iloc[[-1]]  # Latest row only
            return None
        except Exception as e:
            logger.warning(f"Feature computation failed for item {item_id}: {e}")
            return None

    def predict_item(
        self,
        item_id: int,
        features: pd.DataFrame,
        current_high: float,
        current_low: float
    ) -> List[Dict]:
        """Generate predictions for a single item."""
        if item_id not in self.models:
            return []

        predictions = []
        item_models = self.models[item_id]

        for target_name, model in item_models.items():
            try:
                # Parse target name: buy_fills_1pct_4h
                parts = target_name.rsplit('_', 2)
                if len(parts) != 3:
                    continue

                direction = parts[0]  # buy_fills, sell_fills, roundtrip
                offset_str = parts[1]  # 1pct, 2pct, 3pct
                window = parts[2]  # 4h, 8h, etc.

                offset_pct = int(offset_str.replace('pct', '')) / 100.0
                hour_offset = int(window.replace('h', ''))

                # Get prediction probability
                proba = model.predict_proba(features)
                # Handle various return formats from CatBoost
                if isinstance(proba, np.ndarray):
                    if len(proba.shape) > 1:
                        fill_prob = float(proba[0, 1])
                    else:
                        fill_prob = float(proba[0])
                else:
                    # Handle tuple/list formats
                    fill_prob = float(proba[0][1] if isinstance(proba[0], (list, tuple, np.ndarray)) else proba[0])

                fill_prob, _ = clip_probability(fill_prob)  # Returns (value, was_clipped)

                # Calculate target prices
                if direction == 'buy_fills':
                    buy_price = current_low * (1 - offset_pct)
                    sell_price = current_high
                elif direction == 'sell_fills':
                    buy_price = current_low
                    sell_price = current_high * (1 + offset_pct)
                else:  # roundtrip
                    buy_price = current_low * (1 - offset_pct)
                    sell_price = current_high * (1 + offset_pct)

                # Calculate expected value: probability * profit margin
                expected_value = fill_prob * offset_pct

                predictions.append({
                    'item_id': item_id,
                    'hour_offset': hour_offset,
                    'offset_pct': offset_pct,
                    'direction': direction,
                    'fill_probability': fill_prob,
                    'expected_value': expected_value,
                    'buy_price': float(buy_price),
                    'sell_price': float(sell_price),
                    'model_run_id': self.run_id,
                })

            except Exception as e:
                logger.warning(f"Prediction failed for {item_id}/{target_name}: {e}")

        return predictions

    def run_inference_cycle(self, hours: int = 72, table_name: str = 'predictions') -> int:
        """
        Run full inference cycle: load data, compute features, predict, save.

        Returns:
            Number of predictions saved.
        """
        start = time.time()

        # Load price data
        logger.info("Loading price data...")
        price_data = self.load_recent_prices(hours=hours)
        logger.info(f"Loaded data for {len(price_data)} items")

        if not price_data:
            logger.warning("No price data loaded")
            return 0

        # Generate predictions
        all_predictions = []
        prediction_time = datetime.utcnow()

        for item_id, price_df in price_data.items():
            features = self.compute_features(item_id, price_df)
            if features is None:
                continue

            # Get current prices
            current_high = float(price_df['avg_high_price'].iloc[-1])
            current_low = float(price_df['avg_low_price'].iloc[-1])

            if current_high <= 0 or current_low <= 0:
                continue

            preds = self.predict_item(item_id, features, current_high, current_low)
            for p in preds:
                p['time'] = prediction_time
                p['target_hour'] = prediction_time + timedelta(hours=p['hour_offset'])

            all_predictions.extend(preds)

        logger.info(f"Generated {len(all_predictions)} predictions")

        if not all_predictions:
            return 0

        # Save to database
        saved = self._save_predictions(all_predictions, table_name)

        elapsed = time.time() - start
        logger.info(f"Inference cycle complete: {saved} predictions in {elapsed:.1f}s")

        return saved

    def _save_predictions(self, predictions: List[Dict], table_name: str) -> int:
        """Save predictions to database."""
        if not predictions:
            return 0

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Prepare data for insertion
                rows = [
                    (
                        p['time'],
                        p['item_id'],
                        p['hour_offset'],
                        p['offset_pct'],
                        p['fill_probability'],
                        p['expected_value'],
                        p['buy_price'],
                        p['sell_price'],
                        p['target_hour'],
                    )
                    for p in predictions
                ]

                execute_values(
                    cur,
                    f"""
                    INSERT INTO {table_name}
                        (time, item_id, hour_offset, offset_pct, fill_probability,
                         expected_value, buy_price, sell_price, target_hour)
                    VALUES %s
                    """,
                    rows,
                    page_size=1000
                )
                conn.commit()

        return len(predictions)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    predictor = PerTargetBatchPredictor()
    print(f"Loaded {len(predictor.models)} items")

    # Test run
    count = predictor.run_inference_cycle(hours=72)
    print(f"Generated {count} predictions")
