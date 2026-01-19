"""
Multi-Target Batch Predictor for CatBoost MultiLogloss Models

Designed for the new multi-target model format from train_runpod_multitarget.py:
- Single .cbm model per item with 108 targets
- Target format: seq_{hour}h_{offset}pct (e.g., seq_12h_2pct)
- No scalers needed (CatBoost handles internally)

Usage:
    from batch_predictor_multitarget import MultiTargetBatchPredictor

    # Use latest model run automatically
    predictor = MultiTargetBatchPredictor()

    # Or specify a specific run
    predictor = MultiTargetBatchPredictor(models_dir='models/20260111_142024')

    # Run inference
    stats = predictor.run_inference_cycle()
"""

import os
import io
import re
import json
import time
import logging
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("catboost not available")

from feature_engine import FeatureEngine, Granularity
from target_engine import compute_expected_value_pct
from db_utils import ConnectionPool, get_db_connection, get_db_cursor
from inference_config import clip_probability, get_confidence_tier, DEFAULT_AUC
from calibration import CalibrationManager


def compute_stability_fields(price_history: pd.DataFrame) -> dict:
    """Compute price stability and momentum fields from price history.

    Args:
        price_history: DataFrame with columns ['avg_high_price', 'avg_low_price']
                      indexed by timestamp, sorted ascending.

    Returns:
        Dict with keys: median_14d, price_vs_median_ratio, return_1h,
                       return_4h, return_24h, volatility_24h
    """
    result = {
        'median_14d': None,
        'price_vs_median_ratio': None,
        'return_1h': None,
        'return_4h': None,
        'return_24h': None,
        'volatility_24h': None,
    }

    if price_history.empty or len(price_history) < 2:
        return result

    # Compute midpoint prices
    if 'avg_high_price' in price_history.columns and 'avg_low_price' in price_history.columns:
        mids = (price_history['avg_high_price'] + price_history['avg_low_price']) / 2
    else:
        return result

    # Drop NaN values for calculations
    mids = mids.dropna()
    if len(mids) < 2:
        return result

    current_mid = mids.iloc[-1]

    # Median 14d: requires at least 7 days of data (168 hours)
    if len(mids) >= 168:
        # Use last 14 days (336 hours) or all available
        lookback = min(len(mids), 336)
        result['median_14d'] = float(mids.iloc[-lookback:].median())
        if result['median_14d'] > 0:
            result['price_vs_median_ratio'] = float(current_mid / result['median_14d'])

    # Returns: compute from available data
    def safe_return(periods: int) -> float | None:
        if len(mids) > periods:
            old_price = mids.iloc[-(periods + 1)]
            if old_price > 0:
                return float((current_mid - old_price) / old_price)
        return None

    result['return_1h'] = safe_return(1)  # 1 hour ago (assuming hourly data)
    result['return_4h'] = safe_return(4)
    result['return_24h'] = safe_return(24)

    # Volatility 24h: std of hourly returns over last 24 hours
    if len(mids) >= 25:
        returns = mids.pct_change().dropna()
        if len(returns) >= 24:
            result['volatility_24h'] = float(returns.iloc[-24:].std())

    return result


def get_model_trained_at(run_id: str) -> Optional[datetime]:
    """Parse model training timestamp from run_id (YYYYMMDD_HHMMSS format)."""
    try:
        return datetime.strptime(run_id, '%Y%m%d_%H%M%S')
    except ValueError:
        return None


class InferenceStatusTracker:
    """Track inference run status for freshness verification."""

    def __init__(self, model_run_id: str):
        self.model_run_id = model_run_id
        self.model_trained_at = get_model_trained_at(model_run_id)
        self.status_id: Optional[int] = None
        self.run_id: Optional[str] = None

    def start_run(self) -> int:
        """Record start of inference run. Returns status_id."""
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        with get_db_cursor() as cur:
            cur.execute("""
                INSERT INTO inference_status
                (inference_started_at, run_id, model_run_id, model_trained_at, status)
                VALUES (NOW(), %s, %s, %s, 'running')
                RETURNING id
            """, (self.run_id, self.model_run_id, self.model_trained_at))
            self.status_id = cur.fetchone()[0]

        logger.info(f"Inference run {self.run_id} started (status_id={self.status_id})")
        return self.status_id

    def complete_run(self, items_predicted: int, predictions_written: int):
        """Record successful completion of inference run."""
        if self.status_id is None:
            return

        with get_db_cursor() as cur:
            cur.execute("""
                UPDATE inference_status
                SET inference_completed_at = NOW(),
                    items_predicted = %s,
                    predictions_written = %s,
                    status = 'completed'
                WHERE id = %s
            """, (items_predicted, predictions_written, self.status_id))

        logger.info(f"Inference run {self.run_id} completed: "
                    f"{items_predicted} items, {predictions_written} predictions")

    def fail_run(self, error_message: str):
        """Record failed inference run."""
        if self.status_id is None:
            return

        # Truncate error message if too long
        if len(error_message) > 1000:
            error_message = error_message[:997] + '...'

        with get_db_cursor() as cur:
            cur.execute("""
                UPDATE inference_status
                SET inference_completed_at = NOW(),
                    status = 'failed',
                    error_message = %s
                WHERE id = %s
            """, (error_message, self.status_id))

        logger.error(f"Inference run {self.run_id} failed: {error_message[:100]}")


def find_latest_model_run(models_base: str = 'models') -> Optional[str]:
    """Find the latest model run directory by timestamp."""
    models_path = Path(models_base)
    if not models_path.exists():
        return None

    # Look for directories matching YYYYMMDD_HHMMSS pattern
    run_pattern = re.compile(r'^\d{8}_\d{6}$')
    runs = [
        d for d in models_path.iterdir()
        if d.is_dir() and run_pattern.match(d.name)
    ]

    if not runs:
        return None

    # Sort by name (lexicographic = chronological for this format)
    latest = sorted(runs, key=lambda x: x.name)[-1]
    return str(latest)


class MultiTargetBatchPredictor:
    """
    Batch predictor for CatBoost MultiLogloss models with 108 targets per item.

    The new model format has:
    - models/<run_id>/<item_id>/model.cbm - Single multi-output model
    - models/<run_id>/<item_id>/meta.json - Metadata (feature_cols, target_names)
    - models/<run_id>/<item_id>/target_aucs.json - Per-target AUC scores
    """

    def __init__(self, models_dir: Optional[str] = None, use_calibration: bool = True):
        """
        Initialize the predictor.

        Args:
            models_dir: Path to models directory. If None, finds latest run in 'models/'
            use_calibration: Whether to apply calibration to predictions (default: True)
        """
        if models_dir is None:
            models_dir = find_latest_model_run()
            if models_dir is None:
                raise FileNotFoundError("No model runs found in 'models/' directory")
            logger.info(f"Using latest model run: {models_dir}")

        self.models_dir = models_dir
        self.use_calibration = use_calibration
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self.items: List[Dict] = []

        # Extract run_id from models_dir path (e.g., models/20260111_142024 -> 20260111_142024)
        self.run_id = Path(self.models_dir).name

        # Model storage: item_id -> (model, meta)
        self.models: Dict[int, Tuple[CatBoostClassifier, Dict]] = {}

        # Target AUCs: item_id -> {target_name: auc}
        self.target_aucs: Dict[int, Dict[str, float]] = {}

        # Model IDs from registry: item_id -> model_id (for tracking in predictions)
        self.model_ids: Dict[int, int] = {}

        # Calibration managers: item_id -> CalibrationManager
        self.calibrators: Dict[int, CalibrationManager] = {}

        self._load_all_models()
        self._load_model_ids()
        if self.use_calibration:
            self._load_calibration()

    def _load_all_models(self):
        """Load all models into memory at startup."""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        loaded = 0
        skipped = 0

        for item_dir in os.listdir(self.models_dir):
            item_path = Path(self.models_dir) / item_dir

            # Skip non-directories and special files
            if not item_path.is_dir():
                continue

            try:
                item_id = int(item_dir)
            except ValueError:
                continue

            model_path = item_path / 'model.cbm'
            meta_path = item_path / 'meta.json'

            if not model_path.exists() or not meta_path.exists():
                skipped += 1
                continue

            try:
                # Load metadata
                with open(meta_path) as f:
                    meta = json.load(f)

                # Validate target count matches expected 108
                target_names = meta.get('target_names', [])
                if len(target_names) != 108:
                    logger.warning(f"Model for item {item_id} has {len(target_names)} targets, "
                                   f"expected 108. Skipping.")
                    skipped += 1
                    continue

                # Load model
                model = CatBoostClassifier()
                model.load_model(str(model_path))

                item_name = meta.get('item_name', f'Item-{item_id}')
                self.items.append({'item_id': item_id, 'item_name': item_name})
                self.models[item_id] = (model, meta)

                # Load target AUCs if available
                aucs_path = item_path / 'target_aucs.json'
                if aucs_path.exists():
                    with open(aucs_path) as f:
                        aucs_list = json.load(f)
                        self.target_aucs[item_id] = {
                            a['target']: a['auc'] for a in aucs_list if a.get('auc') is not None
                        }

                loaded += 1

            except Exception as e:
                logger.warning(f"Failed to load model for item {item_id}: {e}")
                skipped += 1

        logger.info(f"Loaded {loaded} multi-target models ({skipped} skipped)")

    def _load_model_ids(self):
        """
        Load model IDs from model_registry with validation against loaded disk models.

        Validates that:
        1. ACTIVE registry models match current run_id
        2. Disk models have corresponding ACTIVE registry entries
        3. Registry entries point to models that exist on disk

        This ensures predictions always reference the correct model_id for the
        deployed run, and logs clear errors if registry and disk are out of sync.
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Query ACTIVE models for this specific run_id
                    cur.execute("""
                        SELECT item_id, id as model_id, model_path
                        FROM model_registry
                        WHERE status = 'ACTIVE' AND run_id = %s
                    """, (self.run_id,))

                    registry_models = {}
                    for row in cur.fetchall():
                        item_id, model_id, model_path = row
                        registry_models[item_id] = {
                            'model_id': model_id,
                            'model_path': model_path
                        }

            # Validate alignment between disk and registry
            disk_items = set(self.models.keys())
            registry_items = set(registry_models.keys())

            # Models on disk with ACTIVE registry entry for this run
            matched = disk_items & registry_items
            # Models on disk but NOT ACTIVE in registry for this run
            disk_only = disk_items - registry_items
            # ACTIVE in registry for this run but NOT loaded from disk
            registry_only = registry_items - disk_items

            # Assign model_ids only for matched models
            for item_id in matched:
                self.model_ids[item_id] = registry_models[item_id]['model_id']

            # Report validation status
            logger.info(f"Registry validation for run_id={self.run_id}: "
                        f"Matched (disk+registry): {len(matched)}")

            if disk_only:
                logger.warning(f"{len(disk_only)} models on disk but NOT ACTIVE in registry")
                if len(disk_only) <= 10:
                    logger.warning(f"  Items: {sorted(disk_only)}")
                else:
                    logger.warning(f"  First 10 items: {sorted(disk_only)[:10]}...")

            if registry_only:
                logger.warning(f"{len(registry_only)} ACTIVE in registry but NOT on disk")
                if len(registry_only) <= 10:
                    logger.warning(f"  Items: {sorted(registry_only)}")
                else:
                    logger.warning(f"  First 10 items: {sorted(registry_only)[:10]}...")

        except Exception as e:
            # Model registry may not exist yet - that's okay for initial setup
            logger.info(f"Could not validate against registry: {e}")

    def _load_calibration(self):
        """Load calibration data for all models with calibration.json files."""
        loaded = 0
        for item_id in self.models.keys():
            item_path = Path(self.models_dir) / str(item_id)
            calib_path = item_path / 'calibration.json'

            if calib_path.exists():
                calibrator = CalibrationManager.load(str(calib_path))
                if calibrator is not None:
                    self.calibrators[item_id] = calibrator
                    loaded += 1

        if loaded > 0:
            logger.info(f"Loaded calibration for {loaded} items")
        else:
            logger.info("No calibration files found (run scripts/train_calibration.py)")

    def load_recent_prices(self, hours: int = 72) -> Dict[int, pd.DataFrame]:
        """Load recent price data using connection pool."""
        conn = ConnectionPool.get_conn()
        try:
            item_ids = [item['item_id'] for item in self.items]

            query = """
                SELECT item_id, timestamp, avg_high_price, avg_low_price,
                       high_price_volume, low_price_volume
                FROM price_data_5min
                WHERE item_id = ANY(%s)
                  AND timestamp >= NOW() - make_interval(hours => %s)
                  AND avg_high_price IS NOT NULL
                  AND avg_low_price IS NOT NULL
                ORDER BY item_id, timestamp
            """

            df = pd.read_sql(query, conn, params=[item_ids, hours])

            price_data = {}
            for item_id in item_ids:
                item_df = df[df['item_id'] == item_id].copy()
                if len(item_df) > 0:
                    price_data[item_id] = item_df.reset_index(drop=True)

            return price_data
        finally:
            ConnectionPool.put_conn(conn)

    def _parse_target_name(self, target_name: str) -> Tuple[Optional[int], Optional[float]]:
        """
        Parse target name to extract hour and offset.

        Supports formats:
        - seq_12h_2pct -> (12, 0.02)
        - seq_1h_1.25pct -> (1, 0.0125)
        """
        match = re.match(r'seq_(\d+)h_(\d+\.?\d*)pct', target_name)
        if match:
            hour = int(match.group(1))
            offset = float(match.group(2)) / 100
            return hour, offset
        return None, None

    def _get_feature_value(self, latest: pd.Series, col: str) -> Tuple[float, str]:
        """
        Extract feature value with validation.

        Returns:
            Tuple of (value, status) where:
            - value: The float value or np.nan if invalid
            - status: 'valid', 'missing', 'nan', or 'inf'
        """
        if col not in latest.index:
            return np.nan, 'missing'

        val = latest[col]
        if pd.isna(val):
            return np.nan, 'nan'
        if np.isinf(val):
            return np.nan, 'inf'

        return float(val), 'valid'

    def predict_item(
        self,
        item_id: int,
        features_df: pd.DataFrame,
        prediction_time: datetime,
        hour_start: int = 1,
        hour_end: int = 48
    ) -> Tuple[List[Dict], int, int, int]:
        """
        Generate predictions for an item using the multi-target model.

        This runs all 108 targets in a single forward pass, which is much
        faster than the old per-target approach.

        Returns:
            tuple: (predictions list, clipped_count, missing_features, invalid_features)
        """
        predictions = []
        clipped_count = 0
        missing_count = 0
        invalid_count = 0

        if item_id not in self.models:
            return predictions, clipped_count, missing_count, invalid_count

        model, meta = self.models[item_id]
        feature_cols = meta.get('feature_cols', [])
        target_names = meta.get('target_names', [])

        if not feature_cols or not target_names:
            return predictions, clipped_count, missing_count, invalid_count

        if len(features_df) == 0:
            return predictions, clipped_count, missing_count, invalid_count

        latest = features_df.iloc[-1]

        # Build feature vector with validation tracking
        feature_values = []
        for col in feature_cols:
            val, status = self._get_feature_value(latest, col)
            if status == 'missing':
                missing_count += 1
            elif status in ('nan', 'inf'):
                invalid_count += 1
            feature_values.append(val)

        # Log if significant portion of features are invalid (>10%)
        total_invalid = missing_count + invalid_count
        if total_invalid > len(feature_cols) * 0.1:
            logger.debug(
                f"Item {item_id}: {total_invalid}/{len(feature_cols)} features "
                f"invalid (missing={missing_count}, nan/inf={invalid_count})"
            )

        # Replace NaN with 0.0 for model compatibility (after tracking)
        X = np.array([0.0 if np.isnan(v) else v for v in feature_values]).reshape(1, -1)

        current_high = latest.get('avg_high_price', 0)
        current_low = latest.get('avg_low_price', 0)

        # Skip items with NaN or invalid prices
        if pd.isna(current_high) or pd.isna(current_low) or current_high <= 0 or current_low <= 0:
            logger.debug(
                f"Item {item_id}: Skipping due to invalid prices "
                f"(high={current_high}, low={current_low})"
            )
            return predictions, clipped_count, missing_count, invalid_count

        item_name = next(
            (i['item_name'] for i in self.items if i['item_id'] == item_id),
            f'Item-{item_id}'
        )

        # Run inference - single forward pass for all 108 targets
        try:
            proba = model.predict_proba(X)
        except Exception as e:
            logger.warning(f"Prediction failed for item {item_id}: {e}")
            return predictions, clipped_count, missing_count, invalid_count

        n_targets = len(target_names)

        # Validate model output shape matches expected dimensions
        expected_multilogloss = 2 * n_targets  # MultiLogloss: [neg0, pos0, neg1, pos1, ...]
        expected_direct = n_targets            # Direct probability format

        if proba.shape[1] != expected_multilogloss and proba.shape[1] != expected_direct:
            logger.error(f"Model output shape mismatch for item {item_id}: "
                         f"got {proba.shape[1]} columns, expected {expected_multilogloss} "
                         f"(MultiLogloss) or {expected_direct} (direct). Skipping item.")
            return predictions, clipped_count, missing_count, invalid_count

        # Handle MultiLogloss output format: (N, 2*K) -> [neg0, pos0, neg1, pos1, ...]
        if proba.shape[1] == expected_multilogloss:
            # Extract positive class probabilities
            probabilities = {
                target_names[i]: float(proba[0, 2 * i + 1])
                for i in range(n_targets)
            }
        else:
            # Direct probability format (single column per target)
            # Shape is already validated - no fallback needed
            probabilities = {
                target_names[i]: float(proba[0, i])
                for i in range(n_targets)
            }

        # Get target AUCs for confidence scoring
        item_aucs = self.target_aucs.get(item_id, {})

        # Get calibrator for this item if available
        calibrator = self.calibrators.get(item_id) if self.use_calibration else None

        # Generate prediction records for each target in hour range
        for target_name, prob in probabilities.items():
            hour, offset = self._parse_target_name(target_name)

            if hour is None or offset is None:
                continue

            if hour < hour_start or hour > hour_end:
                continue

            # Apply calibration if available
            if calibrator is not None:
                prob = calibrator.calibrate(hour, prob)

            # Clip probability to valid bounds
            clipped_prob, was_clipped = clip_probability(prob)
            if was_clipped:
                clipped_count += 1

            # Compute buy/sell prices
            buy_price = current_low * (1 - offset)
            sell_price = current_high * (1 + offset)

            # Compute expected value with accurate GP-based tax
            ev = compute_expected_value_pct(clipped_prob, buy_price, sell_price)

            # Confidence based on model AUC for this target (Issue #70)
            auc = item_aucs.get(target_name, DEFAULT_AUC)
            confidence = get_confidence_tier(auc)

            predictions.append({
                'time': prediction_time,
                'item_id': item_id,
                'item_name': item_name,
                'hour_offset': hour,
                'target_hour': prediction_time + timedelta(hours=hour),
                'offset_pct': offset,
                'fill_probability': float(clipped_prob),
                'expected_value': float(ev),
                'buy_price': float(buy_price),
                'sell_price': float(sell_price),
                'current_high': float(current_high) if current_high else None,
                'current_low': float(current_low) if current_low else None,
                'confidence': confidence,
                'model_id': self.model_ids.get(item_id)  # Link to model_registry
            })

        return predictions, clipped_count, missing_count, invalid_count

    def save_to_timescale_copy(self, predictions: List[Dict], table_name: str = 'predictions'):
        """Write predictions using COPY protocol (faster than INSERT)."""
        if not predictions:
            return

        conn = ConnectionPool.get_conn()
        try:
            cur = conn.cursor()

            def format_float(val):
                """Format float value for COPY, handling None."""
                return str(val) if val is not None else '\\N'

            buffer = io.StringIO()
            for p in predictions:
                row = [
                    p['time'].isoformat(),
                    str(p['item_id']),
                    p['item_name'].replace('\t', ' '),
                    str(p['hour_offset']),
                    p['target_hour'].isoformat(),
                    str(p['offset_pct']),
                    str(p['fill_probability']),
                    str(p['expected_value']),
                    str(p['buy_price']),
                    str(p['sell_price']),
                    str(p['current_high']) if p['current_high'] else '\\N',
                    str(p['current_low']) if p['current_low'] else '\\N',
                    p['confidence'] or '\\N',
                    str(p['model_id']) if p.get('model_id') else '\\N',
                    # Stability fields
                    format_float(p.get('median_14d')),
                    format_float(p.get('price_vs_median_ratio')),
                    format_float(p.get('return_1h')),
                    format_float(p.get('return_4h')),
                    format_float(p.get('return_24h')),
                    format_float(p.get('volatility_24h')),
                ]
                buffer.write('\t'.join(row) + '\n')

            buffer.seek(0)

            cur.copy_from(buffer, table_name,
                          columns=('time', 'item_id', 'item_name', 'hour_offset',
                                   'target_hour', 'offset_pct', 'fill_probability',
                                   'expected_value', 'buy_price', 'sell_price',
                                   'current_high', 'current_low', 'confidence',
                                   'model_id', 'median_14d', 'price_vs_median_ratio',
                                   'return_1h', 'return_4h', 'return_24h', 'volatility_24h'))

            conn.commit()
            cur.close()
        finally:
            ConnectionPool.put_conn(conn)

    def save_to_timescale(self, predictions: List[Dict], verbose: bool = True, table_name: str = 'predictions'):
        """Write predictions to TimescaleDB (standard INSERT)."""
        if not predictions:
            return

        conn = ConnectionPool.get_conn()
        try:
            cur = conn.cursor()

            values = [
                (p['time'], p['item_id'], p['item_name'], p['hour_offset'],
                 p['target_hour'], p['offset_pct'], p['fill_probability'],
                 p['expected_value'], p['buy_price'], p['sell_price'],
                 p['current_high'], p['current_low'], p['confidence'],
                 p.get('model_id'),
                 p.get('median_14d'), p.get('price_vs_median_ratio'),
                 p.get('return_1h'), p.get('return_4h'), p.get('return_24h'),
                 p.get('volatility_24h'))
                for p in predictions
            ]

            execute_values(cur, f"""
                INSERT INTO {table_name} (time, item_id, item_name, hour_offset,
                    target_hour, offset_pct, fill_probability, expected_value,
                    buy_price, sell_price, current_high, current_low, confidence,
                    model_id, median_14d, price_vs_median_ratio,
                    return_1h, return_4h, return_24h, volatility_24h)
                VALUES %s
            """, values)

            conn.commit()
            cur.close()

            if verbose:
                logger.info(f"Saved {len(predictions)} predictions to {table_name}")
        finally:
            ConnectionPool.put_conn(conn)

    def compute_all_features(self, price_data: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
        """Pre-compute features for all items."""
        features_cache = {}
        for item in self.items:
            item_id = item['item_id']
            if item_id not in price_data:
                continue
            try:
                features_df = self.feature_engine.compute_features(price_data[item_id].copy())
                features_cache[item_id] = features_df
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Feature computation failed for item {item_id}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error computing features for item {item_id}: {e}")
                continue
        return features_cache

    def _compute_volatility(self, item_id: int, price_data: Dict[int, pd.DataFrame]) -> float:
        """Calculate recent price volatility for prioritization."""
        if item_id not in price_data:
            return 0.0
        df = price_data[item_id]
        if len(df) < 24:
            return 0.0
        prices = df['avg_high_price'].dropna()
        if len(prices) < 24:
            return 0.0
        returns = prices.pct_change(12).tail(48)
        vol = returns.std()
        return float(vol) if not np.isnan(vol) else 0.0

    def run_inference_cycle(self, hours_history: int = 72, use_copy: bool = True,
                            table_name: str = 'predictions', track_status: bool = True) -> Dict:
        """
        Run complete inference cycle.

        Args:
            hours_history: Hours of price data to load
            use_copy: Use COPY protocol for faster writes
            table_name: Target table ('predictions' or 'predictions_staging')
            track_status: Record run status to inference_status table (default: True)

        Returns stats dict with timing and count information.
        """
        start = time.time()

        # Initialize status tracker
        status_tracker = None
        if track_status:
            try:
                status_tracker = InferenceStatusTracker(self.run_id)
                status_tracker.start_run()
            except Exception as e:
                logger.warning(f"Failed to start status tracking: {e}")
                status_tracker = None

        try:
            # Load prices
            logger.info("Loading price data...")
            price_data = self.load_recent_prices(hours_history)
            load_time = time.time() - start
            logger.info(f"Loaded {len(price_data)} items in {load_time:.1f}s")

            # Compute features
            logger.info("Computing features...")
            feature_start = time.time()
            features_cache = self.compute_all_features(price_data)
            feature_time = time.time() - feature_start
            logger.info(f"Computed features for {len(features_cache)} items in {feature_time:.1f}s")

            # Compute stability fields for each item
            logger.info("Computing stability fields...")
            stability_start = time.time()
            stability_cache = {}
            for item_id, df in price_data.items():
                stability_cache[item_id] = compute_stability_fields(df)
            stability_time = time.time() - stability_start
            logger.info(f"Computed stability fields for {len(stability_cache)} items in {stability_time:.2f}s")

            # Sort by volatility (high volatility items first)
            volatilities = {
                item['item_id']: self._compute_volatility(item['item_id'], price_data)
                for item in self.items if item['item_id'] in price_data
            }
            sorted_items = sorted(
                [item for item in self.items if item['item_id'] in price_data],
                key=lambda x: volatilities.get(x['item_id'], 0),
                reverse=True
            )

            # Process by time tier (prioritize near-term predictions)
            prediction_time = datetime.now()
            hour_tiers = [
                (1, 4, 'immediate'),
                (5, 8, 'short'),
                (9, 16, 'medium'),
                (17, 24, 'long'),
                (25, 48, 'extended')
            ]

            stats = {
                'tiers': [],
                'total_predictions': 0,
                'total_clipped': 0,
                'total_missing_features': 0,
                'total_invalid_features': 0
            }

            logger.info("Generating predictions (multi-target, prioritized)...")
            for hour_start, hour_end, tier_name in hour_tiers:
                tier_start = time.time()
                tier_predictions = []
                tier_clipped = 0
                tier_missing = 0
                tier_invalid = 0

                for item in sorted_items:
                    item_id = item['item_id']
                    if item_id not in features_cache:
                        continue

                    preds, clipped, missing, invalid = self.predict_item(
                        item_id, features_cache[item_id], prediction_time, hour_start, hour_end
                    )

                    # Add stability fields to each prediction
                    stability = stability_cache.get(item_id, {})
                    for pred in preds:
                        pred['median_14d'] = stability.get('median_14d')
                        pred['price_vs_median_ratio'] = stability.get('price_vs_median_ratio')
                        pred['return_1h'] = stability.get('return_1h')
                        pred['return_4h'] = stability.get('return_4h')
                        pred['return_24h'] = stability.get('return_24h')
                        pred['volatility_24h'] = stability.get('volatility_24h')

                    tier_predictions.extend(preds)
                    tier_clipped += clipped
                    tier_missing += missing
                    tier_invalid += invalid

                tier_predict_time = time.time() - tier_start

                if tier_predictions:
                    save_start = time.time()
                    if use_copy:
                        self.save_to_timescale_copy(tier_predictions, table_name=table_name)
                    else:
                        self.save_to_timescale(tier_predictions, verbose=False, table_name=table_name)
                    save_time = time.time() - save_start
                else:
                    save_time = 0

                stats['tiers'].append({
                    'tier': tier_name,
                    'hours': f'{hour_start}-{hour_end}',
                    'predictions': len(tier_predictions),
                    'clipped': tier_clipped,
                    'missing_features': tier_missing,
                    'invalid_features': tier_invalid,
                    'predict_time': tier_predict_time,
                    'save_time': save_time
                })
                stats['total_predictions'] += len(tier_predictions)
                stats['total_clipped'] += tier_clipped
                stats['total_missing_features'] += tier_missing
                stats['total_invalid_features'] += tier_invalid

                logger.info(f"Tier {tier_name} (hours {hour_start}-{hour_end}): "
                            f"{len(tier_predictions)} preds, "
                            f"infer={tier_predict_time:.2f}s, save={save_time:.2f}s")

            stats['items'] = len(sorted_items)
            stats['load_time'] = load_time
            stats['feature_time'] = feature_time
            stats['stability_time'] = stability_time
            stats['total_time'] = time.time() - start

            # Record successful completion
            if status_tracker:
                status_tracker.complete_run(stats['items'], stats['total_predictions'])

            return stats

        except Exception as e:
            # Record failure
            if status_tracker:
                status_tracker.fail_run(str(e))
            raise


def main():
    """Test the multi-target batch predictor."""
    # Configure logging for main execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("Initializing MultiTargetBatchPredictor...")
    predictor = MultiTargetBatchPredictor()

    logger.info(f"Models loaded: {len(predictor.models)}")
    logger.info(f"Items: {len(predictor.items)}")

    # Sample model info
    if predictor.models:
        sample_id = next(iter(predictor.models.keys()))
        _, meta = predictor.models[sample_id]
        logger.info(f"Sample model (item {sample_id}): "
                    f"Features={meta.get('n_features', 'unknown')}, "
                    f"Targets={meta.get('n_targets', 'unknown')}, "
                    f"Mean AUC={meta.get('mean_auc', 'unknown'):.4f}")

    logger.info("Running inference cycle...")
    stats = predictor.run_inference_cycle()

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Items: {stats['items']}")
    logger.info(f"Predictions: {stats['total_predictions']}")
    logger.info(f"Total time: {stats['total_time']:.1f}s")


if __name__ == "__main__":
    main()
