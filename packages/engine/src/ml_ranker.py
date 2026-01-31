"""ML-based ranking for trade recommendations.

This module provides a CatBoost-based ranker that scores trade candidates
using prediction metadata, item properties, and market context features.

The ranker is designed for shadow mode deployment: it runs alongside the
existing heuristic scoring but doesn't affect actual recommendations until
validated.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import CatBoost - graceful fallback if not available
try:
    from catboost import CatBoost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available - ML ranker disabled")


class MLRanker:
    """CatBoost-based ranker for trade recommendations.

    Loads a pre-trained model and provides inference-time scoring
    for trade candidates based on prediction features.
    """

    # Feature columns expected by the model (must match training)
    FEATURE_COLUMNS = [
        # Item properties
        'log_buy_limit',
        'log_avg_price',

        # Market context
        'spread_pct',
        'volatility_24h',
        'log_volume_24h',

        # Time features
        'hour_sin', 'hour_cos',
        'is_weekend',

        # Prediction features
        'pred_fill_prob_mean',
        'pred_fill_prob_max',
        'pred_ev_mean',
        'pred_ev_max',
        'pred_fill_prob_1h',
        'pred_fill_prob_24h',
        'pred_ev_1h',
        'pred_ev_24h',
    ]

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the ML ranker.

        Args:
            model_path: Path to the CatBoost model file (.cbm).
                       If None, uses default path.
        """
        self.model = None
        self.model_loaded = False
        self.model_path = model_path

        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost not installed - ML ranker disabled")
            return

        self._load_model()

    def _load_model(self):
        """Load the CatBoost model from disk."""
        if self.model_path is None:
            # Default paths to check
            default_paths = [
                Path.home() / "gept" / "models" / "ranker" / "ranker_v3.cbm",
                Path("/home/ubuntu/gept/models/ranker/ranker_v3.cbm"),
                Path("./models/ranker/ranker_v3.cbm"),
            ]
            for path in default_paths:
                if path.exists():
                    self.model_path = str(path)
                    break

        if self.model_path is None or not Path(self.model_path).exists():
            logger.warning(f"Model not found at {self.model_path} - ML ranker disabled")
            return

        try:
            self.model = CatBoost()
            self.model.load_model(self.model_path)
            self.model_loaded = True
            logger.info(f"Loaded ML ranker model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML ranker model: {e}")
            self.model = None
            self.model_loaded = False

    def is_available(self) -> bool:
        """Check if the ML ranker is available and loaded."""
        return CATBOOST_AVAILABLE and self.model_loaded and self.model is not None

    def score_candidates(
        self,
        candidates: list[dict],
        prediction_features: dict[int, dict],
    ) -> list[dict]:
        """Score candidates using the ML ranker.

        Args:
            candidates: List of candidate dicts from _build_candidate
            prediction_features: Dict mapping item_id to prediction features
                                (from MLFeatureBuilder)

        Returns:
            Candidates with added 'ml_score' field
        """
        if not self.is_available():
            # Return candidates unchanged with null ML scores
            for cand in candidates:
                cand['ml_score'] = None
            return candidates

        # Build feature matrix
        features = []
        valid_indices = []

        for i, cand in enumerate(candidates):
            item_id = cand.get('item_id')

            # Get prediction features for this item
            pred_feats = prediction_features.get(item_id, {})

            # Build feature vector
            feat_vector = self._build_feature_vector(cand, pred_feats)

            if feat_vector is not None:
                features.append(feat_vector)
                valid_indices.append(i)

        if not features:
            for cand in candidates:
                cand['ml_score'] = None
            return candidates

        # Run inference
        X = np.array(features)
        try:
            scores = self.model.predict(X)
        except Exception as e:
            logger.error(f"ML ranker inference failed: {e}")
            for cand in candidates:
                cand['ml_score'] = None
            return candidates

        # Assign scores to candidates
        score_map = {valid_indices[i]: scores[i] for i in range(len(scores))}

        for i, cand in enumerate(candidates):
            cand['ml_score'] = float(score_map.get(i)) if i in score_map else None

        return candidates

    def _build_feature_vector(
        self,
        candidate: dict,
        pred_features: dict,
    ) -> Optional[list]:
        """Build a feature vector for a single candidate.

        Args:
            candidate: Candidate dict with item properties and market data
            pred_features: Prediction features for this item

        Returns:
            Feature vector as list, or None if missing required features
        """
        now = datetime.utcnow()

        # Item properties
        buy_limit = candidate.get('buy_limit', 0) or 0
        buy_price = candidate.get('buy_price', 0) or 0

        # Market context from candidate
        spread_pct = candidate.get('spread_pct', 0) or 0
        volatility_24h = candidate.get('volatility_24h', 0) or 0
        volume_24h = candidate.get('volume_24h', 0) or 0

        # Time features
        hour_of_day = now.hour
        is_weekend = 1 if now.weekday() >= 5 else 0

        # Compute cyclical time features
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)

        # Prediction features
        pred_fill_prob_mean = pred_features.get('pred_fill_prob_mean', 0) or 0
        pred_fill_prob_max = pred_features.get('pred_fill_prob_max', 0) or 0
        pred_ev_mean = pred_features.get('pred_ev_mean', 0) or 0
        pred_ev_max = pred_features.get('pred_ev_max', 0) or 0
        pred_fill_prob_1h = pred_features.get('pred_fill_prob_1h', 0) or 0
        pred_fill_prob_24h = pred_features.get('pred_fill_prob_24h', 0) or 0
        pred_ev_1h = pred_features.get('pred_ev_1h', 0) or 0
        pred_ev_24h = pred_features.get('pred_ev_24h', 0) or 0

        # Build feature vector (order must match FEATURE_COLUMNS)
        return [
            np.log1p(buy_limit),
            np.log1p(buy_price),
            spread_pct,
            volatility_24h,
            np.log1p(volume_24h),
            hour_sin,
            hour_cos,
            is_weekend,
            pred_fill_prob_mean,
            pred_fill_prob_max,
            pred_ev_mean,
            pred_ev_max,
            pred_fill_prob_1h,
            pred_fill_prob_24h,
            pred_ev_1h,
            pred_ev_24h,
        ]


class MLFeatureBuilder:
    """Builds ML features from predictions at inference time.

    Aggregates prediction data across hour_offset/offset_pct combinations
    to create features that match the training data format.
    """

    def __init__(self, prediction_loader):
        """Initialize feature builder.

        Args:
            prediction_loader: PredictionLoader instance for fetching predictions
        """
        self.loader = prediction_loader

    def build_features_for_items(self, item_ids: list[int]) -> dict[int, dict]:
        """Build prediction features for a list of items.

        Args:
            item_ids: List of item IDs to build features for

        Returns:
            Dict mapping item_id to feature dict
        """
        if not item_ids:
            return {}

        # Fetch all predictions for these items
        predictions_df = self.loader.get_predictions_for_items(item_ids)

        if predictions_df.empty:
            return {}

        # Aggregate features per item
        result = {}
        for item_id in item_ids:
            item_preds = predictions_df[predictions_df['item_id'] == item_id]
            if len(item_preds) > 0:
                result[item_id] = self._aggregate_predictions(item_preds)

        return result

    def _aggregate_predictions(self, preds_df) -> dict:
        """Aggregate predictions for a single item into features.

        Args:
            preds_df: DataFrame of predictions for one item

        Returns:
            Feature dict matching training data format
        """
        features = {}

        # Overall statistics
        if 'fill_probability' in preds_df.columns:
            features['pred_fill_prob_mean'] = float(preds_df['fill_probability'].mean())
            features['pred_fill_prob_max'] = float(preds_df['fill_probability'].max())
            features['pred_fill_prob_min'] = float(preds_df['fill_probability'].min())
            features['pred_fill_prob_std'] = float(preds_df['fill_probability'].std())

        if 'expected_value' in preds_df.columns:
            features['pred_ev_mean'] = float(preds_df['expected_value'].mean())
            features['pred_ev_max'] = float(preds_df['expected_value'].max())
            features['pred_ev_min'] = float(preds_df['expected_value'].min())
            features['pred_ev_std'] = float(preds_df['expected_value'].std())

        # Features at specific hour offsets (using 2% offset as primary)
        offset_pct_target = 0.02
        for hour in [1, 4, 12, 24]:
            if 'hour_offset' in preds_df.columns and 'offset_pct' in preds_df.columns:
                subset = preds_df[
                    (preds_df['hour_offset'] == hour) &
                    (abs(preds_df['offset_pct'] - offset_pct_target) < 0.005)
                ]
                if len(subset) > 0:
                    row = subset.iloc[0]
                    if 'fill_probability' in row:
                        features[f'pred_fill_prob_{hour}h'] = float(row['fill_probability'])
                    if 'expected_value' in row:
                        features[f'pred_ev_{hour}h'] = float(row['expected_value'])

        return features
