"""
Model Training Pipeline for GE Flipping Predictions

Trains per-item logistic regression models with proper calibration.
Focus is on fill probability prediction (not price direction).
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)
import warnings
warnings.filterwarnings('ignore')

from feature_engine import FeatureEngine, Granularity
from target_engine import (
    TargetEngine, TargetConfig, compute_expected_value,
    DiscreteHourTargetEngine, DiscreteHourConfig
)
from db_utils import get_simple_connection


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data split ratios (temporal splits)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Target configuration
    target_offsets: List[float] = field(default_factory=lambda: [0.01, 0.015, 0.02, 0.025, 0.03])
    target_windows_hours: List[int] = field(default_factory=lambda: [8, 12, 24, 48])

    # Model configuration
    regularization_C: float = 1.0
    max_iter: int = 1000
    calibration_method: str = 'isotonic'  # or 'sigmoid'

    # Validation thresholds
    min_accuracy: float = 0.52  # Must beat baseline
    min_roc_auc: float = 0.50   # Better than random
    max_brier_score: float = 0.30  # Calibration quality

    # Feature handling
    drop_first_n_rows: int = 300  # Allow features to warm up


@dataclass
class DiscreteHourTrainingConfig:
    """Configuration for discrete hour model training."""
    # Date range for training data
    start_date: str = '2025-06-15'
    end_date: str = '2026-01-06'

    # Data split ratios (temporal splits)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Target configuration (discrete hours)
    target_offsets: List[float] = field(default_factory=lambda: [0.02, 0.025])
    discrete_hours: List[int] = field(default_factory=lambda: list(range(1, 25)))

    # Model configuration
    regularization_C: float = 1.0
    max_iter: int = 1000
    calibration_method: str = 'isotonic'

    # Validation thresholds
    min_accuracy: float = 0.52
    min_roc_auc: float = 0.50
    max_brier_score: float = 0.30

    # Feature handling
    drop_first_n_rows: int = 300


@dataclass
class ModelArtifact:
    """Container for a trained model and its metadata."""
    item_id: int
    item_name: str
    target_name: str
    model: any
    scaler: any
    feature_columns: List[str]
    metrics: Dict
    config: Dict
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ItemTrainer:
    """
    Trains models for a single item.

    Handles:
    - Data loading from PostgreSQL
    - Feature computation
    - Target computation
    - Temporal train/val/test splits
    - Model training with calibration
    - Evaluation and validation
    """

    def __init__(self, item_id: int, item_name: str, config: Optional[TrainingConfig] = None):
        self.item_id = item_id
        self.item_name = item_name
        self.config = config or TrainingConfig()

        # Initialize engines
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self.target_engine = TargetEngine(
            granularity='5m',
            config=TargetConfig(
                offsets=self.config.target_offsets,
                windows_hours=self.config.target_windows_hours
            )
        )

        self.data = None
        self.models = {}
        self.metrics = {}

    def load_data(self) -> pd.DataFrame:
        """Load item data from database."""
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
            ORDER BY timestamp
        """

        try:
            df = pd.read_sql(query, conn, params=[self.item_id])
        finally:
            conn.close()

        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and targets."""
        # Compute features
        df = self.feature_engine.compute_features(df)

        # Compute targets
        df = self.target_engine.compute_targets(df)

        # Drop warmup rows (features need history)
        df = df.iloc[self.config.drop_first_n_rows:].reset_index(drop=True)

        # Drop rows where we can't compute targets (end of series)
        # The target engine looks forward, so last N rows have NaN targets
        max_window = max(self.config.target_windows_hours) * 12  # Convert to periods
        df = df.iloc[:-max_window].reset_index(drop=True)

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal train/val/test split."""
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from DataFrame."""
        feature_cols = self.feature_engine.get_feature_columns()

        # Filter to columns that exist and are numeric
        valid_cols = []
        for col in feature_cols:
            if col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    valid_cols.append(col)

        X = df[valid_cols].values

        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return X, valid_cols

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Tuple[any, any]:
        """Train and calibrate a single model."""
        # Fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train base model
        base_model = LogisticRegression(
            C=self.config.regularization_C,
            max_iter=self.config.max_iter,
            solver='lbfgs',
            class_weight='balanced'  # Handle imbalanced classes
        )
        base_model.fit(X_train_scaled, y_train)

        # Combine train and val for calibration (use cross-validation)
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.hstack([y_train, y_val])

        # Calibrate using cross-validation
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method=self.config.calibration_method,
            cv=3  # Use 3-fold CV for calibration
        )
        calibrated_model.fit(X_combined, y_combined)

        return calibrated_model, scaler

    def evaluate_model(self, model, scaler, X: np.ndarray, y: np.ndarray,
                       target_name: str) -> Dict:
        """Evaluate model on a dataset."""
        X_scaled = scaler.transform(X)

        # Get predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # Calculate metrics
        base_rate = y.mean()

        metrics = {
            'target': target_name,
            'n_samples': len(y),
            'base_rate': float(base_rate),
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
        }

        # ROC AUC (only if both classes present)
        if len(np.unique(y)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y, y_prob))
        else:
            metrics['roc_auc'] = 0.5

        # Calibration metrics
        metrics['brier_score'] = float(brier_score_loss(y, y_prob))
        try:
            metrics['log_loss'] = float(log_loss(y, y_prob))
        except ValueError:
            metrics['log_loss'] = None

        # Expected value calculation (for fill targets)
        if 'roundtrip' in target_name:
            # Extract offset from target name (e.g., "roundtrip_2pct_24h")
            parts = target_name.split('_')
            offset_str = parts[1]  # e.g., "2pct"
            offset = int(offset_str.replace('pct', '')) / 100  # Convert to decimal

            avg_prob = y_prob.mean()
            metrics['avg_predicted_prob'] = float(avg_prob)
            metrics['expected_value'] = float(compute_expected_value(avg_prob, offset))

        return metrics

    def compute_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                  n_bins: int = 10) -> Dict:
        """Compute calibration curve data."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_means = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(y_true[mask].mean())
                bin_counts.append(int(mask.sum()))

        return {
            'bin_centers': bin_centers,
            'actual_rates': bin_means,
            'sample_counts': bin_counts
        }

    def train_all_targets(self) -> Dict[str, ModelArtifact]:
        """Train models for all target variables."""
        # Load and prepare data
        print(f"  Loading data for {self.item_name}...")
        raw_df = self.load_data()

        if len(raw_df) < 5000:
            print(f"  Insufficient data: {len(raw_df)} rows")
            return {}

        print(f"  Preparing features and targets ({len(raw_df)} rows)...")
        df = self.prepare_data(raw_df)

        if len(df) < 3000:
            print(f"  Insufficient data after preparation: {len(df)} rows")
            return {}

        # Split data
        train_df, val_df, test_df = self.split_data(df)
        print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Get feature matrices
        X_train, feature_cols = self.get_feature_matrix(train_df)
        X_val, _ = self.get_feature_matrix(val_df)
        X_test, _ = self.get_feature_matrix(test_df)

        # Train models for each target
        artifacts = {}
        primary_targets = self.target_engine.get_primary_targets()

        for target_name in primary_targets:
            if target_name not in train_df.columns:
                continue

            y_train = train_df[target_name].values.astype(int)
            y_val = val_df[target_name].values.astype(int)
            y_test = test_df[target_name].values.astype(int)

            # Skip if too imbalanced (need at least 1% positive class)
            if y_train.mean() < 0.01 or y_train.mean() > 0.99:
                print(f"    Skipping {target_name}: extreme imbalance ({y_train.mean():.1%})")
                continue

            # Train model
            try:
                model, scaler = self.train_model(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"    Error training {target_name}: {e}")
                continue

            # Evaluate
            train_metrics = self.evaluate_model(model, scaler, X_train, y_train, target_name)
            val_metrics = self.evaluate_model(model, scaler, X_val, y_val, target_name)
            test_metrics = self.evaluate_model(model, scaler, X_test, y_test, target_name)

            # Calibration curve on test set
            X_test_scaled = scaler.transform(X_test)
            y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
            calibration = self.compute_calibration_curve(y_test, y_prob_test)

            # Validate model meets minimum thresholds
            is_valid = (
                test_metrics['accuracy'] >= self.config.min_accuracy and
                test_metrics['roc_auc'] >= self.config.min_roc_auc and
                test_metrics['brier_score'] <= self.config.max_brier_score
            )

            artifact = ModelArtifact(
                item_id=self.item_id,
                item_name=self.item_name,
                target_name=target_name,
                model=model,
                scaler=scaler,
                feature_columns=feature_cols,
                metrics={
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics,
                    'calibration': calibration,
                    'is_valid': is_valid
                },
                config={
                    'train_size': len(train_df),
                    'val_size': len(val_df),
                    'test_size': len(test_df),
                    'training_config': {
                        'regularization_C': self.config.regularization_C,
                        'calibration_method': self.config.calibration_method
                    }
                }
            )

            artifacts[target_name] = artifact

            # Log result
            status = "VALID" if is_valid else "INVALID"
            print(f"    {target_name}: acc={test_metrics['accuracy']:.3f}, "
                  f"auc={test_metrics['roc_auc']:.3f}, brier={test_metrics['brier_score']:.3f} [{status}]")

        return artifacts


class DiscreteHourItemTrainer:
    """
    Trains discrete hour models for a single item.

    Unlike ItemTrainer which trains cumulative window models (e.g., "fills within 24h"),
    this trains discrete hour models (e.g., "fills in hour 3, not before").
    """

    def __init__(self, item_id: int, item_name: str,
                 config: Optional[DiscreteHourTrainingConfig] = None):
        self.item_id = item_id
        self.item_name = item_name
        self.config = config or DiscreteHourTrainingConfig()

        # Initialize engines
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
        self.target_engine = DiscreteHourTargetEngine(
            granularity='5m',
            config=DiscreteHourConfig(
                offsets=self.config.target_offsets,
                discrete_hours=self.config.discrete_hours
            )
        )

        self.data = None
        self.models = {}

    def load_data(self) -> pd.DataFrame:
        """Load item data from database with date range filter."""
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
              AND timestamp >= %s
              AND timestamp <= %s
            ORDER BY timestamp
        """

        try:
            df = pd.read_sql(query, conn, params=[
                self.item_id,
                self.config.start_date,
                self.config.end_date
            ])
        finally:
            conn.close()

        return df

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and discrete hour targets."""
        # Compute features
        df = self.feature_engine.compute_features(df)

        # Compute discrete hour targets
        df = self.target_engine.compute_targets(df)

        # Drop warmup rows (features need history)
        df = df.iloc[self.config.drop_first_n_rows:].reset_index(drop=True)

        # Drop rows where we can't compute targets (need 24h forward look)
        max_periods = max(self.config.discrete_hours) * 12  # 12 periods per hour
        df = df.iloc[:-max_periods].reset_index(drop=True)

        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal train/val/test split."""
        n = len(df)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract feature matrix from DataFrame."""
        feature_cols = self.feature_engine.get_feature_columns()

        valid_cols = []
        for col in feature_cols:
            if col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    valid_cols.append(col)

        X = df[valid_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        return X, valid_cols

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray) -> Tuple[any, any]:
        """Train and calibrate a single model."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        base_model = LogisticRegression(
            C=self.config.regularization_C,
            max_iter=self.config.max_iter,
            solver='lbfgs',
            class_weight='balanced'
        )
        base_model.fit(X_train_scaled, y_train)

        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.hstack([y_train, y_val])

        calibrated_model = CalibratedClassifierCV(
            base_model,
            method=self.config.calibration_method,
            cv=3
        )
        calibrated_model.fit(X_combined, y_combined)

        return calibrated_model, scaler

    def evaluate_model(self, model, scaler, X: np.ndarray, y: np.ndarray,
                       target_name: str, offset: float) -> Dict:
        """Evaluate model on a dataset."""
        X_scaled = scaler.transform(X)

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        base_rate = y.mean()

        metrics = {
            'target': target_name,
            'n_samples': len(y),
            'base_rate': float(base_rate),
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
        }

        if len(np.unique(y)) > 1:
            metrics['roc_auc'] = float(roc_auc_score(y, y_prob))
        else:
            metrics['roc_auc'] = 0.5

        metrics['brier_score'] = float(brier_score_loss(y, y_prob))
        metrics['avg_predicted_prob'] = float(y_prob.mean())
        metrics['expected_value'] = float(compute_expected_value(y_prob.mean(), offset))

        return metrics

    def train_single_target(self, target_name: str, offset: float, hour: int,
                            train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                            X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            feature_cols: List[str]) -> Optional[ModelArtifact]:
        """Train a single discrete hour target model."""
        if target_name not in train_df.columns:
            return None

        y_train = train_df[target_name].values.astype(int)
        y_val = val_df[target_name].values.astype(int)
        y_test = test_df[target_name].values.astype(int)

        # Skip if too imbalanced or too few positive samples
        # Need at least 20 positive samples in training set for reliable learning
        positive_samples = y_train.sum()
        if positive_samples < 20 or y_train.mean() > 0.999:
            return None

        try:
            model, scaler = self.train_model(X_train, y_train, X_val, y_val)
        except Exception:
            return None

        test_metrics = self.evaluate_model(model, scaler, X_test, y_test, target_name, offset)

        is_valid = (
            test_metrics['accuracy'] >= self.config.min_accuracy and
            test_metrics['roc_auc'] >= self.config.min_roc_auc and
            test_metrics['brier_score'] <= self.config.max_brier_score
        )

        artifact = ModelArtifact(
            item_id=self.item_id,
            item_name=self.item_name,
            target_name=target_name,
            model=model,
            scaler=scaler,
            feature_columns=feature_cols,
            metrics={
                'test': test_metrics,
                'is_valid': is_valid,
                'offset': offset,
                'hour': hour
            },
            config={
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'date_range': {
                    'start': self.config.start_date,
                    'end': self.config.end_date
                }
            }
        )

        return artifact

    def train_all_targets(self, verbose: bool = True) -> Dict[str, ModelArtifact]:
        """Train models for all discrete hour targets."""
        if verbose:
            print(f"  Loading data for {self.item_name} ({self.config.start_date} to {self.config.end_date})...")

        raw_df = self.load_data()

        if len(raw_df) < 5000:
            if verbose:
                print(f"  Insufficient data: {len(raw_df)} rows")
            return {}

        if verbose:
            print(f"  Preparing features and targets ({len(raw_df)} rows)...")

        df = self.prepare_data(raw_df)

        if len(df) < 3000:
            if verbose:
                print(f"  Insufficient data after preparation: {len(df)} rows")
            return {}

        train_df, val_df, test_df = self.split_data(df)

        if verbose:
            print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        X_train, feature_cols = self.get_feature_matrix(train_df)
        X_val, _ = self.get_feature_matrix(val_df)
        X_test, _ = self.get_feature_matrix(test_df)

        artifacts = {}
        total_targets = len(self.config.target_offsets) * len(self.config.discrete_hours)

        if verbose:
            print(f"  Training {total_targets} models...")

        trained = 0
        valid = 0

        for offset in self.config.target_offsets:
            for hour in self.config.discrete_hours:
                target_name = self.target_engine.get_target_for_offset_hour(offset, hour)

                artifact = self.train_single_target(
                    target_name, offset, hour,
                    train_df, val_df, test_df,
                    X_train, X_val, X_test,
                    feature_cols
                )

                if artifact:
                    artifacts[target_name] = artifact
                    trained += 1
                    if artifact.metrics.get('is_valid', False):
                        valid += 1

        if verbose:
            print(f"  Trained {trained} models, {valid} valid")

        return artifacts


class TrainingPipeline:
    """
    Main training pipeline.

    Trains models for all specified items and saves to registry.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.registry = {}

    def load_tier_items(self, tiers: List[int] = [1, 2]) -> List[Dict]:
        """Load items from tier files."""
        items = []
        for tier in tiers:
            filepath = f'data/tier_{tier}_items.json'
            if os.path.exists(filepath):
                with open(filepath) as f:
                    tier_items = json.load(f)
                    for item in tier_items:
                        item['tier'] = tier
                    items.extend(tier_items)
        return items

    def train_item(self, item: Dict) -> Optional[Dict[str, ModelArtifact]]:
        """Train all models for a single item."""
        item_id = item['item_id']
        item_name = item.get('name', f'Item-{item_id}')

        trainer = ItemTrainer(item_id, item_name, self.config)
        return trainer.train_all_targets()

    def save_model_registry(self, output_dir: str = 'models'):
        """Save all trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)

        registry_info = {
            'created_at': datetime.now().isoformat(),
            'items': {}
        }

        for item_id, artifacts in self.registry.items():
            item_dir = f'{output_dir}/{item_id}'
            os.makedirs(item_dir, exist_ok=True)

            item_info = {
                'item_id': item_id,
                'models': {}
            }

            for target_name, artifact in artifacts.items():
                # Save model and scaler
                model_path = f'{item_dir}/{target_name}_model.pkl'
                scaler_path = f'{item_dir}/{target_name}_scaler.pkl'
                meta_path = f'{item_dir}/{target_name}_meta.json'

                joblib.dump(artifact.model, model_path)
                joblib.dump(artifact.scaler, scaler_path)

                meta = {
                    'item_id': artifact.item_id,
                    'item_name': artifact.item_name,
                    'target_name': artifact.target_name,
                    'feature_columns': artifact.feature_columns,
                    'metrics': artifact.metrics,
                    'config': artifact.config,
                    'trained_at': artifact.trained_at
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)

                item_info['models'][target_name] = {
                    'model_path': model_path,
                    'scaler_path': scaler_path,
                    'meta_path': meta_path,
                    'is_valid': artifact.metrics.get('is_valid', False),
                    'test_roc_auc': artifact.metrics['test']['roc_auc'],
                    'test_brier': artifact.metrics['test']['brier_score']
                }

            if item_info['models']:
                registry_info['items'][str(item_id)] = item_info

        # Save registry index
        with open(f'{output_dir}/registry.json', 'w') as f:
            json.dump(registry_info, f, indent=2)

        print(f"\nModel registry saved to {output_dir}/")
        print(f"  Total items: {len(registry_info['items'])}")
        total_models = sum(len(info['models']) for info in registry_info['items'].values())
        print(f"  Total models: {total_models}")

    def train_all(self, items: Optional[List[Dict]] = None,
                  limit: Optional[int] = None) -> Dict:
        """Train models for all specified items."""
        if items is None:
            items = self.load_tier_items(tiers=[1, 2])

        if limit:
            items = items[:limit]

        print(f"\n{'='*70}")
        print("TRAINING PIPELINE")
        print(f"{'='*70}")
        print(f"Items to train: {len(items)}")

        successful = 0
        failed = 0

        for i, item in enumerate(items):
            item_id = item['item_id']
            item_name = item.get('name', f'Item-{item_id}')
            tier = item.get('tier', '?')

            print(f"\n[{i+1}/{len(items)}] Training {item_name} (ID: {item_id}, Tier: {tier})")

            try:
                artifacts = self.train_item(item)
                if artifacts:
                    self.registry[item_id] = artifacts
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                failed += 1

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        # Save registry
        self.save_model_registry()

        return {
            'successful': successful,
            'failed': failed,
            'registry': self.registry
        }


def main():
    """Main entry point for training."""
    print("Starting model training pipeline...")

    # Check if tier files exist
    if not os.path.exists('data/tier_1_items.json'):
        print("Error: Run item_analyzer.py first to generate tier files")
        return

    # Load items
    with open('data/tier_1_items.json') as f:
        tier1_items = json.load(f)
    with open('data/tier_2_items.json') as f:
        tier2_items = json.load(f)

    print(f"Found {len(tier1_items)} Tier 1 items and {len(tier2_items)} Tier 2 items")

    # Train all items
    pipeline = TrainingPipeline()

    # Combine tier 1 and tier 2 items
    all_items = tier1_items + tier2_items

    # Train (can limit for testing)
    results = pipeline.train_all(items=all_items, limit=None)

    return results


if __name__ == "__main__":
    main()
