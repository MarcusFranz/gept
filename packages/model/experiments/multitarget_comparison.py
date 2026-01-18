#!/usr/bin/env python3
"""
Multi-Target CatBoost Training Comparison Experiment

Compares two training approaches:
1. Per-Target (baseline): 18 separate CatBoostClassifier models per item
2. Multi-Target: Single CatBoostClassifier with MultiLogloss for all 18 targets

Issue: #24 - Research: Multi-target CatBoost vs 48 per-target models

Usage:
    # Phase 1: Quick validation (10 items)
    python experiments/multitarget_comparison.py --phase 1 --n-items 10 --gpu

    # Phase 2: Full comparison (100 items)
    python experiments/multitarget_comparison.py --phase 2 --n-items 100 --gpu

    # CPU mode (slower but works without NVIDIA GPU)
    python experiments/multitarget_comparison.py --phase 1 --n-items 10
"""

import os
import sys
import json
import time
import argparse
import gc
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil
import psycopg2

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("ERROR: CatBoost not installed. Run: pip install catboost")
    sys.exit(1)

from feature_engine import FeatureEngine, Granularity
from db_utils import CONN_PARAMS


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the multi-target comparison experiment."""
    # Item selection
    n_items: int = 10
    random_seed: int = 42
    min_rows: int = 40000  # Minimum rows required per item

    # Date range (matches production)
    start_date: str = '2025-06-15'
    end_date: str = '2026-01-06'

    # Target configuration (18 targets total)
    hours: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 12, 24])
    offsets: List[float] = field(default_factory=lambda: [0.015, 0.02, 0.025])

    # Data splits (temporal)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # CatBoost parameters (matching production baseline)
    iterations: int = 200
    depth: int = 6
    learning_rate: float = 0.1

    # Validation thresholds
    min_auc: float = 0.52
    min_positive_samples: int = 20
    drop_first_n_rows: int = 300  # Feature warmup

    # Device
    use_gpu: bool = False

    # Output
    output_dir: str = "experiments/results"

    @property
    def n_targets(self) -> int:
        return len(self.hours) * len(self.offsets)

    @property
    def target_names(self) -> List[str]:
        names = []
        for offset in self.offsets:
            for hour in self.hours:
                names.append(f'{hour}h_{offset*100:.1f}pct')
        return names


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class TargetMetrics:
    """Metrics for a single target."""
    name: str
    auc: float
    positive_rate: float
    n_samples: int
    above_threshold: bool


@dataclass
class ItemResult:
    """Results for training one item with one approach."""
    item_id: int
    item_name: str
    approach: str  # 'per_target' or 'multi_target'
    n_rows: int
    n_features: int

    # Per-target metrics
    target_metrics: List[TargetMetrics]

    # Aggregate metrics
    mean_auc: float
    targets_above_threshold: int
    targets_total: int

    # Resource metrics
    train_time_seconds: float
    peak_memory_mb: float

    # Status
    success: bool
    error: Optional[str] = None


@dataclass
class ComparisonReport:
    """Comparison report for the experiment."""
    experiment_id: str
    phase: int
    config: Dict
    timestamp: str
    items_processed: int

    # Summary statistics
    time_savings_pct: float
    avg_auc_difference: float
    per_target_mean_auc: float
    multi_target_mean_auc: float
    per_target_targets_above_52: int
    multi_target_targets_above_52: int
    per_target_total_time_sec: float
    multi_target_total_time_sec: float
    per_target_peak_memory_mb: float
    multi_target_peak_memory_mb: float

    # Per-item results
    per_item_results: List[Dict]

    # Acceptance criteria
    passes_time_criteria: bool  # >= 20% time savings
    passes_auc_criteria: bool   # <= 0.005 AUC drop
    passes_all_criteria: bool

    # Recommendation
    recommendation: str


# =============================================================================
# Resource Profiling
# =============================================================================

@contextmanager
def profile_resources():
    """Context manager for resource profiling."""
    process = psutil.Process()
    gc.collect()

    start_time = time.perf_counter()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    result = {'peak_memory_mb': start_memory}

    def update_peak():
        current = process.memory_info().rss / 1024 / 1024
        result['peak_memory_mb'] = max(result['peak_memory_mb'], current)

    try:
        yield result, update_peak
    finally:
        end_time = time.perf_counter()
        update_peak()
        result['elapsed_seconds'] = end_time - start_time


# =============================================================================
# Base Trainer
# =============================================================================

class BaseTrainer:
    """Base class with shared functionality for both approaches."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.feature_engine = FeatureEngine(granularity=Granularity.FIVE_MIN)

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

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """Compute features and return feature matrix."""
        df_features = self.feature_engine.compute_features(df)
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

    def compute_fill_target(self, df: pd.DataFrame, hour: int, offset: float) -> np.ndarray:
        """Compute fill probability target."""
        periods_per_hour = 12  # 5-min intervals
        lookforward = hour * periods_per_hour

        # Buy price is low * (1 - offset)
        buy_price = df['low'] * (1 - offset)

        # Check if future low goes below our buy price
        future_min_low = df['low'].shift(-lookforward).rolling(lookforward).min()

        # Fill happens if future low <= buy price
        fills = (future_min_low <= buy_price).astype(int)

        return fills.values

    def temporal_split(self, X: np.ndarray, n: int) -> Tuple[int, int]:
        """Get temporal split indices."""
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        return train_end, val_end


# =============================================================================
# Per-Target Trainer (Baseline)
# =============================================================================

class PerTargetTrainer(BaseTrainer):
    """Per-target baseline trainer - 18 separate models per item."""

    def create_model(self) -> CatBoostClassifier:
        """Create a single-target CatBoost model."""
        params = {
            'iterations': self.config.iterations,
            'depth': self.config.depth,
            'learning_rate': self.config.learning_rate,
            'loss_function': 'Logloss',
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 20,
        }

        if self.config.use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        else:
            params['task_type'] = 'CPU'
            params['thread_count'] = 1

        return CatBoostClassifier(**params)

    def train_item(self, item_id: int, item_name: str, df: pd.DataFrame) -> ItemResult:
        """Train all 18 models for one item."""
        with profile_resources() as (resources, update_peak):
            try:
                # Prepare features
                X, feature_cols, df_features = self.prepare_features(df)

                # Drop warmup rows
                X = X[self.config.drop_first_n_rows:]
                df_features = df_features.iloc[self.config.drop_first_n_rows:].reset_index(drop=True)

                # Get split indices
                n = len(X)
                train_end, val_end = self.temporal_split(X, n)

                # Max lookforward for target computation
                max_lookforward = max(self.config.hours) * 12

                X_train = X[:train_end]
                X_val = X[train_end:val_end]
                X_test = X[val_end:]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)

                target_metrics = []

                for offset in self.config.offsets:
                    for hour in self.config.hours:
                        update_peak()

                        target_name = f'{hour}h_{offset*100:.1f}pct'

                        # Compute target
                        y_full = self.compute_fill_target(df_features, hour, offset)
                        y_full = y_full[:-max_lookforward]

                        y_train = y_full[:train_end]
                        y_val = y_full[train_end:val_end]
                        y_test = y_full[val_end:len(y_full)]

                        # Adjust X_test to match y_test
                        X_test_adj = X_test_scaled[:len(y_test)]

                        positive_rate = float(y_train.mean())

                        # Skip if insufficient positive samples
                        if y_train.sum() < self.config.min_positive_samples:
                            target_metrics.append(TargetMetrics(
                                name=target_name,
                                auc=0.5,
                                positive_rate=positive_rate,
                                n_samples=len(y_test),
                                above_threshold=False
                            ))
                            continue

                        # Train model
                        model = self.create_model()
                        model.fit(
                            X_train_scaled[:len(y_train)], y_train,
                            eval_set=(X_val_scaled[:len(y_val)], y_val),
                            verbose=False
                        )

                        # Predict
                        y_prob = model.predict_proba(X_test_adj)[:, 1]

                        # Calculate AUC
                        try:
                            auc = roc_auc_score(y_test, y_prob)
                        except ValueError:
                            auc = 0.5

                        target_metrics.append(TargetMetrics(
                            name=target_name,
                            auc=float(auc),
                            positive_rate=positive_rate,
                            n_samples=len(y_test),
                            above_threshold=auc > self.config.min_auc
                        ))

                        # Clean up
                        del model
                        gc.collect()

                # Aggregate metrics
                valid_aucs = [tm.auc for tm in target_metrics if tm.auc > 0.5]
                mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
                above_threshold = sum(1 for tm in target_metrics if tm.above_threshold)

                return ItemResult(
                    item_id=item_id,
                    item_name=item_name,
                    approach='per_target',
                    n_rows=len(df),
                    n_features=len(feature_cols),
                    target_metrics=target_metrics,
                    mean_auc=mean_auc,
                    targets_above_threshold=above_threshold,
                    targets_total=len(target_metrics),
                    train_time_seconds=resources['elapsed_seconds'],
                    peak_memory_mb=resources['peak_memory_mb'],
                    success=True
                )

            except Exception as e:
                return ItemResult(
                    item_id=item_id,
                    item_name=item_name,
                    approach='per_target',
                    n_rows=len(df),
                    n_features=0,
                    target_metrics=[],
                    mean_auc=0.0,
                    targets_above_threshold=0,
                    targets_total=0,
                    train_time_seconds=resources.get('elapsed_seconds', 0),
                    peak_memory_mb=resources.get('peak_memory_mb', 0),
                    success=False,
                    error=str(e)
                )


# =============================================================================
# Multi-Target Trainer
# =============================================================================

class MultiTargetTrainer(BaseTrainer):
    """Multi-target trainer - single model with MultiLogloss for all 18 targets."""

    def create_model(self) -> CatBoostClassifier:
        """Create a multi-target CatBoost model."""
        params = {
            'iterations': self.config.iterations,
            'depth': self.config.depth,
            'learning_rate': self.config.learning_rate,
            'loss_function': 'MultiLogloss',
            'random_seed': 42,
            'verbose': 0,
            # Note: early_stopping_rounds not supported with MultiLogloss
        }

        if self.config.use_gpu:
            params['task_type'] = 'GPU'
            params['devices'] = '0'
        else:
            params['task_type'] = 'CPU'
            params['thread_count'] = -1  # Use all cores for multi-target

        return CatBoostClassifier(**params)

    def prepare_target_matrix(self, df_features: pd.DataFrame, max_lookforward: int) -> Tuple[np.ndarray, List[str], List[float]]:
        """Create (N, K) target matrix for MultiLogloss."""
        targets = []
        target_names = []
        positive_rates = []

        for offset in self.config.offsets:
            for hour in self.config.hours:
                y = self.compute_fill_target(df_features, hour, offset)
                y = y[:-max_lookforward]
                targets.append(y)
                target_names.append(f'{hour}h_{offset*100:.1f}pct')
                positive_rates.append(float(np.mean(y)))

        # Stack: (N_samples, K_targets)
        return np.column_stack(targets), target_names, positive_rates

    def extract_per_target_aucs(self, model: CatBoostClassifier, X_test: np.ndarray,
                                 y_test_matrix: np.ndarray, target_names: List[str]) -> Dict[str, float]:
        """Extract per-target AUC from multi-target predictions."""
        # MultiLogloss predict_proba returns (N, 2*K) alternating [neg0, pos0, neg1, pos1, ...]
        proba = model.predict_proba(X_test)

        aucs = {}
        n_targets = len(target_names)

        # Handle both possible output formats
        if proba.shape[1] == 2 * n_targets:
            # Format: [neg0, pos0, neg1, pos1, ...]
            for i, name in enumerate(target_names):
                y_prob = proba[:, 2 * i + 1]  # Positive class probability
                y_true = y_test_matrix[:, i]

                try:
                    auc = roc_auc_score(y_true, y_prob)
                except ValueError:
                    auc = 0.5
                aucs[name] = float(auc)
        elif proba.shape[1] == n_targets:
            # Format: direct probabilities per target
            for i, name in enumerate(target_names):
                y_prob = proba[:, i]
                y_true = y_test_matrix[:, i]

                try:
                    auc = roc_auc_score(y_true, y_prob)
                except ValueError:
                    auc = 0.5
                aucs[name] = float(auc)
        else:
            # Unknown format - log warning and use raw values
            print(f"WARNING: Unexpected predict_proba shape: {proba.shape}, expected ({len(X_test)}, {2*n_targets}) or ({len(X_test)}, {n_targets})")
            for i, name in enumerate(target_names):
                aucs[name] = 0.5

        return aucs

    def train_item(self, item_id: int, item_name: str, df: pd.DataFrame) -> ItemResult:
        """Train single multi-target model for one item."""
        with profile_resources() as (resources, update_peak):
            try:
                # Prepare features
                X, feature_cols, df_features = self.prepare_features(df)

                # Drop warmup rows
                X = X[self.config.drop_first_n_rows:]
                df_features = df_features.iloc[self.config.drop_first_n_rows:].reset_index(drop=True)

                # Max lookforward for target computation
                max_lookforward = max(self.config.hours) * 12

                # Prepare target matrix
                y_matrix, target_names, positive_rates = self.prepare_target_matrix(df_features, max_lookforward)

                # Get split indices
                n = len(X)
                train_end, val_end = self.temporal_split(X, n)

                # Adjust for target matrix length
                X = X[:len(y_matrix)]

                X_train = X[:train_end]
                X_val = X[train_end:val_end]
                X_test = X[val_end:]

                y_train = y_matrix[:train_end]
                y_val = y_matrix[train_end:val_end]
                y_test = y_matrix[val_end:]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)

                update_peak()

                # Train multi-target model
                model = self.create_model()
                model.fit(
                    X_train_scaled, y_train,
                    eval_set=(X_val_scaled, y_val),
                    verbose=False
                )

                update_peak()

                # Extract per-target AUCs
                aucs = self.extract_per_target_aucs(model, X_test_scaled, y_test, target_names)

                # Build target metrics
                target_metrics = []
                for i, name in enumerate(target_names):
                    auc = aucs.get(name, 0.5)
                    target_metrics.append(TargetMetrics(
                        name=name,
                        auc=auc,
                        positive_rate=positive_rates[i],
                        n_samples=len(y_test),
                        above_threshold=auc > self.config.min_auc
                    ))

                # Aggregate metrics
                valid_aucs = [tm.auc for tm in target_metrics if tm.auc > 0.5]
                mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
                above_threshold = sum(1 for tm in target_metrics if tm.above_threshold)

                # Clean up
                del model
                gc.collect()

                return ItemResult(
                    item_id=item_id,
                    item_name=item_name,
                    approach='multi_target',
                    n_rows=len(df),
                    n_features=len(feature_cols),
                    target_metrics=target_metrics,
                    mean_auc=mean_auc,
                    targets_above_threshold=above_threshold,
                    targets_total=len(target_metrics),
                    train_time_seconds=resources['elapsed_seconds'],
                    peak_memory_mb=resources['peak_memory_mb'],
                    success=True
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return ItemResult(
                    item_id=item_id,
                    item_name=item_name,
                    approach='multi_target',
                    n_rows=len(df),
                    n_features=0,
                    target_metrics=[],
                    mean_auc=0.0,
                    targets_above_threshold=0,
                    targets_total=0,
                    train_time_seconds=resources.get('elapsed_seconds', 0),
                    peak_memory_mb=resources.get('peak_memory_mb', 0),
                    success=False,
                    error=str(e)
                )


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Orchestrates the full experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.per_target_trainer = PerTargetTrainer(config)
        self.multi_target_trainer = MultiTargetTrainer(config)

    def get_all_items(self) -> List[Dict]:
        """Get all items with sufficient data."""
        conn = psycopg2.connect(**CONN_PARAMS)

        query = """
            WITH item_stats AS (
                SELECT
                    p.item_id,
                    i.name as item_name,
                    COUNT(CASE WHEN avg_high_price IS NOT NULL
                               AND avg_low_price IS NOT NULL THEN 1 END) as valid_rows
                FROM price_data_5min p
                JOIN items i ON p.item_id = i.item_id
                WHERE p.timestamp >= %s AND p.timestamp <= %s
                GROUP BY p.item_id, i.name
                HAVING COUNT(CASE WHEN avg_high_price IS NOT NULL
                                  AND avg_low_price IS NOT NULL THEN 1 END) > %s
            )
            SELECT item_id, item_name, valid_rows
            FROM item_stats
            ORDER BY valid_rows DESC
        """

        df = pd.read_sql(query, conn, params=[
            self.config.start_date,
            self.config.end_date,
            self.config.min_rows
        ])
        conn.close()

        items = []
        for _, row in df.iterrows():
            items.append({
                'item_id': int(row['item_id']),
                'item_name': row['item_name'],
                'valid_rows': int(row['valid_rows'])
            })

        return items

    def select_items(self, n_items: int) -> List[Dict]:
        """Select random items with reproducible seed."""
        all_items = self.get_all_items()
        np.random.seed(self.config.random_seed)
        indices = np.random.choice(len(all_items), size=min(n_items, len(all_items)), replace=False)
        return [all_items[i] for i in sorted(indices)]

    def run_phase(self, phase: int) -> ComparisonReport:
        """Run experiment phase."""
        n_items = self.config.n_items
        items = self.select_items(n_items)

        print(f"\n{'='*70}")
        print(f"PHASE {phase}: Multi-Target vs Per-Target Comparison")
        print(f"{'='*70}")
        print(f"Items: {len(items)}")
        print(f"Targets per item: {self.config.n_targets}")
        print(f"GPU: {self.config.use_gpu}")
        print(f"{'='*70}\n")

        per_target_results: List[ItemResult] = []
        multi_target_results: List[ItemResult] = []

        for i, item in enumerate(items):
            item_id = item['item_id']
            item_name = item['item_name']

            print(f"[{i+1}/{len(items)}] Processing {item_name} (ID: {item_id})...")

            # Load data once
            df = self.per_target_trainer.load_item_data(item_id)

            if len(df) < 5000:
                print(f"  Skipping - insufficient data ({len(df)} rows)")
                continue

            # Run per-target baseline
            print(f"  Training per-target baseline (18 models)...")
            baseline_result = self.per_target_trainer.train_item(item_id, item_name, df)
            per_target_results.append(baseline_result)

            if baseline_result.success:
                print(f"    AUC: {baseline_result.mean_auc:.4f}, "
                      f"Time: {baseline_result.train_time_seconds:.1f}s, "
                      f"Memory: {baseline_result.peak_memory_mb:.0f}MB")
            else:
                print(f"    ERROR: {baseline_result.error}")

            # Run multi-target
            print(f"  Training multi-target (1 model)...")
            multitarget_result = self.multi_target_trainer.train_item(item_id, item_name, df)
            multi_target_results.append(multitarget_result)

            if multitarget_result.success:
                print(f"    AUC: {multitarget_result.mean_auc:.4f}, "
                      f"Time: {multitarget_result.train_time_seconds:.1f}s, "
                      f"Memory: {multitarget_result.peak_memory_mb:.0f}MB")
            else:
                print(f"    ERROR: {multitarget_result.error}")

            # Comparison
            if baseline_result.success and multitarget_result.success:
                auc_diff = multitarget_result.mean_auc - baseline_result.mean_auc
                time_savings = (baseline_result.train_time_seconds - multitarget_result.train_time_seconds) / baseline_result.train_time_seconds * 100
                print(f"  -> AUC diff: {auc_diff:+.4f}, Time savings: {time_savings:.1f}%")

            print()
            gc.collect()

        # Generate comparison report
        return self.generate_report(phase, per_target_results, multi_target_results)

    def generate_report(self, phase: int,
                        per_target_results: List[ItemResult],
                        multi_target_results: List[ItemResult]) -> ComparisonReport:
        """Generate comparison report."""
        # Filter successful results
        successful_pairs = [
            (pt, mt) for pt, mt in zip(per_target_results, multi_target_results)
            if pt.success and mt.success
        ]

        if not successful_pairs:
            return ComparisonReport(
                experiment_id=f"multitarget_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                phase=phase,
                config=asdict(self.config),
                timestamp=datetime.now().isoformat(),
                items_processed=0,
                time_savings_pct=0,
                avg_auc_difference=0,
                per_target_mean_auc=0,
                multi_target_mean_auc=0,
                per_target_targets_above_52=0,
                multi_target_targets_above_52=0,
                per_target_total_time_sec=0,
                multi_target_total_time_sec=0,
                per_target_peak_memory_mb=0,
                multi_target_peak_memory_mb=0,
                per_item_results=[],
                passes_time_criteria=False,
                passes_auc_criteria=False,
                passes_all_criteria=False,
                recommendation="Insufficient data - experiment failed"
            )

        # Compute aggregates
        pt_aucs = [pt.mean_auc for pt, _ in successful_pairs]
        mt_aucs = [mt.mean_auc for _, mt in successful_pairs]
        pt_times = [pt.train_time_seconds for pt, _ in successful_pairs]
        mt_times = [mt.train_time_seconds for _, mt in successful_pairs]
        pt_memory = [pt.peak_memory_mb for pt, _ in successful_pairs]
        mt_memory = [mt.peak_memory_mb for _, mt in successful_pairs]
        pt_above = sum(pt.targets_above_threshold for pt, _ in successful_pairs)
        mt_above = sum(mt.targets_above_threshold for _, mt in successful_pairs)

        per_target_mean_auc = float(np.mean(pt_aucs))
        multi_target_mean_auc = float(np.mean(mt_aucs))
        avg_auc_difference = multi_target_mean_auc - per_target_mean_auc

        per_target_total_time = sum(pt_times)
        multi_target_total_time = sum(mt_times)
        time_savings_pct = (per_target_total_time - multi_target_total_time) / per_target_total_time * 100

        # Per-item results
        per_item_results = []
        for pt, mt in successful_pairs:
            per_item_results.append({
                'item_id': pt.item_id,
                'item_name': pt.item_name,
                'per_target': {
                    'mean_auc': pt.mean_auc,
                    'targets_above_52': pt.targets_above_threshold,
                    'train_time_sec': pt.train_time_seconds,
                    'peak_memory_mb': pt.peak_memory_mb
                },
                'multi_target': {
                    'mean_auc': mt.mean_auc,
                    'targets_above_52': mt.targets_above_threshold,
                    'train_time_sec': mt.train_time_seconds,
                    'peak_memory_mb': mt.peak_memory_mb
                },
                'auc_difference': mt.mean_auc - pt.mean_auc,
                'time_savings_pct': (pt.train_time_seconds - mt.train_time_seconds) / pt.train_time_seconds * 100
            })

        # Acceptance criteria
        passes_time = time_savings_pct >= 20.0
        passes_auc = abs(avg_auc_difference) <= 0.005
        passes_all = passes_time and passes_auc

        # Generate recommendation
        if passes_all:
            recommendation = f"ADOPT: Multi-target training achieves {time_savings_pct:.1f}% time savings with only {abs(avg_auc_difference):.4f} AUC difference"
        elif passes_time and not passes_auc:
            recommendation = f"REJECT: Time savings of {time_savings_pct:.1f}% but AUC drop of {abs(avg_auc_difference):.4f} exceeds threshold"
        elif passes_auc and not passes_time:
            recommendation = f"INVESTIGATE: AUC within tolerance but only {time_savings_pct:.1f}% time savings (target: 20%)"
        else:
            recommendation = f"REJECT: Neither time ({time_savings_pct:.1f}%) nor AUC ({avg_auc_difference:.4f}) criteria met"

        return ComparisonReport(
            experiment_id=f"multitarget_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            phase=phase,
            config=asdict(self.config),
            timestamp=datetime.now().isoformat(),
            items_processed=len(successful_pairs),
            time_savings_pct=float(time_savings_pct),
            avg_auc_difference=float(avg_auc_difference),
            per_target_mean_auc=per_target_mean_auc,
            multi_target_mean_auc=multi_target_mean_auc,
            per_target_targets_above_52=pt_above,
            multi_target_targets_above_52=mt_above,
            per_target_total_time_sec=per_target_total_time,
            multi_target_total_time_sec=multi_target_total_time,
            per_target_peak_memory_mb=float(max(pt_memory)) if pt_memory else 0,
            multi_target_peak_memory_mb=float(max(mt_memory)) if mt_memory else 0,
            per_item_results=per_item_results,
            passes_time_criteria=passes_time,
            passes_auc_criteria=passes_auc,
            passes_all_criteria=passes_all,
            recommendation=recommendation
        )

    def save_report(self, report: ComparisonReport, phase: int):
        """Save report to JSON file."""
        phase_dir = os.path.join(self.config.output_dir, f"phase{phase}_{self.config.n_items}items")
        os.makedirs(phase_dir, exist_ok=True)

        report_path = os.path.join(phase_dir, "comparison_report.json")

        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)

        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

        print(f"\nReport saved to: {report_path}")
        return report_path


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Target CatBoost Training Comparison Experiment'
    )
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Experiment phase (1=10 items, 2=100 items)')
    parser.add_argument('--n-items', type=int, default=None,
                        help='Number of items to test (overrides phase default)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for item selection')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Output directory for results')
    args = parser.parse_args()

    # Determine n_items
    if args.n_items is not None:
        n_items = args.n_items
    else:
        n_items = 10 if args.phase == 1 else 100

    # Create config
    config = ExperimentConfig(
        n_items=n_items,
        random_seed=args.seed,
        use_gpu=args.gpu,
        output_dir=args.output_dir
    )

    # Check GPU availability
    if config.use_gpu:
        try:
            test_model = CatBoostClassifier(task_type='GPU', devices='0', verbose=False)
            del test_model
            print("GPU acceleration: ENABLED")
        except Exception as e:
            print(f"GPU not available ({e}), falling back to CPU")
            config.use_gpu = False

    # Run experiment
    runner = ExperimentRunner(config)
    report = runner.run_phase(args.phase)

    # Save report
    runner.save_report(report, args.phase)

    # Print summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Items processed: {report.items_processed}")
    print(f"\nTime Comparison:")
    print(f"  Per-target total: {report.per_target_total_time_sec:.1f}s")
    print(f"  Multi-target total: {report.multi_target_total_time_sec:.1f}s")
    print(f"  Time savings: {report.time_savings_pct:.1f}%")
    print(f"\nAUC Comparison:")
    print(f"  Per-target mean AUC: {report.per_target_mean_auc:.4f}")
    print(f"  Multi-target mean AUC: {report.multi_target_mean_auc:.4f}")
    print(f"  AUC difference: {report.avg_auc_difference:+.4f}")
    print(f"\nTargets above 0.52:")
    print(f"  Per-target: {report.per_target_targets_above_52}")
    print(f"  Multi-target: {report.multi_target_targets_above_52}")
    print(f"\nAcceptance Criteria:")
    print(f"  Time savings >= 20%: {'PASS' if report.passes_time_criteria else 'FAIL'}")
    print(f"  AUC drop <= 0.005: {'PASS' if report.passes_auc_criteria else 'FAIL'}")
    print(f"\n>>> RECOMMENDATION: {report.recommendation}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
