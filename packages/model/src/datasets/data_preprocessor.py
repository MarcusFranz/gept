"""
Data Preprocessing and Quality Filters

Additional processing beyond basic feature computation:
- Sparse data detection and filtering
- Outlier handling
- Feature validation
- Data quality reports
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""

    # Sparsity filters
    max_missing_pct: float = 0.05  # Drop samples with >5% missing features
    max_feature_missing_pct: float = 0.10  # Drop features with >10% missing values

    # Outlier handling
    clip_outliers: bool = True
    outlier_std_threshold: float = 5.0  # Clip values beyond ±5 std

    # Feature validation
    drop_constant_features: bool = True  # Drop features with no variance
    min_feature_variance: float = 1e-8

    # Target validation
    min_positive_samples: int = 20  # Minimum positive samples per target
    drop_imbalanced_targets: bool = True
    max_imbalance_ratio: float = 0.001  # Drop if <0.1% positive

    # Duplicate detection
    drop_duplicates: bool = True

    # Reporting
    verbose: bool = True


class DataPreprocessor:
    """Preprocess datasets for better training quality"""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.preprocessing_stats = {}

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        item_metadata: Optional[List[Dict]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Apply all preprocessing steps

        Args:
            X: Feature DataFrame
            y: Target DataFrame
            item_metadata: Optional item metadata

        Returns:
            (X_clean, y_clean, stats_dict)
        """
        logger.info("Starting data preprocessing...")
        initial_rows = len(X)

        stats = {
            'initial_rows': initial_rows,
            'initial_features': len(X.columns),
            'initial_targets': len(y.columns),
            'steps': []
        }

        # Step 1: Check for misalignment
        assert len(X) == len(y), "X and y must have same number of rows"

        # Step 2: Remove duplicates
        if self.config.drop_duplicates:
            X, y, dup_stats = self._remove_duplicates(X, y)
            stats['steps'].append(dup_stats)

        # Step 3: Handle missing values in features
        X, missing_stats = self._handle_missing_features(X)
        stats['steps'].append(missing_stats)

        # Step 4: Clip outliers
        if self.config.clip_outliers:
            X, outlier_stats = self._clip_outliers(X)
            stats['steps'].append(outlier_stats)

        # Step 5: Remove low-variance features
        if self.config.drop_constant_features:
            X, variance_stats = self._remove_low_variance_features(X)
            stats['steps'].append(variance_stats)

        # Step 6: Validate targets
        if self.config.drop_imbalanced_targets:
            y, target_stats = self._validate_targets(y)
            stats['steps'].append(target_stats)

        # Step 7: Remove rows with sparse features
        X, y, sparsity_stats = self._remove_sparse_rows(X, y)
        stats['steps'].append(sparsity_stats)

        # Final stats
        stats['final_rows'] = len(X)
        stats['final_features'] = len(X.columns)
        stats['final_targets'] = len(y.columns)
        stats['rows_removed'] = initial_rows - len(X)
        stats['rows_removed_pct'] = (initial_rows - len(X)) / initial_rows * 100

        if self.config.verbose:
            self._print_summary(stats)

        self.preprocessing_stats = stats
        # Store fitted parameters for transform()
        self._fitted_feature_cols = list(X.columns)
        self._fitted_target_cols = list(y.columns)
        return X, y, stats

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transform val/test data using stats from fit_transform()

        This ensures val/test have same columns as train and uses train statistics.

        Args:
            X: Feature DataFrame
            y: Target DataFrame

        Returns:
            (X_transformed, y_transformed)
        """
        if not hasattr(self, '_fitted_feature_cols'):
            raise RuntimeError("Must call fit_transform() before transform()")

        # Align columns with training data
        # Add missing columns (fill with 0)
        for col in self._fitted_feature_cols:
            if col not in X.columns:
                X[col] = 0

        for col in self._fitted_target_cols:
            if col not in y.columns:
                y[col] = 0

        # Drop extra columns
        X = X[self._fitted_feature_cols]
        y = y[self._fitted_target_cols]

        # Only apply transformations that don't depend on statistics
        # (duplicates, missing values with forward fill)
        if self.config.drop_duplicates:
            duplicated = X.duplicated()
            X = X[~duplicated].copy()
            y = y[~duplicated].copy()

        # Fill missing with forward fill (don't use training stats)
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)

        logger.info(f"Transformed validation/test data: {len(X)} rows, {len(X.columns)} features, {len(y.columns)} targets")
        return X, y

    def _remove_duplicates(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Remove duplicate rows"""
        initial_rows = len(X)

        # Find duplicates based on features only
        duplicated = X.duplicated()
        n_duplicates = duplicated.sum()

        if n_duplicates > 0:
            X = X[~duplicated].reset_index(drop=True)
            y = y[~duplicated].reset_index(drop=True)
            logger.info(f"Removed {n_duplicates} duplicate rows")

        return X, y, {
            'step': 'remove_duplicates',
            'rows_removed': n_duplicates,
            'duplicate_pct': n_duplicates / initial_rows * 100
        }

    def _handle_missing_features(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Handle missing values in features"""
        initial_features = len(X.columns)

        # Compute missing percentage per feature
        missing_pct = X.isnull().sum() / len(X)

        # Drop features with too many missing values
        features_to_drop = missing_pct[missing_pct > self.config.max_feature_missing_pct].index.tolist()

        if features_to_drop:
            if 'item_id' in features_to_drop:
                features_to_drop.remove('item_id')  # Never drop item_id

            X = X.drop(columns=features_to_drop)
            logger.info(f"Dropped {len(features_to_drop)} features with >{self.config.max_feature_missing_pct:.1%} missing")

        # Fill remaining missing values with 0 (already done in builder, but safe)
        X = X.fillna(0)

        return X, {
            'step': 'handle_missing_features',
            'features_dropped': len(features_to_drop),
            'dropped_features': features_to_drop
        }

    def _clip_outliers(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Clip extreme outliers"""
        clipped_features = []

        for col in X.columns:
            if col == 'item_id':
                continue

            # Compute mean and std (robust to outliers)
            mean = X[col].mean()
            std = X[col].std()

            if std < 1e-8:  # Constant column
                continue

            # Clip to ±N standard deviations
            lower_bound = mean - self.config.outlier_std_threshold * std
            upper_bound = mean + self.config.outlier_std_threshold * std

            n_clipped = ((X[col] < lower_bound) | (X[col] > upper_bound)).sum()

            if n_clipped > 0:
                X[col] = X[col].clip(lower_bound, upper_bound)
                clipped_features.append((col, n_clipped))

        total_clipped = sum(n for _, n in clipped_features)

        if total_clipped > 0:
            logger.info(f"Clipped {total_clipped} outlier values across {len(clipped_features)} features")

        return X, {
            'step': 'clip_outliers',
            'features_clipped': len(clipped_features),
            'total_values_clipped': total_clipped
        }

    def _remove_low_variance_features(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Remove features with near-zero variance"""
        initial_features = len(X.columns)

        # Compute variance per feature
        variances = X.var()

        # Find low-variance features
        low_var_features = variances[variances < self.config.min_feature_variance].index.tolist()

        if 'item_id' in low_var_features:
            low_var_features.remove('item_id')  # Keep item_id even if constant

        if low_var_features:
            X = X.drop(columns=low_var_features)
            logger.info(f"Dropped {len(low_var_features)} features with variance < {self.config.min_feature_variance}")

        return X, {
            'step': 'remove_low_variance_features',
            'features_dropped': len(low_var_features),
            'dropped_features': low_var_features
        }

    def _validate_targets(
        self,
        y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Validate and filter targets"""
        initial_targets = len(y.columns)

        targets_to_drop = []

        for col in y.columns:
            # Count positive samples
            n_positive = y[col].sum()
            n_total = len(y)
            positive_rate = n_positive / n_total

            # Drop if insufficient positive samples
            if n_positive < self.config.min_positive_samples:
                targets_to_drop.append((col, f"only {n_positive} positive samples"))
                continue

            # Drop if too imbalanced
            if positive_rate < self.config.max_imbalance_ratio:
                targets_to_drop.append((col, f"only {positive_rate:.4%} positive"))
                continue

        if targets_to_drop:
            cols_to_drop = [col for col, reason in targets_to_drop]
            y = y.drop(columns=cols_to_drop)
            logger.warning(f"Dropped {len(targets_to_drop)} imbalanced targets")
            for col, reason in targets_to_drop[:5]:  # Show first 5
                logger.debug(f"  - {col}: {reason}")

        return y, {
            'step': 'validate_targets',
            'targets_dropped': len(targets_to_drop),
            'dropped_targets': [col for col, _ in targets_to_drop]
        }

    def _remove_sparse_rows(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Remove rows with too many missing features"""
        initial_rows = len(X)

        # Compute missing percentage per row (excluding item_id)
        X_features = X.drop(columns=['item_id']) if 'item_id' in X.columns else X
        missing_pct_per_row = X_features.isnull().sum(axis=1) / len(X_features.columns)

        # Find sparse rows
        sparse_mask = missing_pct_per_row > self.config.max_missing_pct
        n_sparse = sparse_mask.sum()

        if n_sparse > 0:
            X = X[~sparse_mask].reset_index(drop=True)
            y = y[~sparse_mask].reset_index(drop=True)
            logger.info(f"Removed {n_sparse} rows with >{self.config.max_missing_pct:.1%} missing features")

        return X, y, {
            'step': 'remove_sparse_rows',
            'rows_removed': n_sparse,
            'sparse_pct': n_sparse / initial_rows * 100
        }

    def _print_summary(self, stats: Dict):
        """Print preprocessing summary"""
        print("\n" + "="*70)
        print("DATA PREPROCESSING SUMMARY")
        print("="*70)
        print(f"\nInitial Dataset:")
        print(f"  Rows:     {stats['initial_rows']:,}")
        print(f"  Features: {stats['initial_features']}")
        print(f"  Targets:  {stats['initial_targets']}")

        print(f"\nPreprocessing Steps:")
        for step_stats in stats['steps']:
            step_name = step_stats['step'].replace('_', ' ').title()
            print(f"\n  {step_name}:")

            if 'rows_removed' in step_stats:
                print(f"    Rows removed: {step_stats['rows_removed']:,}")
            if 'features_dropped' in step_stats:
                print(f"    Features dropped: {step_stats['features_dropped']}")
                if step_stats['features_dropped'] > 0 and len(step_stats.get('dropped_features', [])) <= 10:
                    print(f"      {step_stats['dropped_features']}")
            if 'targets_dropped' in step_stats:
                print(f"    Targets dropped: {step_stats['targets_dropped']}")
            if 'total_values_clipped' in step_stats:
                print(f"    Values clipped: {step_stats['total_values_clipped']:,}")

        print(f"\nFinal Dataset:")
        print(f"  Rows:     {stats['final_rows']:,} ({stats['rows_removed_pct']:.1f}% removed)")
        print(f"  Features: {stats['final_features']}")
        print(f"  Targets:  {stats['final_targets']}")
        print("="*70 + "\n")

    def get_stats(self) -> Dict:
        """Get preprocessing statistics"""
        return self.preprocessing_stats


def generate_quality_report(
    X: pd.DataFrame,
    y: pd.DataFrame,
    output_path: Optional[str] = None
) -> Dict:
    """
    Generate comprehensive data quality report

    Args:
        X: Feature DataFrame
        y: Target DataFrame
        output_path: Optional path to save report

    Returns:
        Dictionary with quality metrics
    """
    report = {
        'feature_quality': {},
        'target_quality': {},
        'overall': {}
    }

    # Feature quality
    for col in X.columns:
        if col == 'item_id':
            continue

        report['feature_quality'][col] = {
            'mean': float(X[col].mean()),
            'std': float(X[col].std()),
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'missing_pct': float(X[col].isnull().sum() / len(X) * 100),
            'zeros_pct': float((X[col] == 0).sum() / len(X) * 100),
            'unique_values': int(X[col].nunique())
        }

    # Target quality
    for col in y.columns:
        n_positive = int(y[col].sum())
        n_total = len(y)

        report['target_quality'][col] = {
            'positive_samples': n_positive,
            'negative_samples': n_total - n_positive,
            'positive_rate': float(n_positive / n_total),
            'imbalance_ratio': float(min(n_positive, n_total - n_positive) / max(n_positive, n_total - n_positive))
        }

    # Overall stats
    report['overall'] = {
        'total_samples': len(X),
        'total_features': len(X.columns) - (1 if 'item_id' in X.columns else 0),
        'total_targets': len(y.columns),
        'total_missing_values': int(X.isnull().sum().sum()),
        'missing_pct': float(X.isnull().sum().sum() / (len(X) * len(X.columns)) * 100)
    }

    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Quality report saved to: {output_path}")

    return report


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, "/Users/marcusfranz/Documents/gept/packages/model/src")

    from datasets.dataset_loader import DatasetLoader

    # Load dataset
    loader = DatasetLoader()
    X_df, y_df = loader.load_wide_format("baseline_1.0", split="train")

    print(f"Loaded: X={X_df.shape}, y={y_df.shape}")

    # Preprocess
    config = PreprocessingConfig(
        max_missing_pct=0.05,
        clip_outliers=True,
        drop_imbalanced_targets=True,
        verbose=True
    )

    preprocessor = DataPreprocessor(config)
    X_clean, y_clean, stats = preprocessor.fit_transform(X_df, y_df)

    print(f"Cleaned: X={X_clean.shape}, y={y_clean.shape}")

    # Generate quality report
    report = generate_quality_report(X_clean, y_clean, "quality_report.json")
    print(f"\nQuality report saved!")
