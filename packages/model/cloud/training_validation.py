"""
Training Data Validation - Pre-flight checks for training pipeline

Provides validation functions for:
- Config schema validation
- Feature column validation
- Sample count validation
- Data shape consistency checks

Usage:
    from training_validation import (
        validate_config,
        validate_item_config,
        validate_feature_columns,
        validate_training_data,
        ValidationError
    )

    # Validate config at startup
    config = load_config(...)
    validate_config(config)

    # Validate per-item data before training
    validate_training_data(df, feature_cols, min_samples=1500)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when training data validation fails."""
    pass


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate training run config schema.

    Args:
        config: Config dict loaded from config.json

    Raises:
        ValidationError: If config is invalid
    """
    # Check required top-level keys
    if 'items' not in config:
        raise ValidationError("Config missing required 'items' key")

    items = config['items']

    # Validate items is a non-empty list
    if not isinstance(items, list):
        raise ValidationError(f"Config 'items' must be a list, got {type(items).__name__}")

    if len(items) == 0:
        raise ValidationError("Config 'items' list is empty")

    # Validate each item has required fields
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValidationError(f"Item at index {i} must be a dict, got {type(item).__name__}")

        if 'item_id' not in item:
            raise ValidationError(f"Item at index {i} missing required 'item_id' key")

        item_id = item['item_id']
        if not isinstance(item_id, (int, float)):
            raise ValidationError(
                f"Item at index {i} has invalid item_id type: {type(item_id).__name__}"
            )

    logger.debug(f"Config validated: {len(items)} items")


def validate_item_config(item: Dict[str, Any], index: int = 0) -> int:
    """
    Validate a single item config entry.

    Args:
        item: Item config dict
        index: Index in items list (for error messages)

    Returns:
        item_id as int

    Raises:
        ValidationError: If item config is invalid
    """
    if not isinstance(item, dict):
        raise ValidationError(f"Item at index {index} must be a dict")

    if 'item_id' not in item:
        raise ValidationError(f"Item at index {index} missing 'item_id'")

    item_id = item['item_id']

    # Convert to int if float
    if isinstance(item_id, float):
        item_id = int(item_id)

    if not isinstance(item_id, int) or item_id <= 0:
        raise ValidationError(f"Item at index {index} has invalid item_id: {item_id}")

    return item_id


def validate_feature_columns(
    df: pd.DataFrame,
    feature_cols: List[str],
    item_id: Optional[int] = None
) -> None:
    """
    Validate feature columns exist and are non-empty.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        item_id: Item ID for error messages (optional)

    Raises:
        ValidationError: If feature columns are invalid
    """
    item_str = f" for item {item_id}" if item_id else ""

    # Check feature_cols is non-empty
    if not feature_cols:
        raise ValidationError(f"No feature columns found{item_str}")

    # Check all feature columns exist in dataframe
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        # Log warning but don't fail - will be handled by reindex with NaN
        logger.warning(
            f"Missing {len(missing)} feature columns{item_str}: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    # Check at least some numeric columns exist
    present_cols = [col for col in feature_cols if col in df.columns]
    if not present_cols:
        raise ValidationError(
            f"None of the {len(feature_cols)} feature columns exist in dataframe{item_str}"
        )


def validate_training_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_samples: int = 1500,
    item_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Comprehensive validation of training data before model training.

    Args:
        df: DataFrame with features and price data
        feature_cols: List of feature column names
        min_samples: Minimum required samples (default 1500)
        item_id: Item ID for error messages (optional)

    Returns:
        Dict with validation stats

    Raises:
        ValidationError: If data is invalid for training
    """
    item_str = f" for item {item_id}" if item_id else ""
    stats = {
        'n_rows': len(df),
        'n_feature_cols': len(feature_cols),
        'warnings': [],
    }

    # 1. Early sample count check
    if len(df) < min_samples:
        raise ValidationError(
            f"Insufficient samples{item_str}: {len(df)} < {min_samples} required"
        )
    stats['min_samples_ok'] = True

    # 2. Validate feature columns
    validate_feature_columns(df, feature_cols, item_id)

    # 3. Check required price columns for target computation
    required_price_cols = ['low', 'high']
    missing_price = [col for col in required_price_cols if col not in df.columns]
    if missing_price:
        raise ValidationError(
            f"Missing required price columns{item_str}: {missing_price}"
        )
    stats['price_cols_ok'] = True

    # 4. Check for all-NaN feature columns
    present_feature_cols = [col for col in feature_cols if col in df.columns]
    if present_feature_cols:
        feature_df = df[present_feature_cols]
        all_nan_cols = feature_df.columns[feature_df.isna().all()].tolist()
        if all_nan_cols:
            stats['warnings'].append(
                f"Found {len(all_nan_cols)} all-NaN columns{item_str}: "
                f"{all_nan_cols[:3]}{'...' if len(all_nan_cols) > 3 else ''}"
            )

        # Check percentage of NaN values
        nan_pct = feature_df.isna().sum().sum() / feature_df.size * 100
        stats['nan_percentage'] = round(nan_pct, 2)
        if nan_pct > 50:
            stats['warnings'].append(
                f"High NaN percentage{item_str}: {nan_pct:.1f}% of feature values"
            )

    # 5. Check for constant (zero-variance) columns
    if present_feature_cols:
        numeric_df = feature_df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            constant_cols = numeric_df.columns[numeric_df.std() == 0].tolist()
            if constant_cols:
                stats['warnings'].append(
                    f"Found {len(constant_cols)} constant columns{item_str}"
                )
            stats['n_numeric_features'] = len(numeric_df.columns) - len(constant_cols)
        else:
            stats['n_numeric_features'] = 0

    # 6. Check price columns for valid values
    for col in required_price_cols:
        if col in df.columns:
            price_series = df[col]
            invalid_count = ((price_series <= 0) | price_series.isna()).sum()
            if invalid_count > len(df) * 0.1:  # More than 10% invalid
                stats['warnings'].append(
                    f"High invalid price values in '{col}'{item_str}: {invalid_count}/{len(df)}"
                )

    # Log warnings
    for warning in stats['warnings']:
        logger.warning(warning)

    return stats


def validate_xy_shapes(
    X: np.ndarray,
    y: np.ndarray,
    item_id: Optional[int] = None
) -> None:
    """
    Validate X and y array shapes are compatible.

    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,) or (n_samples, n_targets)
        item_id: Item ID for error messages (optional)

    Raises:
        ValidationError: If shapes are incompatible
    """
    item_str = f" for item {item_id}" if item_id else ""

    if X.shape[0] != y.shape[0]:
        raise ValidationError(
            f"X/y shape mismatch{item_str}: X has {X.shape[0]} samples, "
            f"y has {y.shape[0]} samples"
        )

    if X.shape[0] == 0:
        raise ValidationError(f"Empty arrays{item_str}: X shape is {X.shape}")

    if len(X.shape) != 2:
        raise ValidationError(
            f"X must be 2D{item_str}, got shape {X.shape}"
        )

    if X.shape[1] == 0:
        raise ValidationError(f"No features{item_str}: X has 0 columns")


def validate_target_matrix(
    y: np.ndarray,
    target_names: List[str],
    min_positive_rate: float = 0.001,
    item_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate target matrix for multi-target training.

    Args:
        y: Target array (n_samples, n_targets)
        target_names: List of target names
        min_positive_rate: Minimum positive rate per target (default 0.1%)
        item_id: Item ID for error messages (optional)

    Returns:
        Dict with target statistics

    Raises:
        ValidationError: If targets are invalid
    """
    item_str = f" for item {item_id}" if item_id else ""

    if len(y.shape) != 2:
        raise ValidationError(f"Target array must be 2D{item_str}, got {y.shape}")

    if y.shape[1] != len(target_names):
        raise ValidationError(
            f"Target count mismatch{item_str}: array has {y.shape[1]} columns, "
            f"expected {len(target_names)} targets"
        )

    stats = {
        'n_samples': y.shape[0],
        'n_targets': y.shape[1],
        'valid_targets': 0,
        'low_positive_targets': [],
    }

    for i, name in enumerate(target_names):
        pos_rate = y[:, i].mean()
        if pos_rate >= min_positive_rate:
            stats['valid_targets'] += 1
        else:
            stats['low_positive_targets'].append({
                'name': name,
                'positive_rate': float(pos_rate),
            })

    if stats['valid_targets'] == 0:
        raise ValidationError(
            f"No valid targets{item_str}: all {len(target_names)} targets have "
            f"positive rate < {min_positive_rate*100:.2f}%"
        )

    return stats
