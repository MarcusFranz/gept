#!/usr/bin/env python3
"""Train CatBoost ranking model for recommendations.

This script:
1. Loads the prepared feature dataset
2. Splits into train/validation sets
3. Trains a CatBoost ranking model
4. Evaluates and saves the model
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from catboost import CatBoost, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')


# Feature columns for the ranker (excluding identifiers and target)
FEATURE_COLUMNS = [
    # Item properties
    'buy_limit', 'is_members', 'highalch',
    'log_buy_limit',

    # Market context
    'avg_price', 'spread_pct', 'price_trend_1h', 'price_trend_24h',
    'volatility_24h', 'volume_24h',
    'log_volume_24h', 'log_avg_price',

    # Time features
    'hour_of_day', 'day_of_week', 'is_weekend',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',

    # Trade features
    'trade_quantity', 'trade_margin_pct', 'trade_capital',
    'log_trade_capital',
]


def load_dataset(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature and pairs datasets."""
    features = pd.read_parquet(os.path.join(data_dir, 'trade_features.parquet'))
    pairs = pd.read_parquet(os.path.join(data_dir, 'trade_pairs.parquet'))
    return features, pairs


def prepare_ranking_data(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """Prepare data for CatBoost ranking.

    For ranking, we need:
    - X: Feature matrix
    - y: Relevance labels (profit, or rank by profit)
    - group_id: Query groups (for ranking, each trade is its own group for pointwise)

    For pairwise ranking, we'll use profit as the relevance score.
    """
    # Use available feature columns
    available_features = [c for c in FEATURE_COLUMNS if c in features_df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = features_df[available_features].values

    # Use profit as relevance (higher = better)
    # Normalize to [0, 1] range for ranking
    profit = features_df['profit'].values
    y = (profit - profit.min()) / (profit.max() - profit.min() + 1e-8)

    # For pointwise ranking, each sample is its own group
    # But we can also use time-based groups for listwise ranking
    # Let's use day as group (trades on same day compete)
    features_df['date'] = pd.to_datetime(features_df['first_buy_time']).dt.date
    groups = features_df.groupby('date').ngroup().values

    return X, y, groups, available_features


def train_ranking_model(X_train, y_train, groups_train, X_val, y_val, groups_val):
    """Train CatBoost ranking model."""
    # Create Pool objects for CatBoost
    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=groups_train,
    )

    val_pool = Pool(
        data=X_val,
        label=y_val,
        group_id=groups_val,
    )

    # Model parameters for ranking
    params = {
        'loss_function': 'YetiRank',  # Pairwise ranking loss
        'custom_metric': ['NDCG', 'PFound'],
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': 50,
        'early_stopping_rounds': 50,
        'use_best_model': True,
    }

    print("Training CatBoost ranking model...")
    model = CatBoost(params)
    model.fit(train_pool, eval_set=val_pool)

    return model


def evaluate_model(model, X, y, groups, features_df, name=""):
    """Evaluate model performance."""
    predictions = model.predict(X)

    # Add predictions to dataframe for analysis
    results_df = features_df.copy()
    results_df['prediction'] = predictions

    # Calculate NDCG per group
    ndcg_scores = []
    for group_id in np.unique(groups):
        mask = groups == group_id
        if mask.sum() < 2:
            continue
        true_relevance = y[mask].reshape(1, -1)
        pred_relevance = predictions[mask].reshape(1, -1)
        try:
            score = ndcg_score(true_relevance, pred_relevance)
            ndcg_scores.append(score)
        except:
            pass

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0

    # Calculate ranking accuracy: does top-1 predicted beat random?
    # For each group, check if highest predicted is above median profit
    top1_above_median = []
    for group_id in np.unique(groups):
        mask = groups == group_id
        if mask.sum() < 2:
            continue
        group_y = y[mask]
        group_pred = predictions[mask]
        top1_idx = np.argmax(group_pred)
        median_y = np.median(group_y)
        top1_above_median.append(1 if group_y[top1_idx] > median_y else 0)

    top1_accuracy = np.mean(top1_above_median) if top1_above_median else 0

    # Correlation between predicted rank and actual profit
    correlation = np.corrcoef(predictions, y)[0, 1]

    metrics = {
        'avg_ndcg': avg_ndcg,
        'top1_above_median': top1_accuracy,
        'rank_correlation': correlation,
        'num_groups': len(np.unique(groups)),
        'num_samples': len(y),
    }

    print(f"\n{name} Metrics:")
    print(f"  Avg NDCG: {avg_ndcg:.4f}")
    print(f"  Top-1 Above Median: {top1_accuracy:.4f}")
    print(f"  Rank Correlation: {correlation:.4f}")
    print(f"  Num Groups: {len(np.unique(groups))}")

    return metrics, results_df


def main():
    data_dir = os.path.expanduser('~/gept/data/ranker_training')
    output_dir = os.path.expanduser('~/gept/models/ranker')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading dataset...")
    features_df, pairs_df = load_dataset(data_dir)
    print(f"Loaded {len(features_df)} trades, {len(pairs_df)} pairs")

    # Prepare ranking data
    X, y, groups, feature_names = prepare_ranking_data(features_df)
    print(f"Feature matrix shape: {X.shape}")

    # Time-based split (last 20% for validation)
    features_df_sorted = features_df.sort_values('first_buy_time').reset_index(drop=True)
    split_idx = int(len(features_df_sorted) * 0.8)

    train_indices = features_df_sorted.index[:split_idx].tolist()
    val_indices = features_df_sorted.index[split_idx:].tolist()

    X_train, y_train, groups_train = X[train_indices], y[train_indices], groups[train_indices]
    X_val, y_val, groups_val = X[val_indices], y[val_indices], groups[val_indices]

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # Train model
    model = train_ranking_model(X_train, y_train, groups_train, X_val, y_val, groups_val)

    # Evaluate on both sets
    train_metrics, train_results = evaluate_model(
        model, X_train, y_train, groups_train,
        features_df_sorted.iloc[train_indices], "Train"
    )
    val_metrics, val_results = evaluate_model(
        model, X_val, y_val, groups_val,
        features_df_sorted.iloc[val_indices], "Validation"
    )

    # Feature importance (pass training pool for LossFunctionChange type)
    train_pool = Pool(data=X_train, label=y_train, group_id=groups_train)
    importance = model.get_feature_importance(train_pool)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))

    # Save model
    model_path = os.path.join(output_dir, 'ranker_v1.cbm')
    model.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    # Save metadata
    metadata = {
        'version': 'v1',
        'created_at': datetime.now().isoformat(),
        'feature_columns': feature_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'feature_importance': importance_df.to_dict('records'),
    }

    metadata_path = os.path.join(output_dir, 'ranker_v1_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_path}")

    # Save feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved feature importance to {importance_path}")

    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()
