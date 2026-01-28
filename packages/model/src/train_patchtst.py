#!/usr/bin/env python3
"""
PatchTSMixer Training Script for GePT

Trains time-series forecasting models on OSRS price data using PatchTSMixer.
Designed for Docker containers with NVIDIA GPU support.

Based on: experiments/notebooks/patchtsmixer_experiment.ipynb
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
from loguru import logger

from datasets.dataset_loader import DatasetLoader


def create_sliding_windows(features, context_len, pred_len):
    """
    Create sliding windows for time series forecasting

    Args:
        features: numpy array of shape (timesteps, num_features)
        context_len: number of past timesteps to use as input
        pred_len: number of future timesteps to predict

    Returns:
        X: (samples, context_len, num_features)
        y: (samples, pred_len, 1) - predicting first feature only
    """
    from numpy.lib.stride_tricks import sliding_window_view

    total_len = context_len + pred_len

    # Create sliding windows
    windows = sliding_window_view(features, window_shape=total_len, axis=0)
    windows = windows.transpose(0, 2, 1)  # (samples, timesteps, features)

    # Split into X (context) and y (future)
    X = windows[:, :context_len, :]
    y = windows[:, context_len:, 0:1]  # Predict first feature (high price)

    return X, y


def normalize_data(X_train, y_train, X_test, y_test):
    """
    Normalize features and targets using local normalization

    Features: Global z-score normalization
    Targets: Local normalization (difference from last context value)

    Returns:
        X_train_norm, y_train_norm, X_test_norm, y_test_norm, normalization_stats
    """
    # Get last context value (anchor point for predictions)
    X_train_last = X_train[:, -1:, 0:1]
    X_test_last = X_test[:, -1:, 0:1]

    # Normalize X features (global z-score)
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True)
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # Normalize y as difference from last known value (local normalization)
    y_train_diff = y_train - X_train_last
    y_test_diff = y_test - X_test_last

    # Scale the differences
    y_diff_std = y_train_diff.std()
    y_train_norm = y_train_diff / y_diff_std
    y_test_norm = y_test_diff / y_diff_std

    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_diff_std': float(y_diff_std),
        'X_test_last': X_test_last
    }

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm, stats


def denormalize_predictions(pred_norm, last_values, y_diff_std):
    """Denormalize predictions back to original scale"""
    # Unscale, then add anchor
    pred_real = (pred_norm * y_diff_std) + last_values
    return pred_real


def prepare_dataset_for_timeseries(dataset_version, context_len, pred_len, feature_cols=None):
    """
    Load dataset and prepare for time series forecasting

    Args:
        dataset_version: Dataset to load (e.g., "baseline_1.0")
        context_len: Number of past timesteps
        pred_len: Number of future timesteps to predict
        feature_cols: List of feature names to use (defaults to ['high', 'spread', 'velocity', 'volatility'])

    Returns:
        X_train, y_train, X_test, y_test, stats
    """
    if feature_cols is None:
        feature_cols = ['high', 'spread', 'mid', 'spread_pct']

    loader = DatasetLoader()

    # Load as pandas for easier feature engineering
    X_df, y_df = loader.load_wide_format(dataset_version, split="train")

    logger.info(f"Loaded dataset: {dataset_version}")
    logger.info(f"Available features: {list(X_df.columns)[:20]}...")  # Show first 20

    # Compute additional features if needed
    if 'velocity' not in X_df.columns and 'high' in X_df.columns:
        X_df['velocity'] = X_df['high'].diff().fillna(0)

    if 'volatility' not in X_df.columns and 'high' in X_df.columns:
        X_df['volatility'] = X_df['high'].rolling(10).std().fillna(0)

    # Drop rows with NaN
    X_df = X_df.dropna()

    # Select feature columns
    missing_cols = [col for col in feature_cols if col not in X_df.columns]
    if missing_cols:
        logger.warning(f"Missing features: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in X_df.columns]

    logger.info(f"Using features: {feature_cols}")

    features = X_df[feature_cols].values

    # Create sliding windows
    X, y = create_sliding_windows(features, context_len, pred_len)

    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    # Normalize
    X_train_norm, y_train_norm, X_test_norm, y_test_norm, stats = normalize_data(
        X_train, y_train, X_test, y_test
    )

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"X shape: {X_train_norm.shape}, y shape: {y_train_norm.shape}")

    return X_train_norm, y_train_norm, X_test_norm, y_test_norm, stats


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(past_values=batch_x)
        predictions = outputs.prediction_outputs[:, :, 0:1]
        loss = nn.functional.mse_loss(predictions, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_on_test(model, loader, device):
    """Test model performance"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(past_values=batch_x)
            predictions = outputs.prediction_outputs[:, :, 0:1]
            loss = nn.functional.mse_loss(predictions, batch_y)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train PatchTSMixer for GePT")

    # Dataset args
    parser.add_argument("--dataset", type=str, default="baseline_1.0")
    parser.add_argument("--dataset-dir", type=str, default="/workspace/datasets")
    parser.add_argument("--features", nargs="+", default=["high", "spread", "velocity", "volatility"])

    # Time series args
    parser.add_argument("--context-len", type=int, default=90,
                       help="Number of past timesteps (lookback)")
    parser.add_argument("--pred-len", type=int, default=10,
                       help="Number of future timesteps to predict")

    # Model args
    parser.add_argument("--patch-len", type=int, default=10)
    parser.add_argument("--patch-stride", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--expansion-factor", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.2)

    # Training args
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Output args
    parser.add_argument("--output-dir", type=str, default="/workspace/outputs")
    parser.add_argument("--model-dir", type=str, default="/workspace/models")
    parser.add_argument("--experiment-name", type=str, default=None)

    # System args
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Setup experiment
    if args.experiment_name is None:
        args.experiment_name = f"patchtst_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir) / args.experiment_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / "training.log"
    logger.add(log_file)
    logger.info(f"Experiment: {args.experiment_name}")

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Prepare data
    logger.info("Preparing dataset for time series forecasting...")
    X_train, y_train, X_test, y_test, norm_stats = prepare_dataset_for_timeseries(
        args.dataset,
        args.context_len,
        args.pred_len,
        args.features
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    num_features = X_train.shape[2]
    logger.info(f"Creating PatchTSMixer: context={args.context_len}, pred={args.pred_len}, features={num_features}")

    config = PatchTSMixerConfig(
        context_length=args.context_len,
        prediction_length=args.pred_len,
        patch_length=args.patch_len,
        num_input_channels=num_features,
        num_targets=1,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_layers=args.num_layers,
        expansion_factor=args.expansion_factor,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        mode="common_channel",
    )

    model = PatchTSMixerForPrediction(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best_test_loss = float('inf')

    logger.info("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate_on_test(model, test_loader, device)

        logger.info(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_path = model_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.to_dict(),
                'norm_stats': norm_stats,
                'train_loss': train_loss,
                'test_loss': test_loss,
            }, best_model_path)
            logger.info(f"Saved best model (test_loss={test_loss:.4f})")

    # Save final model
    final_model_path = model_dir / "final_model.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'norm_stats': norm_stats,
    }, final_model_path)

    writer.close()
    logger.info(f"Training complete! Best test loss: {best_test_loss:.4f}")

    # Save summary
    summary = {
        'experiment_name': args.experiment_name,
        'dataset': args.dataset,
        'context_len': args.context_len,
        'pred_len': args.pred_len,
        'features': args.features,
        'epochs': args.epochs,
        'best_test_loss': float(best_test_loss),
        'model_params': sum(p.numel() for p in model.parameters())
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
