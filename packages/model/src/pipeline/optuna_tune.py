"""
Optuna Hyperparameter Tuning for PatchTST
==========================================

Runs systematic hyperparameter search using Bayesian optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from glob import glob
from pathlib import Path
import logging
import yaml
import json
import optuna
from optuna.trial import TrialState
import gc
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.pipeline.model import PatchTSTModel, combined_quantile_loss
from src.pipeline.config import ModelConfig


def load_all_chunks(chunk_dir, subsample_ratio=1.0):
    """Load all chunks into memory with memory-efficient concatenation."""
    chunk_paths = sorted(glob(str(chunk_dir / 'chunk_*.npz')))

    if subsample_ratio < 1.0:
        n_chunks = max(1, int(len(chunk_paths) * subsample_ratio))
        step = len(chunk_paths) // n_chunks
        chunk_paths = chunk_paths[::step][:n_chunks]

    logger.info(f'Loading {len(chunk_paths)} chunks from {chunk_dir}...')

    result = {}
    for array_name in ['recent', 'medium', 'long', 'item_ids', 'targets']:
        arrays = []
        for i, path in enumerate(chunk_paths):
            data = np.load(path)
            arrays.append(data[array_name])
            del data
        result[array_name] = np.concatenate(arrays, axis=0)
        del arrays
        gc.collect()

    return result


def create_model(trial, config):
    """Create model with trial-suggested hyperparameters."""

    # Architecture hyperparameters
    d_model = trial.suggest_categorical('d_model', [256, 384, 512])
    n_layers = trial.suggest_int('n_layers', 4, 8)
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.05)

    model_config = ModelConfig(
        n_items=config['model']['n_items'],
        item_embed_dim=trial.suggest_categorical('item_embed_dim', [16, 32, 64]),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )

    return PatchTSTModel(model_config)


def train_epoch(model, loader, optimizer, scaler, device, quantiles):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for recent, medium, long, item_ids, targets in loader:
        recent = recent.to(device, non_blocking=True)
        medium = medium.to(device, non_blocking=True)
        long = long.to(device, non_blocking=True)
        item_ids = item_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(recent, medium, long, item_ids)
            loss = combined_quantile_loss(outputs, targets, quantiles)

        if not torch.isnan(loss) and not torch.isinf(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            valid_batches += 1

    return total_loss / max(valid_batches, 1)


def validate_model(model, loader, device, quantiles):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for recent, medium, long, item_ids, targets in loader:
            recent = recent.to(device, non_blocking=True)
            medium = medium.to(device, non_blocking=True)
            long = long.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(recent, medium, long, item_ids)
                loss = combined_quantile_loss(outputs, targets, quantiles)

            if not torch.isnan(loss):
                total_loss += loss.item()

    return total_loss / len(loader)


def objective(trial, train_dataset, val_dataset, config, device):
    """Optuna objective function."""

    # Training hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 768, 1024])
    warmup_steps = trial.suggest_int('warmup_steps', 100, 1000, step=100)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Create model
    model = create_model(trial, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trial {trial.number}: {n_params:,} parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler()
    quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)

    # Training with early stopping
    max_epochs = 25
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, quantiles)
        val_loss = validate_model(model, val_loader, device, quantiles)

        logger.info(f'Trial {trial.number} Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}')

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f'Trial {trial.number}: Early stopping at epoch {epoch+1}')
                break

    # Cleanup
    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--study_name', type=str, default='patchtst_tuning')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    features_dir = Path(config['output_dir']) / 'features'

    # Load data once
    logger.info('Loading training data...')
    train_data = load_all_chunks(features_dir / 'train')
    logger.info(f'Train samples: {len(train_data["recent"]):,}')

    logger.info('Loading validation data (50% subsample)...')
    val_data = load_all_chunks(features_dir / 'val', subsample_ratio=0.5)
    logger.info(f'Val samples: {len(val_data["recent"]):,}')

    # Convert to tensors
    logger.info('Converting to tensors...')
    train_tensors = []
    for key in ['recent', 'medium', 'long', 'item_ids', 'targets']:
        dtype = torch.long if key == 'item_ids' else torch.float32
        train_tensors.append(torch.tensor(train_data[key], dtype=dtype))
        del train_data[key]
        gc.collect()
    del train_data
    train_dataset = TensorDataset(*train_tensors)
    del train_tensors
    gc.collect()

    val_tensors = []
    for key in ['recent', 'medium', 'long', 'item_ids', 'targets']:
        dtype = torch.long if key == 'item_ids' else torch.float32
        val_tensors.append(torch.tensor(val_data[key], dtype=dtype))
        del val_data[key]
        gc.collect()
    del val_data
    val_dataset = TensorDataset(*val_tensors)
    del val_tensors
    gc.collect()

    logger.info(f'Train: {len(train_dataset):,}, Val: {len(val_dataset):,}')

    # Create Optuna study with SQLite WAL mode for better concurrency
    db_path = f'/workspace/optuna_{args.study_name}.db'
    storage = optuna.storages.RDBStorage(
        url=f'sqlite:///{db_path}',
        engine_kwargs={'connect_args': {'timeout': 30}},  # 30s timeout for locks
    )

    # Enable WAL mode for concurrent access
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.close()

    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=storage,
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, config, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Report results
    logger.info('\n' + '='*60)
    logger.info('OPTUNA TUNING COMPLETE')
    logger.info('='*60)

    logger.info(f'\nBest trial: {study.best_trial.number}')
    logger.info(f'Best val_loss: {study.best_trial.value:.4f}')
    logger.info('\nBest hyperparameters:')
    for key, value in study.best_trial.params.items():
        logger.info(f'  {key}: {value}')

    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'timestamp': datetime.utcnow().isoformat(),
    }

    results_path = Path('/workspace/optuna_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to {results_path}')

    # Also save all trials
    trials_data = []
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            trials_data.append({
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
            })

    with open('/workspace/optuna_all_trials.json', 'w') as f:
        json.dump(trials_data, f, indent=2)


if __name__ == '__main__':
    main()
