"""
Optuna Hyperparameter Tuning with Volume Prediction
====================================================

Two-stage calibration:
1. First N_CALIBRATION_TRIALS complete without volume pruning
2. Subsequent trials pruned if val_volume_loss > median + MAD

Objective: val_price_loss (not total loss)
Volume weight is fixed at VOLUME_WEIGHT = 0.2
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
from typing import Optional, Tuple, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.pipeline.model import PatchTSTModel, combined_loss_with_volume
from src.pipeline.config import ModelConfig

# Fixed constants
VOLUME_WEIGHT = 0.2
N_CALIBRATION_TRIALS = 8
PRICE_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
VOLUME_QUANTILES = (0.1, 0.5, 0.9)


def load_all_chunks_with_volume(chunk_dir: Path, subsample_ratio: float = 1.0) -> Dict[str, np.ndarray]:
    """Load all chunks including volume_targets into memory."""
    chunk_paths = sorted(glob(str(chunk_dir / 'chunk_*.npz')))

    if subsample_ratio < 1.0:
        n_chunks = max(1, int(len(chunk_paths) * subsample_ratio))
        step = len(chunk_paths) // n_chunks
        chunk_paths = chunk_paths[::step][:n_chunks]

    logger.info(f'Loading {len(chunk_paths)} chunks from {chunk_dir}...')

    array_names = ['recent', 'medium', 'long', 'item_ids', 'targets', 'volume_targets']
    result = {}

    for array_name in array_names:
        arrays = []
        for i, path in enumerate(chunk_paths):
            data = np.load(path)
            if array_name in data:
                arrays.append(data[array_name])
            del data

        if arrays:
            result[array_name] = np.concatenate(arrays, axis=0)
        del arrays
        gc.collect()

    return result


def create_model_with_volume(trial, config: dict) -> Tuple[PatchTSTModel, ModelConfig]:
    """Create model with trial-suggested hyperparameters and volume head enabled."""

    # Architecture hyperparameters
    d_model = trial.suggest_categorical('d_model', [256, 384, 512])
    n_layers = trial.suggest_int('n_layers', 4, 8)
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.05)
    item_embed_dim = trial.suggest_categorical('item_embed_dim', [16, 32, 64])

    model_config = ModelConfig(
        n_items=config['model']['n_items'],
        item_embed_dim=item_embed_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        # Volume head settings
        enable_volume_head=True,
        volume_quantiles=VOLUME_QUANTILES,
        volume_hidden_dim=64,
    )

    return PatchTSTModel(model_config), model_config


def train_epoch_with_volume(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    model_config: ModelConfig,
) -> Tuple[float, float, float]:
    """Train for one epoch with volume loss tracking.

    Returns:
        Tuple of (total_loss, price_loss, volume_loss)
    """
    model.train()
    total_loss_sum = 0.0
    price_loss_sum = 0.0
    volume_loss_sum = 0.0
    valid_batches = 0

    for batch in loader:
        recent, medium, long, item_ids, targets, volume_targets = batch
        recent = recent.to(device, non_blocking=True)
        medium = medium.to(device, non_blocking=True)
        long = long.to(device, non_blocking=True)
        item_ids = item_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        volume_targets = volume_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda'):
            outputs = model(recent, medium, long, item_ids)
            loss, loss_dict = combined_loss_with_volume(
                outputs, targets, volume_targets, model_config, volume_weight=VOLUME_WEIGHT
            )

        if not torch.isnan(loss) and not torch.isinf(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += loss_dict['total_loss']
            price_loss_sum += loss_dict['price_loss']
            volume_loss_sum += loss_dict['volume_loss']
            valid_batches += 1

    n = max(valid_batches, 1)
    return total_loss_sum / n, price_loss_sum / n, volume_loss_sum / n


def validate_with_volume(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_config: ModelConfig,
) -> Tuple[float, float, float]:
    """Validate model with volume loss tracking.

    Returns:
        Tuple of (total_loss, price_loss, volume_loss)
    """
    model.eval()
    total_loss_sum = 0.0
    price_loss_sum = 0.0
    volume_loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            recent, medium, long, item_ids, targets, volume_targets = batch
            recent = recent.to(device, non_blocking=True)
            medium = medium.to(device, non_blocking=True)
            long = long.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            volume_targets = volume_targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                outputs = model(recent, medium, long, item_ids)
                loss, loss_dict = combined_loss_with_volume(
                    outputs, targets, volume_targets, model_config, volume_weight=VOLUME_WEIGHT
                )

            if not torch.isnan(loss):
                total_loss_sum += loss_dict['total_loss']
                price_loss_sum += loss_dict['price_loss']
                volume_loss_sum += loss_dict['volume_loss']

    n = len(loader)
    return total_loss_sum / n, price_loss_sum / n, volume_loss_sum / n


class VolumeCalibrationPruner:
    """
    Two-stage pruning for volume-aware training.

    Stage 1 (calibration): First N_CALIBRATION_TRIALS run to completion
    Stage 2 (pruning): Prune if val_volume_loss > median + MAD of calibration trials
    """

    def __init__(self, n_calibration_trials: int = N_CALIBRATION_TRIALS):
        self.n_calibration_trials = n_calibration_trials
        self.calibration_volume_losses: List[float] = []
        self.volume_threshold: Optional[float] = None

    def record_calibration_trial(self, val_volume_loss: float) -> None:
        """Record volume loss from a calibration trial."""
        self.calibration_volume_losses.append(val_volume_loss)

        # Once we have enough calibration trials, compute threshold
        if len(self.calibration_volume_losses) >= self.n_calibration_trials:
            self._compute_threshold()

    def _compute_threshold(self) -> None:
        """Compute pruning threshold as median + MAD."""
        losses = np.array(self.calibration_volume_losses)
        median = np.median(losses)
        mad = np.median(np.abs(losses - median))
        self.volume_threshold = median + mad
        logger.info(f"Volume pruning threshold set: {self.volume_threshold:.4f} "
                    f"(median={median:.4f}, MAD={mad:.4f})")

    def should_prune(self, val_volume_loss: float) -> bool:
        """Check if trial should be pruned based on volume loss."""
        if self.volume_threshold is None:
            return False  # Still in calibration phase
        return val_volume_loss > self.volume_threshold

    @property
    def is_calibrated(self) -> bool:
        return self.volume_threshold is not None


def objective_with_volume(
    trial,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    config: dict,
    device: torch.device,
    volume_pruner: VolumeCalibrationPruner,
) -> float:
    """Optuna objective function with volume-aware training and pruning.

    Returns val_price_loss (not total_loss) as the optimization objective.
    """

    # Training hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.001, 0.1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 768, 1024])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Create model with volume head
    model, model_config = create_model_with_volume(trial, config)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trial {trial.number}: {n_params:,} parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler()

    # Training with early stopping
    max_epochs = 25
    patience = 5
    best_val_price_loss = float('inf')
    best_val_volume_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        train_total, train_price, train_vol = train_epoch_with_volume(
            model, train_loader, optimizer, scaler, device, model_config
        )
        val_total, val_price, val_vol = validate_with_volume(
            model, val_loader, device, model_config
        )

        logger.info(f'Trial {trial.number} Epoch {epoch+1}: '
                    f'train_price={train_price:.4f}, val_price={val_price:.4f}, '
                    f'train_vol={train_vol:.4f}, val_vol={val_vol:.4f}')

        # Report price loss to Optuna (this is our objective)
        trial.report(val_price, epoch)

        # Check Optuna's built-in pruning (MedianPruner on price loss)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Check volume-based pruning (only after calibration)
        if volume_pruner.should_prune(val_vol):
            logger.info(f'Trial {trial.number}: Pruned due to high volume loss '
                        f'({val_vol:.4f} > {volume_pruner.volume_threshold:.4f})')
            raise optuna.TrialPruned()

        # Early stopping based on price loss
        if val_price < best_val_price_loss:
            best_val_price_loss = val_price
            best_val_volume_loss = val_vol
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f'Trial {trial.number}: Early stopping at epoch {epoch+1}')
                break

    # Record volume loss for calibration (if not pruned)
    if not volume_pruner.is_calibrated:
        volume_pruner.record_calibration_trial(best_val_volume_loss)

    # Store volume loss as user attribute for analysis
    trial.set_user_attr('val_volume_loss', best_val_volume_loss)
    trial.set_user_attr('val_price_loss', best_val_price_loss)

    # Cleanup
    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    # Return price loss as objective (not total loss)
    return best_val_price_loss


def main():
    parser = argparse.ArgumentParser(description='Optuna tuning with volume prediction')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--study_name', type=str, default='patchtst_volume_tuning')
    parser.add_argument('--val_subsample', type=float, default=0.5,
                        help='Subsample ratio for validation data (default: 0.5)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Volume weight: {VOLUME_WEIGHT}')
    logger.info(f'Calibration trials: {N_CALIBRATION_TRIALS}')

    features_dir = Path(config['output_dir']) / 'features'

    # Load data with volume targets
    logger.info('Loading training data with volume targets...')
    train_data = load_all_chunks_with_volume(features_dir / 'train')
    logger.info(f'Train samples: {len(train_data["recent"]):,}')

    if 'volume_targets' not in train_data:
        raise ValueError("No volume_targets in training data. Run stage3_precompute with --enable-volume")

    logger.info(f'Loading validation data ({args.val_subsample*100:.0f}% subsample)...')
    val_data = load_all_chunks_with_volume(features_dir / 'val', subsample_ratio=args.val_subsample)
    logger.info(f'Val samples: {len(val_data["recent"]):,}')

    # Convert to tensors (including volume_targets)
    logger.info('Converting to tensors...')
    array_keys = ['recent', 'medium', 'long', 'item_ids', 'targets', 'volume_targets']

    train_tensors = []
    for key in array_keys:
        dtype = torch.long if key == 'item_ids' else torch.float32
        train_tensors.append(torch.tensor(train_data[key], dtype=dtype))
        del train_data[key]
        gc.collect()
    del train_data
    train_dataset = TensorDataset(*train_tensors)
    del train_tensors
    gc.collect()

    val_tensors = []
    for key in array_keys:
        dtype = torch.long if key == 'item_ids' else torch.float32
        val_tensors.append(torch.tensor(val_data[key], dtype=dtype))
        del val_data[key]
        gc.collect()
    del val_data
    val_dataset = TensorDataset(*val_tensors)
    del val_tensors
    gc.collect()

    logger.info(f'Train: {len(train_dataset):,}, Val: {len(val_dataset):,}')

    # Create Optuna study
    db_path = f'/workspace/optuna_{args.study_name}.db'
    storage = optuna.storages.RDBStorage(
        url=f'sqlite:///{db_path}',
        engine_kwargs={'connect_args': {'timeout': 30}},
    )

    # Enable WAL mode for concurrent access
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.close()

    study = optuna.create_study(
        study_name=args.study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=storage,
        load_if_exists=True,
    )

    # Volume calibration pruner
    volume_pruner = VolumeCalibrationPruner(n_calibration_trials=N_CALIBRATION_TRIALS)

    # Run optimization
    study.optimize(
        lambda trial: objective_with_volume(
            trial, train_dataset, val_dataset, config, device, volume_pruner
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # Report results
    logger.info('\n' + '='*60)
    logger.info('OPTUNA TUNING WITH VOLUME COMPLETE')
    logger.info('='*60)

    logger.info(f'\nBest trial: {study.best_trial.number}')
    logger.info(f'Best val_price_loss: {study.best_trial.value:.4f}')

    if 'val_volume_loss' in study.best_trial.user_attrs:
        logger.info(f'Best val_volume_loss: {study.best_trial.user_attrs["val_volume_loss"]:.4f}')

    logger.info('\nBest hyperparameters:')
    for key, value in study.best_trial.params.items():
        logger.info(f'  {key}: {value}')

    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_price_loss': study.best_trial.value,
        'best_volume_loss': study.best_trial.user_attrs.get('val_volume_loss'),
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'volume_weight': VOLUME_WEIGHT,
        'calibration_trials': N_CALIBRATION_TRIALS,
        'timestamp': datetime.utcnow().isoformat(),
    }

    results_path = Path('/workspace/optuna_volume_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to {results_path}')

    # Save all trials with volume metrics
    trials_data = []
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            trials_data.append({
                'number': trial.number,
                'price_loss': trial.value,
                'volume_loss': trial.user_attrs.get('val_volume_loss'),
                'params': trial.params,
            })

    with open('/workspace/optuna_volume_all_trials.json', 'w') as f:
        json.dump(trials_data, f, indent=2)


if __name__ == '__main__':
    main()
