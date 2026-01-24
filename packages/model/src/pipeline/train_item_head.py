"""
Train Per-Item Head for Cannonball
===================================

Fine-tunes a small specialized head on top of the frozen PatchTST base.

Usage:
    python -m src.pipeline.train_item_head \
        --base-model /path/to/patchtst.pt \
        --data /path/to/chunk.npz \
        --item-idx 0 \
        --output /path/to/cannonball_head.pt
"""

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, Tuple

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig
from src.pipeline.per_item_head import (
    PerItemHead,
    PatchTSTWithItemHead,
    extract_item_samples,
    combined_loss
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_base_model(model_path: str, device: torch.device) -> PatchTSTModel:
    """Load the pre-trained PatchTST base model."""
    config = ModelConfig(n_items=1000)
    model = PatchTSTModel(config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def prepare_data(
    data_path: str,
    item_idx: int,
    val_split: float = 0.2
) -> Tuple[Dict, Dict]:
    """Load and split data for specific item."""
    data = np.load(data_path)
    item_data = extract_item_samples(dict(data), item_idx)

    n_samples = len(item_data['recent'])
    n_val = int(n_samples * val_split)
    indices = np.random.permutation(n_samples)

    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_data = {k: v[train_idx] for k, v in item_data.items()}
    val_data = {k: v[val_idx] for k, v in item_data.items()}

    return train_data, val_data


def create_dataloader(
    data: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader from numpy arrays."""
    recent = torch.tensor(data['recent'], dtype=torch.float32)
    medium = torch.tensor(data['medium'], dtype=torch.float32)
    long = torch.tensor(data['long'], dtype=torch.float32)
    item_ids = torch.tensor(data['item_ids'], dtype=torch.long)
    targets = torch.tensor(data['targets'], dtype=torch.float32)

    dataset = TensorDataset(recent, medium, long, item_ids, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(
    model: PatchTSTWithItemHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.item_head.train()
    total_loss = 0.0
    n_batches = 0

    for recent, medium, long, item_ids, targets in loader:
        recent = recent.to(device)
        medium = medium.to(device)
        long = long.to(device)
        item_ids = item_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(recent, medium, long, item_ids)
        loss = combined_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def run_validation(
    model: PatchTSTWithItemHead,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Run validation and compute metrics."""
    model.item_head.eval()
    total_loss = 0.0
    n_batches = 0

    all_high_pred = []
    all_low_pred = []
    all_targets = []

    with torch.no_grad():
        for recent, medium, long, item_ids, targets in loader:
            recent = recent.to(device)
            medium = medium.to(device)
            long = long.to(device)
            item_ids = item_ids.to(device)
            targets = targets.to(device)

            outputs = model(recent, medium, long, item_ids)
            loss = combined_loss(outputs, targets)

            total_loss += loss.item()
            n_batches += 1

            all_high_pred.append(outputs['high_quantiles'].cpu())
            all_low_pred.append(outputs['low_quantiles'].cpu())
            all_targets.append(targets.cpu())

    # Compute coverage metrics
    high_pred = torch.cat(all_high_pred)
    low_pred = torch.cat(all_low_pred)
    targets = torch.cat(all_targets)

    # Coverage: what fraction of actuals fall within p10-p90 range
    high_actual = targets[..., 0]
    low_actual = targets[..., 1]

    high_p10 = high_pred[..., 0]
    high_p90 = high_pred[..., 4]
    low_p10 = low_pred[..., 0]
    low_p90 = low_pred[..., 4]

    high_coverage = ((high_actual >= high_p10) & (high_actual <= high_p90)).float().mean()
    low_coverage = ((low_actual >= low_p10) & (low_actual <= low_p90)).float().mean()

    metrics = {
        'high_coverage': high_coverage.item(),
        'low_coverage': low_coverage.item(),
    }

    return total_loss / n_batches, metrics


def compare_with_base(
    base_model: PatchTSTModel,
    item_model: PatchTSTWithItemHead,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Compare per-item head with shared base head."""
    base_losses = []
    item_losses = []

    with torch.no_grad():
        for recent, medium, long, item_ids, targets in val_loader:
            recent = recent.to(device)
            medium = medium.to(device)
            long = long.to(device)
            item_ids = item_ids.to(device)
            targets = targets.to(device)

            # Base model predictions
            base_out = base_model(recent, medium, long, item_ids)
            base_loss = combined_loss(base_out, targets)
            base_losses.append(base_loss.item())

            # Per-item head predictions
            item_out = item_model(recent, medium, long, item_ids)
            item_loss = combined_loss(item_out, targets)
            item_losses.append(item_loss.item())

    return {
        'base_loss': np.mean(base_losses),
        'item_head_loss': np.mean(item_losses),
        'improvement': (np.mean(base_losses) - np.mean(item_losses)) / np.mean(base_losses) * 100
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True, help='Path to PatchTST base model')
    parser.add_argument('--data', required=True, help='Path to data chunk (.npz)')
    parser.add_argument('--item-idx', type=int, default=0, help='Item index (0 for Cannonball)')
    parser.add_argument('--output', default='/tmp/cannonball_head.pt', help='Output path')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load base model
    logger.info(f"Loading base model from {args.base_model}")
    base_model = load_base_model(args.base_model, device)
    d_model = base_model.config.d_model

    # Create per-item head
    logger.info(f"Creating per-item head (hidden_dim={args.hidden_dim})")
    item_head = PerItemHead(
        d_model=d_model,
        hidden_dim=args.hidden_dim,
        n_horizons=7,
        n_quantiles=5,
        dropout=0.1
    ).to(device)

    # Wrap base + item head
    model = PatchTSTWithItemHead(base_model, item_head, freeze_base=True)

    # Count parameters
    total_params = sum(p.numel() for p in item_head.parameters())
    logger.info(f"Per-item head parameters: {total_params:,}")

    # Prepare data
    logger.info(f"Loading data from {args.data}")
    train_data, val_data = prepare_data(args.data, args.item_idx)
    logger.info(f"Train samples: {len(train_data['recent'])}, Val samples: {len(val_data['recent'])}")

    train_loader = create_dataloader(train_data, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, args.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(item_head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    best_val_loss = float('inf')
    best_state = None

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, metrics = run_validation(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = item_head.state_dict()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | "
                f"High Cov: {metrics['high_coverage']:.1%} | "
                f"Low Cov: {metrics['low_coverage']:.1%}"
            )

    # Load best model
    item_head.load_state_dict(best_state)

    # Compare with base model
    logger.info("\n=== Comparison with Base Model ===")
    comparison = compare_with_base(base_model, model, val_loader, device)
    logger.info(f"Base model loss:     {comparison['base_loss']:.4f}")
    logger.info(f"Per-item head loss:  {comparison['item_head_loss']:.4f}")
    logger.info(f"Improvement:         {comparison['improvement']:.1f}%")

    # Save
    torch.save(best_state, args.output)
    logger.info(f"\nSaved per-item head to {args.output}")


if __name__ == '__main__':
    main()
