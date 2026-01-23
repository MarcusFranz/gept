"""
Stage 4: Training
=================

Train PatchTST model on precomputed feature chunks.

Usage:
    python -m src.pipeline.stage4_train --config configs/research_small.yaml
    python -m src.pipeline.stage4_train --config configs/research_small.yaml --resume
    python -m src.pipeline.stage4_train --config configs/research_small.yaml --test-only
"""

import argparse
import json
import logging
import math
import shutil
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.pipeline.config import PipelineConfig, ModelConfig
from src.pipeline.model import PatchTSTModel, combined_quantile_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkDataset(Dataset):
    """
    Dataset that loads precomputed .npz feature chunks.

    Each chunk contains ~10K samples with:
    - recent: (chunk_size, 288, 6) - 5-min features
    - medium: (chunk_size, 168, 10) - 1-hour features
    - long: (chunk_size, 180, 10) - 4-hour features
    - item_ids: (chunk_size,) - item indices
    - targets: (chunk_size, 7, 2) - (max_high, min_low) per horizon
    """

    def __init__(self, chunk_dir: Path):
        """
        Args:
            chunk_dir: Directory containing chunk_*.npz files
        """
        self.chunk_dir = Path(chunk_dir)
        self.chunk_paths = sorted(glob(str(self.chunk_dir / "chunk_*.npz")))

        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunks found in {chunk_dir}")

        # Load first chunk to get size
        with np.load(self.chunk_paths[0]) as data:
            self.chunk_size = len(data['item_ids'])

        # Calculate total samples (last chunk might be smaller)
        self.n_chunks = len(self.chunk_paths)
        with np.load(self.chunk_paths[-1]) as data:
            last_chunk_size = len(data['item_ids'])

        self.total_samples = (self.n_chunks - 1) * self.chunk_size + last_chunk_size

        # Cache for current chunk
        self._cache_idx = -1
        self._cache_data = None

        logger.info(f"ChunkDataset: {self.n_chunks} chunks, "
                    f"{self.total_samples:,} samples, chunk_size={self.chunk_size}")

    def __len__(self) -> int:
        return self.total_samples

    def _load_chunk(self, chunk_idx: int) -> None:
        """Load a chunk into cache."""
        if chunk_idx != self._cache_idx:
            self._cache_data = np.load(self.chunk_paths[chunk_idx])
            self._cache_idx = chunk_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk_idx = idx // self.chunk_size
        sample_idx = idx % self.chunk_size

        self._load_chunk(chunk_idx)

        return {
            'recent': torch.tensor(self._cache_data['recent'][sample_idx], dtype=torch.float32),
            'medium': torch.tensor(self._cache_data['medium'][sample_idx], dtype=torch.float32),
            'long': torch.tensor(self._cache_data['long'][sample_idx], dtype=torch.float32),
            'item_id': torch.tensor(self._cache_data['item_ids'][sample_idx], dtype=torch.long),
            'targets': torch.tensor(self._cache_data['targets'][sample_idx], dtype=torch.float32),
        }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    quantiles: tuple,
    use_amp: bool = True,
    log_every: int = 50
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Determine device type for autocast
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    # MPS doesn't support autocast well, use CPU autocast or disable
    if device.type == 'mps':
        use_amp = False

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        recent = batch['recent'].to(device, non_blocking=True)
        medium = batch['medium'].to(device, non_blocking=True)
        long_seq = batch['long'].to(device, non_blocking=True)
        item_ids = batch['item_id'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(recent, medium, long_seq, item_ids)
            loss = combined_quantile_loss(outputs, targets, quantiles)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}, lr={lr:.2e}")

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    quantiles: tuple,
    use_amp: bool = True
) -> Dict[str, float]:
    """Run validation."""
    model.train(False)
    total_loss = 0.0
    n_batches = 0

    # Determine device type for autocast
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    # MPS doesn't support autocast well
    if device.type == 'mps':
        use_amp = False

    for batch in dataloader:
        recent = batch['recent'].to(device, non_blocking=True)
        medium = batch['medium'].to(device, non_blocking=True)
        long_seq = batch['long'].to(device, non_blocking=True)
        item_ids = batch['item_id'].to(device, non_blocking=True)
        targets = batch['targets'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(recent, medium, long_seq, item_ids)
            loss = combined_quantile_loss(outputs, targets, quantiles)

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / n_batches}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    val_loss: float,
    path: Path
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Train PatchTST model")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--test-only', action='store_true', help='Only run validation')
    parser.add_argument('--data-dir', type=str, help='Override data directory from config')
    args = parser.parse_args()

    # Load config
    config = PipelineConfig.from_yaml(args.config)

    # Override data dir if specified
    if args.data_dir:
        config.output_dir = Path(args.data_dir)

    features_dir = config.output_dir / "features"
    model_dir = config.model_dir

    logger.info(f"Config: {args.config}")
    logger.info(f"Features: {features_dir}")
    logger.info(f"Model output: {model_dir}")

    # Check GPU (MPS > CUDA > CPU)
    device = get_device()
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
    elif device.type == 'mps':
        logger.info("Using Apple Metal (MPS) acceleration")

    # Create datasets
    train_dataset = ChunkDataset(features_dir / "train")
    val_dataset = ChunkDataset(features_dir / "val")

    # pin_memory only works with CUDA
    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory
    )

    # Create model
    model_config = ModelConfig(
        recent_len=config.model.recent_len,
        recent_features=config.model.recent_features,
        medium_len=config.model.medium_len,
        medium_features=config.model.medium_features,
        long_len=config.model.long_len,
        long_features=config.model.long_features,
        patch_size=config.model.patch_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        n_horizons=config.model.n_horizons,
        n_quantiles=config.model.n_quantiles,
        quantiles=config.model.quantiles,
        n_items=config.model.n_items,
        item_embed_dim=config.model.item_embed_dim,
    )
    model = PatchTSTModel(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )

    total_steps = len(train_loader) * config.training.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=total_steps
    )

    # Only use GradScaler with CUDA (not supported on MPS)
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None
    if scaler is None:
        logger.info("GradScaler disabled (not using CUDA)")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "latest_checkpoint.pt"

    if args.resume and checkpoint_path.exists():
        logger.info(f"Resuming from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Test-only mode
    if args.test_only:
        logger.info("Running validation only...")
        val_metrics = validate(model, val_loader, device, config.model.quantiles)
        logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
        return 0

    # Training loop
    training_log = []
    quantiles = config.model.quantiles

    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{config.training.epochs} ===")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, quantiles, use_amp=True
        )

        # Validate
        val_metrics = validate(model, val_loader, device, quantiles)
        val_loss = val_metrics['loss']

        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Log
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'timestamp': datetime.utcnow().isoformat()
        })

        # Save checkpoint
        if (epoch + 1) % config.training.checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = model_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved (val_loss={val_loss:.4f})")

    # Save final model
    final_path = model_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")

    # Save training log
    log_path = model_dir / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    # Copy config
    shutil.copy(args.config, model_dir / "config.yaml")

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
