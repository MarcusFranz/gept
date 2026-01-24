"""
Training script for PatchTST using precomputed features.

This script loads precomputed numpy arrays for fast GPU-bound training.
Optimized for RTX 5090 with large batch sizes.

Usage:
    python train_precomputed.py --data-dir /workspace/gept/precomputed --checkpoint-dir /workspace/gept/models
"""

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== Model Architecture ==============

@dataclass
class ModelConfig:
    """Configuration for PatchTST model."""
    recent_len: int = 288
    recent_features: int = 6
    medium_len: int = 168
    medium_features: int = 10
    long_len: int = 180
    long_features: int = 10

    patch_size: int = 16
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 768
    dropout: float = 0.1

    n_horizons: int = 7
    n_quantiles: int = 5
    quantiles: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)

    n_items: int = 5000
    item_embed_dim: int = 32

    @property
    def total_patches(self) -> int:
        return (self.recent_len // self.patch_size +
                self.medium_len // self.patch_size +
                self.long_len // self.patch_size)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, n_features: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        pad_len = (self.patch_size - seq_len % self.patch_size) % self.patch_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        n_patches = (seq_len + pad_len) // self.patch_size
        x = x.reshape(batch_size, n_patches, self.patch_size * n_features)
        return self.projection(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResolutionEncoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, n_features, d_model)
        self.resolution_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(x)
        return patches + self.resolution_embed


class QuantileHead(nn.Module):
    def __init__(self, d_model: int, n_horizons: int, n_quantiles: int):
        super().__init__()
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_quantiles)
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [head(x) for head in self.horizon_heads]
        return torch.stack(outputs, dim=1)


class PatchTSTModel(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        self.recent_encoder = ResolutionEncoder(c.recent_len, c.recent_features, c.patch_size, c.d_model)
        self.medium_encoder = ResolutionEncoder(c.medium_len, c.medium_features, c.patch_size, c.d_model)
        self.long_encoder = ResolutionEncoder(c.long_len, c.long_features, c.patch_size, c.d_model)

        self.pos_encoder = PositionalEncoding(c.d_model, c.total_patches + 10, c.dropout)

        self.item_embedding = nn.Embedding(c.n_items, c.item_embed_dim)
        self.item_proj = nn.Linear(c.item_embed_dim, c.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.n_heads, dim_feedforward=c.d_ff,
            dropout=c.dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=c.n_layers)

        self.pool_norm = nn.LayerNorm(c.d_model)
        self.high_head = QuantileHead(c.d_model, c.n_horizons, c.n_quantiles)
        self.low_head = QuantileHead(c.d_model, c.n_horizons, c.n_quantiles)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, recent_5m, medium_1h, long_4h, item_ids=None, current_high=None, current_low=None):
        recent_patches = self.recent_encoder(recent_5m)
        medium_patches = self.medium_encoder(medium_1h)
        long_patches = self.long_encoder(long_4h)

        all_patches = torch.cat([recent_patches, medium_patches, long_patches], dim=1)

        if item_ids is not None:
            item_embed = self.item_embedding(item_ids)
            item_token = self.item_proj(item_embed).unsqueeze(1)
            all_patches = torch.cat([item_token, all_patches], dim=1)

        x = self.pos_encoder(all_patches)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.pool_norm(x)

        high_q = self.high_head(x)
        low_q = self.low_head(x)

        if current_high is not None and current_low is not None:
            current_high = current_high.unsqueeze(-1).unsqueeze(-1)
            current_low = current_low.unsqueeze(-1).unsqueeze(-1)
            high_q = high_q * current_high
            low_q = low_q * current_low

        return {'high_quantiles': high_q, 'low_quantiles': low_q}


# ============== Dataset ==============

class PrecomputedDataset(Dataset):
    """Dataset that loads precomputed numpy arrays."""

    def __init__(self, data_dir: Path, split: str = 'train'):
        self.data_dir = Path(data_dir)

        # Load metadata
        with open(self.data_dir / f"{split}_metadata.json", 'r') as f:
            self.metadata = json.load(f)

        self.n_samples = self.metadata['n_samples']
        logger.info(f"Loading {split} dataset: {self.n_samples:,} samples")

        # Load memory-mapped arrays for fast access
        recent_shape = tuple(self.metadata['recent_shape'])
        medium_shape = tuple(self.metadata['medium_shape'])
        long_shape = tuple(self.metadata['long_shape'])

        self.recent = np.memmap(self.data_dir / f"{split}_recent.npy", dtype=np.float32, mode='r', shape=recent_shape)
        self.medium = np.memmap(self.data_dir / f"{split}_medium.npy", dtype=np.float32, mode='r', shape=medium_shape)
        self.long = np.memmap(self.data_dir / f"{split}_long.npy", dtype=np.float32, mode='r', shape=long_shape)

        self.item_idx = np.load(self.data_dir / f"{split}_item_idx.npy")
        self.current_high = np.load(self.data_dir / f"{split}_current_high.npy")
        self.current_low = np.load(self.data_dir / f"{split}_current_low.npy")
        self.targets_min_low = np.load(self.data_dir / f"{split}_targets_min_low.npy")
        self.targets_max_high = np.load(self.data_dir / f"{split}_targets_max_high.npy")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'item_id': torch.tensor(self.item_idx[idx], dtype=torch.long),
            'recent_5m': torch.tensor(self.recent[idx], dtype=torch.float32),
            'medium_1h': torch.tensor(self.medium[idx], dtype=torch.float32),
            'long_4h': torch.tensor(self.long[idx], dtype=torch.float32),
            'current_high': torch.tensor(self.current_high[idx], dtype=torch.float32),
            'current_low': torch.tensor(self.current_low[idx], dtype=torch.float32),
            'targets_low': torch.tensor(self.targets_min_low[idx], dtype=torch.float32),
            'targets_high': torch.tensor(self.targets_max_high[idx], dtype=torch.float32),
        }


# ============== Loss Functions ==============

def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    target = target.unsqueeze(-1).expand_as(pred)
    quantiles_t = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    quantiles_t = quantiles_t.view(1, 1, -1)

    errors = target - pred
    loss = torch.max(quantiles_t * errors, (quantiles_t - 1) * errors)
    return loss.mean()


def combined_loss(outputs: dict, targets_high: torch.Tensor, targets_low: torch.Tensor,
                  current_high: torch.Tensor, current_low: torch.Tensor,
                  quantiles: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)) -> dict:
    """Combined quantile loss for high and low predictions."""
    high_q = outputs['high_quantiles']
    low_q = outputs['low_quantiles']

    # Normalize targets by current price for ratio-based learning
    current_high = current_high.unsqueeze(-1)
    current_low = current_low.unsqueeze(-1)

    high_ratio = targets_high / (current_high + 1e-6)
    low_ratio = targets_low / (current_low + 1e-6)

    # Normalize predictions too (undo the scaling in forward)
    high_q_ratio = high_q / (current_high.unsqueeze(-1) + 1e-6)
    low_q_ratio = low_q / (current_low.unsqueeze(-1) + 1e-6)

    high_loss = quantile_loss(high_q_ratio, high_ratio, quantiles)
    low_loss = quantile_loss(low_q_ratio, low_ratio, quantiles)

    total_loss = high_loss + low_loss

    return {
        'total': total_loss,
        'high': high_loss,
        'low': low_loss,
        'quantile': total_loss,
    }


# ============== Training ==============

@dataclass
class TrainingConfig:
    data_dir: str = "/workspace/gept/precomputed"
    checkpoint_dir: str = "/workspace/gept/models"

    # Model
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 768
    dropout: float = 0.1
    n_items: int = 500

    # Training - optimized for RTX 5090
    batch_size: int = 512  # Large batch for 5090
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_every: int = 50
    save_every: int = 5  # Save every 5 epochs


def train_epoch(model, dataloader, optimizer, scheduler, scaler, config, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        recent = batch['recent_5m'].to(device, non_blocking=True)
        medium = batch['medium_1h'].to(device, non_blocking=True)
        long_seq = batch['long_4h'].to(device, non_blocking=True)
        item_ids = batch['item_id'].to(device, non_blocking=True)
        current_high = batch['current_high'].to(device, non_blocking=True)
        current_low = batch['current_low'].to(device, non_blocking=True)
        targets_high = batch['targets_high'].to(device, non_blocking=True)
        targets_low = batch['targets_low'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=config.use_amp):
            outputs = model(recent, medium, long_seq, item_ids, current_high, current_low)
            losses = combined_loss(outputs, targets_high, targets_low, current_high, current_low)
            loss = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % config.log_every == 0:
            lr = scheduler.get_last_lr()[0]
            logger.info(f"  Batch {batch_idx}/{len(dataloader)}: loss={loss.item():.4f}, q={losses['quantile'].item():.4f}, lr={lr:.2e}")

    return total_loss / n_batches


@torch.no_grad()
def run_validation(model, dataloader, device, config):
    """Run validation on the model."""
    model.eval()
    total_loss = 0
    total_high_loss = 0
    total_low_loss = 0
    n_batches = 0

    for batch in dataloader:
        recent = batch['recent_5m'].to(device, non_blocking=True)
        medium = batch['medium_1h'].to(device, non_blocking=True)
        long_seq = batch['long_4h'].to(device, non_blocking=True)
        item_ids = batch['item_id'].to(device, non_blocking=True)
        current_high = batch['current_high'].to(device, non_blocking=True)
        current_low = batch['current_low'].to(device, non_blocking=True)
        targets_high = batch['targets_high'].to(device, non_blocking=True)
        targets_low = batch['targets_low'].to(device, non_blocking=True)

        with autocast(enabled=config.use_amp):
            outputs = model(recent, medium, long_seq, item_ids, current_high, current_low)
            losses = combined_loss(outputs, targets_high, targets_low, current_high, current_low)

        total_loss += losses['total'].item()
        total_high_loss += losses['high'].item()
        total_low_loss += losses['low'].item()
        n_batches += 1

    return {
        'total': total_loss / n_batches,
        'high': total_high_loss / n_batches,
        'low': total_low_loss / n_batches,
    }


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/workspace/gept/precomputed')
    parser.add_argument('--checkpoint-dir', type=str, default='/workspace/gept/models')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = PrecomputedDataset(config.data_dir, 'train')
    val_dataset = PrecomputedDataset(config.data_dir, 'val')

    # Create dataloaders - many workers for fast loading
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model_config = ModelConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        n_items=config.n_items,
    )
    model = PatchTSTModel(model_config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    scaler = GradScaler(enabled=config.use_amp)

    # Resume if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting Training with Precomputed Data")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Total epochs: {config.epochs}")
    logger.info(f"AMP enabled: {config.use_amp}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, config.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, config, device, epoch)
        logger.info(f"Train loss: {train_loss:.4f}")

        # Validate every epoch
        val_losses = run_validation(model, val_loader, device, config)
        logger.info(f"Val loss: {val_losses['total']:.4f} (high={val_losses['high']:.4f}, low={val_losses['low']:.4f})")

        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': model_config.__dict__,
            }, Path(config.checkpoint_dir) / 'best_model.pt')
            logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % config.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': model_config.__dict__,
            }, Path(config.checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pt')

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
