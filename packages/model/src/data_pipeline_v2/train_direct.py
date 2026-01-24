"""
Direct training on RTX 5090 with on-the-fly feature computation.

Uses 64 dataloader workers to compute features in parallel.
With 256 cores, this should keep the 5090 fed.

Usage:
    python train_direct.py --data-dir /dev/shm/gept --epochs 50 --batch-size 512
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    recent_hours: int = 24      # 5-min resolution
    medium_days: int = 7        # 1-hour resolution
    long_days: int = 30         # 4-hour resolution

    @property
    def recent_len(self) -> int:
        return self.recent_hours * 12  # 288

    @property
    def medium_len(self) -> int:
        return self.medium_days * 24   # 168

    @property
    def long_len(self) -> int:
        return self.long_days * 6      # 180

    horizons: tuple = (1, 2, 4, 8, 12, 24, 48)


def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# Global data storage for workers (shared via fork)
_df_5min = None
_df_1h = None
_df_4h = None


def init_global_data(data_dir: Path):
    """Load data into global variables for sharing across workers."""
    global _df_5min, _df_1h, _df_4h

    logger.info(f"Loading data from {data_dir}")

    _df_5min = pd.read_parquet(data_dir / "price_data_5min.parquet")
    _df_5min['timestamp'] = pd.to_datetime(_df_5min['timestamp'], utc=True)
    _df_5min = _df_5min.set_index(['item_id', 'timestamp']).sort_index()

    _df_1h = pd.read_parquet(data_dir / "price_data_1h.parquet")
    _df_1h['timestamp'] = pd.to_datetime(_df_1h['timestamp'], utc=True)
    _df_1h = _df_1h.set_index(['item_id', 'timestamp']).sort_index()

    _df_4h = pd.read_parquet(data_dir / "price_data_4h.parquet")
    _df_4h['timestamp'] = pd.to_datetime(_df_4h['timestamp'], utc=True)
    _df_4h = _df_4h.set_index(['item_id', 'timestamp']).sort_index()

    logger.info(f"  5min: {len(_df_5min):,} rows")
    logger.info(f"  1h: {len(_df_1h):,} rows")
    logger.info(f"  4h: {len(_df_4h):,} rows")


def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros at start or truncate to target length."""
    if len(arr) >= target_len:
        return arr[-target_len:].astype(np.float32)
    else:
        padding = np.zeros((target_len - len(arr), arr.shape[1]), dtype=np.float32)
        return np.vstack([padding, arr]).astype(np.float32)


def prepare_5min_features(df: pd.DataFrame, target_len: int) -> np.ndarray:
    """Prepare 5-min features: log-normalized prices and volumes"""
    if len(df) == 0:
        return np.zeros((target_len, 6), dtype=np.float32)

    df = df.copy()
    df["avg_high_price"] = df["avg_high_price"].ffill()
    df["avg_low_price"] = df["avg_low_price"].ffill()

    # Log-transform prices and volumes to handle wide range (1 to millions)
    high_price = np.log1p(df["avg_high_price"].fillna(0).clip(lower=0).values)
    low_price = np.log1p(df["avg_low_price"].fillna(0).clip(lower=0).values)
    high_vol = np.log1p(df["high_price_volume"].fillna(0).clip(lower=0).values)
    low_vol = np.log1p(df["low_price_volume"].fillna(0).clip(lower=0).values)
    # Spread as ratio (normalized)
    spread = (df["avg_high_price"] - df["avg_low_price"]).fillna(0).values
    base = df["avg_high_price"].fillna(1).clip(lower=1).values
    spread_ratio = spread / base  # Spread relative to price
    staleness = df["avg_high_price"].isna().cumsum().values / 100.0  # Scale down

    features = np.column_stack([
        high_price, low_price, high_vol, low_vol, spread_ratio, staleness
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


def prepare_aggregated_features(df: pd.DataFrame, target_len: int) -> np.ndarray:
    """Prepare 1h/4h features from OHLC bars - log-normalized."""
    if len(df) == 0:
        return np.zeros((target_len, 10), dtype=np.float32)

    df = df.copy()

    high_close = df.get("high_close", pd.Series(0, index=df.index))
    low_close = df.get("low_close", pd.Series(0, index=df.index))
    high_vol = df.get("high_volume", pd.Series(0, index=df.index))
    low_vol = df.get("low_volume", pd.Series(0, index=df.index))
    sample_count = df.get("sample_count", pd.Series(12, index=df.index))
    high_high = df.get("high_high", high_close)
    high_low = df.get("high_low", high_close)
    low_high = df.get("low_high", low_close)
    low_low = df.get("low_low", low_close)

    high_close = high_close.ffill()
    low_close = low_close.ffill()

    # Log-transform prices and volumes
    hc = np.log1p(high_close.fillna(0).clip(lower=0).values)
    lc = np.log1p(low_close.fillna(0).clip(lower=0).values)
    hv = np.log1p(high_vol.fillna(0).clip(lower=0).values)
    lv = np.log1p(low_vol.fillna(0).clip(lower=0).values)
    # Spread as ratio
    base = high_close.fillna(1).clip(lower=1).values
    spread_ratio = (high_close - low_close).fillna(0).values / base
    # Normalized counts
    sc = sample_count.fillna(0).values / 12.0  # Normalize by expected count
    # High/low range ratios
    hh_hl_ratio = (high_high - high_low).fillna(0).values / base.clip(min=1)
    lh_ll_ratio = (low_high - low_low).fillna(0).values / base.clip(min=1)
    # Total volume (log)
    total_vol = np.log1p((high_vol + low_vol).fillna(0).clip(lower=0).values)

    features = np.column_stack([
        hc, lc, hv, lv, spread_ratio, sc, np.zeros(len(df)), hh_hl_ratio, lh_ll_ratio, total_vol
    ]).astype(np.float32)

    return pad_or_truncate(features, target_len)


class DirectDataset(Dataset):
    """Dataset that computes features on-the-fly from global parquet data."""

    def __init__(self, samples: list, item_id_to_idx: dict, config: DataConfig):
        self.samples = samples  # List of (item_id, timestamp) tuples
        self.item_id_to_idx = item_id_to_idx
        self.config = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item_id, timestamp = self.samples[idx]

        try:
            # Recent 5-min sequence
            start = timestamp - pd.Timedelta(hours=self.config.recent_hours)
            item_data = _df_5min.loc[item_id]
            df = item_data[(item_data.index > start) & (item_data.index <= timestamp)]
            recent = prepare_5min_features(df, self.config.recent_len)

            # Medium 1-hour sequence
            start = timestamp - pd.Timedelta(days=self.config.medium_days)
            hour_ts = timestamp.floor('h')
            item_data_1h = _df_1h.loc[item_id]
            df = item_data_1h[(item_data_1h.index > start) & (item_data_1h.index <= hour_ts)]
            medium = prepare_aggregated_features(df, self.config.medium_len)

            # Long 4-hour sequence
            start = timestamp - pd.Timedelta(days=self.config.long_days)
            hour = timestamp.hour
            block_hour = hour - (hour % 4)
            block_ts = timestamp.replace(hour=block_hour, minute=0, second=0, microsecond=0)
            item_data_4h = _df_4h.loc[item_id]
            df = item_data_4h[(item_data_4h.index > start) & (item_data_4h.index <= block_ts)]
            long_seq = prepare_aggregated_features(df, self.config.long_len)

            # Current prices
            try:
                row = _df_5min.loc[(item_id, timestamp)]
                current_high = float(row["avg_high_price"]) if pd.notna(row["avg_high_price"]) else 0.0
                current_low = float(row["avg_low_price"]) if pd.notna(row["avg_low_price"]) else 0.0
            except (KeyError, TypeError):
                current_high, current_low = 0.0, 0.0

            # Targets - normalize by current price to get relative changes
            base_price = max(current_high, current_low, 1.0)  # Avoid division by zero

            targets_min_low = []
            targets_max_high = []
            for horizon in self.config.horizons:
                end_time = timestamp + pd.Timedelta(hours=horizon)
                window = item_data[(item_data.index > timestamp) & (item_data.index <= end_time)]

                if len(window) > 0:
                    min_low = window["avg_low_price"].min()
                    max_high = window["avg_high_price"].max()
                    # Normalize by base price
                    min_low_norm = (min_low / base_price) if pd.notna(min_low) else 1.0
                    max_high_norm = (max_high / base_price) if pd.notna(max_high) else 1.0
                    targets_min_low.append(float(min_low_norm) if np.isfinite(min_low_norm) else 1.0)
                    targets_max_high.append(float(max_high_norm) if np.isfinite(max_high_norm) else 1.0)
                else:
                    targets_min_low.append(1.0)  # No change
                    targets_max_high.append(1.0)

            # Sanitize features - replace NaN/inf with 0
            recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
            medium = np.nan_to_num(medium, nan=0.0, posinf=0.0, neginf=0.0)
            long_seq = np.nan_to_num(long_seq, nan=0.0, posinf=0.0, neginf=0.0)

            return {
                'item_idx': self.item_id_to_idx[item_id],
                'recent': torch.from_numpy(recent).float(),
                'medium': torch.from_numpy(medium).float(),
                'long': torch.from_numpy(long_seq).float(),
                'current_high': torch.tensor(max(current_high, 1.0), dtype=torch.float32),
                'current_low': torch.tensor(max(current_low, 1.0), dtype=torch.float32),
                'targets_min_low': torch.tensor(targets_min_low, dtype=torch.float32),
                'targets_max_high': torch.tensor(targets_max_high, dtype=torch.float32),
            }
        except Exception as e:
            # Return zeros on error
            return {
                'item_idx': 0,
                'recent': torch.zeros(self.config.recent_len, 6),
                'medium': torch.zeros(self.config.medium_len, 10),
                'long': torch.zeros(self.config.long_len, 10),
                'current_high': torch.tensor(0.0),
                'current_low': torch.tensor(0.0),
                'targets_min_low': torch.zeros(len(self.config.horizons)),
                'targets_max_high': torch.zeros(len(self.config.horizons)),
            }


def build_sample_index(
    item_ids: list[int],
    start_date: datetime,
    end_date: datetime,
    config: DataConfig,
    sample_interval_hours: int = 1
) -> list[tuple[int, pd.Timestamp]]:
    """Build index of valid (item_id, timestamp) sample points."""
    samples = []

    min_date = pd.Timestamp(to_utc(start_date + timedelta(days=config.long_days)))
    max_date = pd.Timestamp(to_utc(end_date - timedelta(hours=max(config.horizons))))

    for item_id in item_ids:
        if item_id not in _df_5min.index.get_level_values(0):
            continue

        item_data = _df_5min.loc[item_id]
        timestamps = item_data.index

        valid_ts = timestamps[(timestamps >= min_date) & (timestamps < max_date)]

        sample_every = sample_interval_hours * 12
        for i, ts in enumerate(valid_ts):
            if i % sample_every == 0:
                samples.append((item_id, ts))

    return samples


class PatchTSTEncoder(nn.Module):
    """Simplified PatchTST encoder."""

    def __init__(self, seq_len: int, n_features: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4, patch_size: int = 16):
        super().__init__()

        self.patch_size = patch_size
        n_patches = seq_len // patch_size
        patch_dim = patch_size * n_features

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, features)
        B, S, F = x.shape

        # Reshape to patches
        n_patches = S // self.patch_size
        x = x[:, :n_patches * self.patch_size, :]  # Trim to fit patches
        x = x.reshape(B, n_patches, self.patch_size * F)

        # Embed patches
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)

        # Mean pool
        return x.mean(dim=1)


class PatchTSTModel(nn.Module):
    """Multi-resolution PatchTST for price quantile prediction."""

    def __init__(self, n_items: int, config: DataConfig, d_model: int = 256):
        super().__init__()

        self.recent_encoder = PatchTSTEncoder(config.recent_len, 6, d_model, patch_size=12)
        self.medium_encoder = PatchTSTEncoder(config.medium_len, 10, d_model, patch_size=12)
        self.long_encoder = PatchTSTEncoder(config.long_len, 10, d_model, patch_size=12)

        self.item_embed = nn.Embedding(n_items, d_model // 4)

        fusion_dim = d_model * 3 + d_model // 4 + 2  # encoders + item + prices

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Output heads for each horizon
        n_horizons = len(config.horizons)
        self.min_low_head = nn.Linear(d_model, n_horizons)
        self.max_high_head = nn.Linear(d_model, n_horizons)

    def forward(self, recent, medium, long_seq, item_idx, current_high, current_low):
        # Encode each resolution
        recent_enc = self.recent_encoder(recent)
        medium_enc = self.medium_encoder(medium)
        long_enc = self.long_encoder(long_seq)

        # Item embedding
        item_enc = self.item_embed(item_idx)

        # Fuse all
        prices = torch.stack([current_high, current_low], dim=-1)
        fused = torch.cat([recent_enc, medium_enc, long_enc, item_enc, prices], dim=-1)
        fused = self.fusion(fused)

        # Predict
        pred_min_low = self.min_low_head(fused)
        pred_max_high = self.max_high_head(fused)

        return pred_min_low, pred_max_high


def train_epoch(model, loader, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        recent = batch['recent'].to(device)
        medium = batch['medium'].to(device)
        long_seq = batch['long'].to(device)
        item_idx = batch['item_idx'].to(device)
        current_high = batch['current_high'].to(device)
        current_low = batch['current_low'].to(device)
        targets_min_low = batch['targets_min_low'].to(device)
        targets_max_high = batch['targets_max_high'].to(device)

        with autocast():
            pred_min_low, pred_max_high = model(
                recent, medium, long_seq, item_idx, current_high, current_low
            )

            # Huber loss
            loss_min = nn.functional.huber_loss(pred_min_low, targets_min_low)
            loss_max = nn.functional.huber_loss(pred_max_high, targets_max_high)
            loss = loss_min + loss_max

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def do_validation(model, loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            recent = batch['recent'].to(device)
            medium = batch['medium'].to(device)
            long_seq = batch['long'].to(device)
            item_idx = batch['item_idx'].to(device)
            current_high = batch['current_high'].to(device)
            current_low = batch['current_low'].to(device)
            targets_min_low = batch['targets_min_low'].to(device)
            targets_max_high = batch['targets_max_high'].to(device)

            with autocast():
                pred_min_low, pred_max_high = model(
                    recent, medium, long_seq, item_idx, current_high, current_low
                )

                loss_min = nn.functional.huber_loss(pred_min_low, targets_min_low)
                loss_max = nn.functional.huber_loss(pred_max_high, targets_max_high)
                loss = loss_min + loss_max

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/dev/shm/gept')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--train-end', type=str, default='2026-01-12')
    parser.add_argument('--val-end', type=str, default='2026-01-20')
    parser.add_argument('--test-samples', type=int, default=0,
                        help='Limit train samples for quick test (0=full)')
    parser.add_argument('--test-items', type=int, default=0,
                        help='Limit items for quick test (0=all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data globally
    data_dir = Path(args.data_dir)
    init_global_data(data_dir)

    # Load items
    items_df = pd.read_parquet(data_dir / "training_items.parquet")
    item_ids = items_df['item_id'].tolist()
    if args.test_items > 0:
        item_ids = item_ids[:args.test_items]
        logger.info(f"TEST MODE: Limited to {len(item_ids)} items")
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    n_items = len(item_ids)
    logger.info(f"Loaded {n_items} items")

    config = DataConfig()

    # Build sample indices
    data_start = _df_5min.index.get_level_values('timestamp').min()
    train_end_dt = datetime.fromisoformat(args.train_end).replace(tzinfo=timezone.utc)
    val_end_dt = datetime.fromisoformat(args.val_end).replace(tzinfo=timezone.utc)

    logger.info("Building train sample index...")
    train_samples = build_sample_index(item_ids, data_start, train_end_dt, config)
    if args.test_samples > 0:
        train_samples = train_samples[:args.test_samples]
        logger.info(f"TEST MODE: Limited to {len(train_samples):,} train samples")
    else:
        logger.info(f"Train samples: {len(train_samples):,}")

    logger.info("Building val sample index...")
    val_samples = build_sample_index(item_ids, train_end_dt - timedelta(days=config.long_days), val_end_dt, config)
    if args.test_samples > 0:
        val_samples = val_samples[:min(args.test_samples // 10, len(val_samples))]
        logger.info(f"TEST MODE: Limited to {len(val_samples):,} val samples")
    else:
        logger.info(f"Val samples: {len(val_samples):,}")

    # Create datasets
    train_dataset = DirectDataset(train_samples, item_id_to_idx, config)
    val_dataset = DirectDataset(val_samples, item_id_to_idx, config)

    # Create dataloaders with many workers
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers // 2, pin_memory=True, persistent_workers=True
    )

    # Create model
    model = PatchTSTModel(n_items, config, d_model=args.d_model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = do_validation(model, val_loader, device)

        scheduler.step()

        epoch_time = time.time() - start_time
        samples_per_sec = len(train_samples) / epoch_time

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Time: {epoch_time:.0f}s | {samples_per_sec:.0f} samples/s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), data_dir / "best_model.pt")
            logger.info(f"  Saved best model (val_loss={val_loss:.4f})")

    logger.info(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
