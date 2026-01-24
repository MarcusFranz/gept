"""
Memory-loaded PatchTST Training
===============================

Loads all data into RAM before training for maximum GPU utilization.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from glob import glob
from pathlib import Path
import logging
import yaml
from datetime import datetime
import json
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.pipeline.model import PatchTSTModel, combined_quantile_loss
from src.pipeline.config import ModelConfig


def load_all_chunks(chunk_dir, subsample_ratio=1.0):
    """Load all chunks into memory with memory-efficient concatenation.

    Args:
        chunk_dir: Directory containing chunk files
        subsample_ratio: Ratio of data to keep (1.0 = all, 0.5 = half)
    """
    import gc

    chunk_paths = sorted(glob(str(chunk_dir / 'chunk_*.npz')))

    # Optionally subsample chunks to reduce memory
    if subsample_ratio < 1.0:
        n_chunks = max(1, int(len(chunk_paths) * subsample_ratio))
        # Take every Nth chunk to maintain temporal distribution
        step = len(chunk_paths) // n_chunks
        chunk_paths = chunk_paths[::step][:n_chunks]

    logger.info(f'Loading {len(chunk_paths)} chunks from {chunk_dir}...')

    # Load chunks one array type at a time to reduce peak memory
    result = {}

    for array_name in ['recent', 'medium', 'long', 'item_ids', 'targets']:
        logger.info(f'  Loading and concatenating {array_name}...')
        arrays = []
        for i, path in enumerate(chunk_paths):
            if i % 100 == 0:
                logger.info(f'    Chunk {i}/{len(chunk_paths)}')
            data = np.load(path)
            arrays.append(data[array_name])
            del data  # Release file handle

        logger.info(f'    Concatenating {len(arrays)} arrays...')
        result[array_name] = np.concatenate(arrays, axis=0)
        del arrays
        gc.collect()
        logger.info(f'    {array_name} shape: {result[array_name].shape}')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    if device.type == 'cuda':
        logger.info(f'GPU: {torch.cuda.get_device_name()}')

    features_dir = Path(config['output_dir']) / 'features'

    import gc

    logger.info('Loading training data into memory...')
    train_data = load_all_chunks(features_dir / 'train')
    logger.info(f'Train samples: {len(train_data["recent"]):,}')

    # Convert training data to tensors immediately and free numpy arrays
    logger.info('Converting training data to tensors...')
    train_tensors = []
    for key in ['recent', 'medium', 'long', 'item_ids', 'targets']:
        dtype = torch.long if key == 'item_ids' else torch.float32
        logger.info(f'  Converting {key}...')
        train_tensors.append(torch.tensor(train_data[key], dtype=dtype))
        del train_data[key]
        gc.collect()
    del train_data
    gc.collect()

    train_dataset = TensorDataset(*train_tensors)
    del train_tensors
    gc.collect()
    logger.info(f'Train dataset created: {len(train_dataset):,} samples')

    # Subsample validation to reduce memory (val is 2.3x larger than train)
    val_subsample = config.get('training', {}).get('val_subsample', 0.5)
    logger.info(f'Loading validation data into memory (subsample={val_subsample})...')
    val_data = load_all_chunks(features_dir / 'val', subsample_ratio=val_subsample)
    logger.info(f'Val samples: {len(val_data["recent"]):,}')

    # Convert validation data to tensors immediately and free numpy arrays
    logger.info('Converting validation data to tensors...')
    val_tensors = []
    for key in ['recent', 'medium', 'long', 'item_ids', 'targets']:
        dtype = torch.long if key == 'item_ids' else torch.float32
        logger.info(f'  Converting {key}...')
        val_tensors.append(torch.tensor(val_data[key], dtype=dtype))
        del val_data[key]
        gc.collect()
    del val_data
    gc.collect()

    val_dataset = TensorDataset(*val_tensors)
    del val_tensors
    gc.collect()
    logger.info(f'Val dataset created: {len(val_dataset):,} samples')

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f'Train batches: {len(train_loader)}')
    logger.info(f'Val batches: {len(val_loader)}')

    model_config = ModelConfig(
        n_items=config['model']['n_items'],
        item_embed_dim=config['model']['item_embed_dim'],
    )
    model = PatchTSTModel(model_config).to(device)
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    scaler = torch.amp.GradScaler()
    quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)

    n_epochs = config['training']['epochs']
    best_val_loss = float('inf')
    model_dir = Path(config['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    training_log = []

    for epoch in range(n_epochs):
        logger.info(f'\n=== Epoch {epoch+1}/{n_epochs} ===')

        model.train()
        train_loss = 0.0
        for batch_idx, (recent, medium, long, item_ids, targets) in enumerate(train_loader):
            recent = recent.to(device, non_blocking=True)
            medium = medium.to(device, non_blocking=True)
            long = long.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(recent, medium, long, item_ids)
                loss = combined_quantile_loss(outputs, targets, quantiles)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(f'  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}')

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for recent, medium, long, item_ids, targets in val_loader:
                recent = recent.to(device, non_blocking=True)
                medium = medium.to(device, non_blocking=True)
                long = long.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(recent, medium, long, item_ids)
                    loss = combined_quantile_loss(outputs, targets, quantiles)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        logger.info(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')

        training_log.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'timestamp': datetime.utcnow().isoformat()
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_dir / 'best_model.pt')
            logger.info(f'New best model saved!')

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, model_dir / 'checkpoint.pt')
            logger.info('Checkpoint saved')

    torch.save(model.state_dict(), model_dir / 'final_model.pt')
    with open(model_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    logger.info(f'\nTraining complete!')
    logger.info(f'Best val_loss: {best_val_loss:.4f}')
    logger.info(f'Models saved to: {model_dir}')


if __name__ == '__main__':
    main()
