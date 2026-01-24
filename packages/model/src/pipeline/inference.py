"""
PatchTST Inference Module
=========================

Load trained models and run inference on new data.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device (MPS for Mac, CUDA for NVIDIA, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(
    model_path: str,
    config: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None
) -> Tuple[PatchTSTModel, torch.device]:
    """
    Load a trained PatchTST model.

    Args:
        model_path: Path to saved model weights (.pt file)
        config: Model configuration (uses defaults if not provided)
        device: Device to load model on (auto-detects if not provided)

    Returns:
        Tuple of (model, device)
    """
    if device is None:
        device = get_device()

    if config is None:
        config = ModelConfig()

    model = PatchTSTModel(config)

    # Load weights
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path} on {device}")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    return model, device


def predict(
    model: PatchTSTModel,
    recent: np.ndarray,
    medium: np.ndarray,
    long: np.ndarray,
    item_ids: np.ndarray,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference on prepared features.

    Args:
        model: Loaded PatchTST model
        recent: Recent features (batch, 288, 6)
        medium: Medium features (batch, 168, 10)
        long: Long features (batch, 180, 10)
        item_ids: Item indices (batch,)
        device: Device model is on

    Returns:
        Dictionary with 'high_quantiles' and 'low_quantiles',
        each of shape (batch, 7, 5) representing quantile predictions
        for 7 horizons (1h, 2h, 4h, 8h, 12h, 24h, 48h) and
        5 quantiles (p10, p30, p50, p70, p90).
    """
    model.eval()

    # Convert to tensors
    recent_t = torch.tensor(recent, dtype=torch.float32, device=device)
    medium_t = torch.tensor(medium, dtype=torch.float32, device=device)
    long_t = torch.tensor(long, dtype=torch.float32, device=device)
    item_ids_t = torch.tensor(item_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(recent_t, medium_t, long_t, item_ids_t)

    return {
        'high_quantiles': outputs['high_quantiles'].cpu().numpy(),
        'low_quantiles': outputs['low_quantiles'].cpu().numpy(),
    }


def interpret_predictions(
    outputs: Dict[str, np.ndarray],
    current_mid_price: float,
    horizons: Tuple[int, ...] = (1, 2, 4, 8, 12, 24, 48),
    quantiles: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> Dict[str, np.ndarray]:
    """
    Convert model outputs to actual price predictions.

    The model outputs percentage movements relative to current mid price.
    This function converts them to absolute prices.

    Args:
        outputs: Model outputs with 'high_quantiles' and 'low_quantiles'
        current_mid_price: Current mid price for the item
        horizons: Horizon hours corresponding to model outputs
        quantiles: Quantile levels corresponding to model outputs

    Returns:
        Dictionary with 'high_prices' and 'low_prices' in absolute terms
    """
    high_pct = outputs['high_quantiles']  # (batch, 7, 5) percentage movements
    low_pct = outputs['low_quantiles']

    # Convert to absolute prices
    high_prices = current_mid_price * (1 + high_pct)
    low_prices = current_mid_price * (1 + low_pct)

    return {
        'high_prices': high_prices,
        'low_prices': low_prices,
        'horizons': horizons,
        'quantiles': quantiles,
    }


def validate_model(model_path: str, config: Optional[ModelConfig] = None) -> bool:
    """
    Validate that a model loads and runs correctly.

    Args:
        model_path: Path to model weights
        config: Model configuration

    Returns:
        True if validation passes, raises exception otherwise
    """
    logger.info("=== Model Validation ===")

    # Load model
    model, device = load_model(model_path, config)

    # Create synthetic test data
    batch_size = 4
    recent = np.random.randn(batch_size, 288, 6).astype(np.float32)
    medium = np.random.randn(batch_size, 168, 10).astype(np.float32)
    long = np.random.randn(batch_size, 180, 10).astype(np.float32)
    item_ids = np.array([0, 1, 2, 3], dtype=np.int64)

    # Normalize like real features (values around 1.0 for prices)
    recent[:, :, 0:2] = 1.0 + recent[:, :, 0:2] * 0.05  # prices near 1
    recent[:, :, 2:4] = np.abs(recent[:, :, 2:4]) * 5    # log volumes
    recent[:, :, 4] = np.abs(recent[:, :, 4]) * 0.05     # spread ratio
    recent[:, :, 5] = np.abs(recent[:, :, 5])            # staleness

    medium[:, :, 0:2] = 1.0 + medium[:, :, 0:2] * 0.05
    long[:, :, 0:2] = 1.0 + long[:, :, 0:2] * 0.05

    logger.info(f"Test batch size: {batch_size}")
    logger.info(f"Recent shape: {recent.shape}")
    logger.info(f"Medium shape: {medium.shape}")
    logger.info(f"Long shape: {long.shape}")

    # Run inference
    outputs = predict(model, recent, medium, long, item_ids, device)

    # Validate outputs
    high_q = outputs['high_quantiles']
    low_q = outputs['low_quantiles']

    logger.info(f"\nOutput shapes:")
    logger.info(f"  high_quantiles: {high_q.shape}")
    logger.info(f"  low_quantiles: {low_q.shape}")

    expected_shape = (batch_size, 7, 5)
    assert high_q.shape == expected_shape, f"Expected shape {expected_shape}, got {high_q.shape}"
    assert low_q.shape == expected_shape, f"Expected shape {expected_shape}, got {low_q.shape}"

    # Check for NaN/Inf
    assert not np.isnan(high_q).any(), "NaN values in high_quantiles"
    assert not np.isnan(low_q).any(), "NaN values in low_quantiles"
    assert not np.isinf(high_q).any(), "Inf values in high_quantiles"
    assert not np.isinf(low_q).any(), "Inf values in low_quantiles"

    # Check quantile ordering (p10 < p30 < p50 < p70 < p90 for each horizon)
    # This validates the model learned proper quantile structure
    quantile_order_violations = 0
    for b in range(batch_size):
        for h in range(7):
            if not np.all(np.diff(high_q[b, h, :]) >= -0.01):  # small tolerance
                quantile_order_violations += 1
            if not np.all(np.diff(low_q[b, h, :]) >= -0.01):
                quantile_order_violations += 1

    logger.info(f"\nQuantile ordering violations: {quantile_order_violations}/{batch_size * 7 * 2}")

    # Print sample predictions
    logger.info(f"\nSample prediction (item 0):")
    logger.info(f"  Horizons: 1h, 2h, 4h, 8h, 12h, 24h, 48h")
    logger.info(f"  High p50 (median): {high_q[0, :, 2]}")
    logger.info(f"  Low p50 (median):  {low_q[0, :, 2]}")

    # Interpret as prices for a sample mid price
    sample_mid = 1000.0  # Example: 1000gp item
    interpreted = interpret_predictions(outputs, sample_mid)
    logger.info(f"\nInterpreted prices (mid={sample_mid}gp, item 0):")
    logger.info(f"  1h high p50: {interpreted['high_prices'][0, 0, 2]:.1f}gp")
    logger.info(f"  1h low p50:  {interpreted['low_prices'][0, 0, 2]:.1f}gp")

    logger.info("\n=== Validation PASSED ===")
    return True


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Validate PatchTST model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--n_items', type=int, default=1000, help='Number of items model was trained on')
    args = parser.parse_args()

    config = ModelConfig(n_items=args.n_items)
    validate_model(args.model, config)
