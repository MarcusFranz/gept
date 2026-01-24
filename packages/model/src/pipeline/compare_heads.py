"""
Compare Base Model vs Per-Item Head Predictions
================================================

Visualize predictions from both models side-by-side.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Dict, Optional
import argparse
import logging

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig
from src.pipeline.per_item_head import (
    PerItemHead,
    PatchTSTWithItemHead,
    extract_item_samples
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ITEM_INFO = {
    0: ("Cannonball", 180),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_models(base_path: str, head_path: str, device: torch.device):
    """Load base model and per-item head."""
    # Base model
    config = ModelConfig(n_items=1000)
    base_model = PatchTSTModel(config)
    state_dict = torch.load(base_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.to(device)

    # Per-item head
    item_head = PerItemHead(
        d_model=config.d_model,
        hidden_dim=64,
        n_horizons=7,
        n_quantiles=5,
        dropout=0.1
    )
    head_state = torch.load(head_path, map_location=device, weights_only=True)
    item_head.load_state_dict(head_state)
    item_head.to(device)

    # Combined model
    combined = PatchTSTWithItemHead(base_model, item_head, freeze_base=True)

    return base_model, combined


def run_inference(model, recent, medium, long, item_id, device):
    """Run inference on a single sample."""
    recent_t = torch.tensor(recent[np.newaxis], dtype=torch.float32, device=device)
    medium_t = torch.tensor(medium[np.newaxis], dtype=torch.float32, device=device)
    long_t = torch.tensor(long[np.newaxis], dtype=torch.float32, device=device)
    item_ids_t = torch.tensor([item_id], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(recent_t, medium_t, long_t, item_ids_t)

    return (
        outputs['high_quantiles'].cpu().numpy()[0],
        outputs['low_quantiles'].cpu().numpy()[0]
    )


def visualize_comparison(
    recent: np.ndarray,
    base_high: np.ndarray,
    base_low: np.ndarray,
    head_high: np.ndarray,
    head_low: np.ndarray,
    targets: np.ndarray,
    item_idx: int,
    sample_idx: int,
    save_path: Optional[str] = None
):
    """Create side-by-side comparison visualization."""
    item_name, typical_price = ITEM_INFO.get(item_idx, (f"Item {item_idx}", 1000))
    current_gp = typical_price

    hist_high = recent[:, 0] * current_gp
    hist_low = recent[:, 1] * current_gp
    hist_mid = (hist_high + hist_low) / 2
    hist_time = np.linspace(-24, 0, 288)
    horizons = np.array([1, 2, 4, 8, 12, 24, 48])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, high_q, low_q, title in [
        (axes[0], base_high, base_low, "Base Model (Shared Head)"),
        (axes[1], head_high, head_low, "Per-Item Head (Cannonball-Specific)")
    ]:
        # Historical
        ax.fill_between(hist_time, hist_low, hist_high, alpha=0.2, color='#404040',
                        label='Historical Range')
        ax.plot(hist_time, hist_mid, color='#606060', linewidth=1, alpha=0.7)

        # Current price
        ax.scatter([0], [current_gp], color='black', s=150, zorder=10, marker='D',
                   edgecolors='white', linewidths=2)

        # Predictions
        pred_high_p50 = (1.0 + high_q[:, 2]) * current_gp
        pred_low_p50 = (1.0 + low_q[:, 2]) * current_gp
        pred_high_p90 = (1.0 + high_q[:, 4]) * current_gp
        pred_low_p10 = (1.0 + low_q[:, 0]) * current_gp

        ext_horizons = np.concatenate([[0], horizons])
        smooth_h = np.linspace(0, 48, 100)

        def smooth(arr):
            ext = np.concatenate([[current_gp], arr])
            return interp1d(ext_horizons, ext, kind='quadratic')(smooth_h)

        high_p50_s, low_p50_s = smooth(pred_high_p50), smooth(pred_low_p50)
        high_p90_s, low_p10_s = smooth(pred_high_p90), smooth(pred_low_p10)

        ax.fill_between(smooth_h, low_p10_s, high_p90_s, alpha=0.15, color='#2196F3',
                        label='Predicted p10-p90')
        ax.fill_between(smooth_h, low_p50_s, high_p50_s, alpha=0.35, color='#2196F3',
                        label='Predicted Median')
        ax.plot(smooth_h, high_p50_s, '#1976D2', linewidth=2)
        ax.plot(smooth_h, low_p50_s, '#1976D2', linewidth=2)

        # Actuals
        actual_high = (1.0 + targets[:, 0]) * current_gp
        actual_low = (1.0 + targets[:, 1]) * current_gp
        ax.plot(horizons, actual_high, '#E53935', linewidth=2.5, marker='s', markersize=8,
                label='Actual Max High')
        ax.plot(horizons, actual_low, '#E53935', linewidth=2.5, marker='s', markersize=8,
                label='Actual Min Low')
        ax.fill_between(horizons, actual_low, actual_high, alpha=0.1, color='#E53935')

        ax.axhline(y=current_gp, color='black', linestyle=':', alpha=0.4)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Price (GP)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-25, 50)

    plt.suptitle(f'{item_name} - Base Model vs Per-Item Head (Sample {sample_idx})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved to {save_path}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--item-idx', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load models
    base_model, combined_model = load_models(args.base_model, args.head, device)

    # Load data
    data = np.load(args.data)
    item_data = extract_item_samples(dict(data), args.item_idx)

    sample_idx = args.sample
    recent = item_data['recent'][sample_idx]
    medium = item_data['medium'][sample_idx]
    long = item_data['long'][sample_idx]
    item_id = item_data['item_ids'][sample_idx]
    targets = item_data['targets'][sample_idx]

    logger.info(f"Sample {sample_idx}: item_idx={item_id}")

    # Run inference with both models
    base_high, base_low = run_inference(base_model, recent, medium, long, item_id, device)
    head_high, head_low = run_inference(combined_model, recent, medium, long, item_id, device)

    # Visualize
    visualize_comparison(
        recent, base_high, base_low, head_high, head_low,
        targets, args.item_idx, sample_idx, args.save
    )


if __name__ == '__main__':
    main()
