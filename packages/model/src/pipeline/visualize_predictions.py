"""
PatchTST Prediction Visualizer
==============================

Visualize model predictions vs actual price movements.
Shows historical prices, predicted quantile ranges, and actual outcomes.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import argparse
import logging

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_path: str, n_items: int = 1000) -> Tuple[PatchTSTModel, torch.device]:
    """Load the trained model."""
    device = get_device()
    config = ModelConfig(n_items=n_items)
    model = PatchTSTModel(config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.set_trace = lambda x: None  # Dummy to avoid hook issue
    return model, device


def run_inference(
    model: PatchTSTModel,
    recent: np.ndarray,
    medium: np.ndarray,
    long: np.ndarray,
    item_id: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a single sample."""
    recent_t = torch.tensor(recent[np.newaxis], dtype=torch.float32, device=device)
    medium_t = torch.tensor(medium[np.newaxis], dtype=torch.float32, device=device)
    long_t = torch.tensor(long[np.newaxis], dtype=torch.float32, device=device)
    item_ids_t = torch.tensor([item_id], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(recent_t, medium_t, long_t, item_ids_t)

    high_q = outputs['high_quantiles'].cpu().numpy()[0]
    low_q = outputs['low_quantiles'].cpu().numpy()[0]
    return high_q, low_q


def visualize_combined(
    recent_features: np.ndarray,
    high_quantiles: np.ndarray,
    low_quantiles: np.ndarray,
    targets: np.ndarray,
    sample_idx: int = 0,
    save_path: Optional[str] = None
):
    """
    Create a combined visualization showing price channel predictions.
    Shows high and low predictions together as a price envelope.
    """
    from scipy.interpolate import interp1d

    HIGH_PRICE_IDX = 0
    LOW_PRICE_IDX = 1

    hist_high = recent_features[:, HIGH_PRICE_IDX]
    hist_low = recent_features[:, LOW_PRICE_IDX]
    hist_mid = (hist_high + hist_low) / 2

    hist_time = np.linspace(-24, 0, 288)
    horizons = np.array([1, 2, 4, 8, 12, 24, 48])

    fig, ax = plt.subplots(figsize=(16, 8))

    # Historical price channel
    ax.fill_between(hist_time, hist_low, hist_high, alpha=0.2, color='gray',
                    label='Historical Price Range')
    ax.plot(hist_time, hist_mid, 'k-', alpha=0.6, linewidth=1, label='Historical Mid')

    # Current price
    ax.scatter([0], [1.0], color='black', s=150, zorder=10, marker='D',
               label='Current Price')

    # Predicted envelope (using median predictions)
    pred_high_p50 = 1.0 + high_quantiles[:, 2]
    pred_low_p50 = 1.0 + low_quantiles[:, 2]

    # Full prediction range (p10 low to p90 high)
    pred_high_p90 = 1.0 + high_quantiles[:, 4]
    pred_low_p10 = 1.0 + low_quantiles[:, 0]

    # Extend horizons to include current time
    ext_horizons = np.concatenate([[0], horizons])
    ext_high_p50 = np.concatenate([[1.0], pred_high_p50])
    ext_low_p50 = np.concatenate([[1.0], pred_low_p50])
    ext_high_p90 = np.concatenate([[1.0], pred_high_p90])
    ext_low_p10 = np.concatenate([[1.0], pred_low_p10])

    # Smooth interpolation
    smooth_horizons = np.linspace(0, 48, 100)
    interp_kind = 'quadratic'

    high_p50_smooth = interp1d(ext_horizons, ext_high_p50, kind=interp_kind)(smooth_horizons)
    low_p50_smooth = interp1d(ext_horizons, ext_low_p50, kind=interp_kind)(smooth_horizons)
    high_p90_smooth = interp1d(ext_horizons, ext_high_p90, kind=interp_kind)(smooth_horizons)
    low_p10_smooth = interp1d(ext_horizons, ext_low_p10, kind=interp_kind)(smooth_horizons)

    # Plot prediction envelope
    ax.fill_between(smooth_horizons, low_p10_smooth, high_p90_smooth,
                    alpha=0.15, color='blue', label='Predicted Range (p10-p90)')
    ax.fill_between(smooth_horizons, low_p50_smooth, high_p50_smooth,
                    alpha=0.3, color='blue', label='Predicted Median Channel')

    ax.plot(smooth_horizons, high_p50_smooth, 'b-', linewidth=2, alpha=0.8)
    ax.plot(smooth_horizons, low_p50_smooth, 'b-', linewidth=2, alpha=0.8)

    # Plot prediction points
    ax.scatter(horizons, pred_high_p50, color='blue', s=60, zorder=5, marker='^')
    ax.scatter(horizons, pred_low_p50, color='blue', s=60, zorder=5, marker='v')

    # Actual outcomes
    actual_high = 1.0 + targets[:, 0]
    actual_low = 1.0 + targets[:, 1]

    ax.plot(horizons, actual_high, 'r-', linewidth=2, marker='s', markersize=8,
            label='Actual Max High', alpha=0.8)
    ax.plot(horizons, actual_low, 'r-', linewidth=2, marker='s', markersize=8,
            label='Actual Min Low', alpha=0.8)

    # Fill between actuals
    ax.fill_between(horizons, actual_low, actual_high, alpha=0.1, color='red')

    # Reference lines
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)

    # Labels and formatting
    ax.set_xlabel('Time (hours from prediction point)', fontsize=12)
    ax.set_ylabel('Price (relative to current = 1.0)', fontsize=12)
    ax.set_title(f'PatchTST Price Prediction vs Actual (Sample {sample_idx})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-24, 50)

    # Add percentage annotations on right y-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    yticks = ax.get_yticks()
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{(y-1)*100:+.0f}%' for y in yticks])
    ax2.set_ylabel('Change from Current (%)', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    plt.show()
    return fig


def visualize_grid(
    data: dict,
    model: PatchTSTModel,
    device: torch.device,
    samples: list,
    save_path: Optional[str] = None
):
    """
    Create a grid of predictions for multiple samples.
    """
    from scipy.interpolate import interp1d

    n_samples = len(samples)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    horizons = np.array([1, 2, 4, 8, 12, 24, 48])

    for idx, sample_idx in enumerate(samples[:4]):
        ax = axes[idx]

        recent = data['recent'][sample_idx]
        medium = data['medium'][sample_idx]
        long = data['long'][sample_idx]
        item_id = data['item_ids'][sample_idx]
        targets = data['targets'][sample_idx]

        # Run inference
        high_q, low_q = run_inference(model, recent, medium, long, item_id, device)

        # Historical data
        hist_high = recent[:, 0]
        hist_low = recent[:, 1]
        hist_mid = (hist_high + hist_low) / 2
        hist_time = np.linspace(-24, 0, 288)

        # Plot historical
        ax.fill_between(hist_time, hist_low, hist_high, alpha=0.2, color='gray')
        ax.plot(hist_time, hist_mid, 'k-', alpha=0.5, linewidth=0.8)

        # Current price
        ax.scatter([0], [1.0], color='black', s=100, zorder=10, marker='D')

        # Predictions
        pred_high_p50 = 1.0 + high_q[:, 2]
        pred_low_p50 = 1.0 + low_q[:, 2]
        pred_high_p90 = 1.0 + high_q[:, 4]
        pred_low_p10 = 1.0 + low_q[:, 0]

        # Smooth interpolation
        ext_horizons = np.concatenate([[0], horizons])
        smooth_h = np.linspace(0, 48, 50)

        high_p50_s = interp1d(ext_horizons, np.concatenate([[1.0], pred_high_p50]), kind='quadratic')(smooth_h)
        low_p50_s = interp1d(ext_horizons, np.concatenate([[1.0], pred_low_p50]), kind='quadratic')(smooth_h)
        high_p90_s = interp1d(ext_horizons, np.concatenate([[1.0], pred_high_p90]), kind='quadratic')(smooth_h)
        low_p10_s = interp1d(ext_horizons, np.concatenate([[1.0], pred_low_p10]), kind='quadratic')(smooth_h)

        ax.fill_between(smooth_h, low_p10_s, high_p90_s, alpha=0.15, color='blue')
        ax.fill_between(smooth_h, low_p50_s, high_p50_s, alpha=0.3, color='blue')
        ax.plot(smooth_h, high_p50_s, 'b-', linewidth=1.5)
        ax.plot(smooth_h, low_p50_s, 'b-', linewidth=1.5)

        # Actuals
        actual_high = 1.0 + targets[:, 0]
        actual_low = 1.0 + targets[:, 1]
        ax.plot(horizons, actual_high, 'r--', linewidth=2, marker='s', markersize=5)
        ax.plot(horizons, actual_low, 'r--', linewidth=2, marker='s', markersize=5)

        ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'Sample {sample_idx} (Item {item_id})', fontsize=11)
        ax.set_xlim(-24, 50)
        ax.grid(True, alpha=0.3)

        if idx >= 2:
            ax.set_xlabel('Hours from prediction')
        if idx % 2 == 0:
            ax.set_ylabel('Price (rel. to current)')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='gray', alpha=0.2, label='Historical Range'),
        Patch(facecolor='blue', alpha=0.3, label='Predicted Channel'),
        Line2D([0], [0], color='r', linestyle='--', marker='s', label='Actual'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11)

    plt.suptitle('PatchTST Predictions vs Actuals', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {save_path}")

    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize PatchTST predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to feature chunk (.npz)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--n_items', type=int, default=1000, help='Number of items')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure')
    parser.add_argument('--grid', action='store_true', help='Show 4-sample grid')
    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}")
    model, device = load_model(args.model, args.n_items)

    # Load data
    logger.info(f"Loading data from {args.data}")
    data = np.load(args.data)

    if args.grid:
        # Show grid of 4 random samples
        np.random.seed(42)
        samples = np.random.choice(len(data['recent']), 4, replace=False).tolist()
        visualize_grid(data, model, device, samples, args.save)
    else:
        sample_idx = args.sample
        recent = data['recent'][sample_idx]
        medium = data['medium'][sample_idx]
        long = data['long'][sample_idx]
        item_id = data['item_ids'][sample_idx]
        targets = data['targets'][sample_idx]

        logger.info(f"Sample {sample_idx}: item_id={item_id}")

        high_q, low_q = run_inference(model, recent, medium, long, item_id, device)
        visualize_combined(recent, high_q, low_q, targets, sample_idx, args.save)


if __name__ == '__main__':
    main()
