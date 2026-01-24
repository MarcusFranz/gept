"""
PatchTST Prediction Visualizer - GP Values
==========================================

Visualize model predictions with actual GP price values.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from typing import Optional, Tuple
import json
import logging

from src.pipeline.model import PatchTSTModel
from src.pipeline.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OSRS Item database (common items)
OSRS_ITEMS = {
    2: ("Cannonball", 180),
    6: ("Cannon base", 195000),
    30: ("Bronze pickaxe", 220),
    36: ("Steel dagger", 40),
    52: ("Steel axe", 85),
    93: ("Steel platebody", 700),
    # High value flip items
    11232: ("Dragon dart tip", 8500),
    11834: ("Bandos tassets", 28000000),
    11832: ("Bandos chestplate", 16000000),
    4151: ("Abyssal whip", 2800000),
    6585: ("Amulet of fury", 3200000),
    560: ("Death rune", 180),
    561: ("Nature rune", 155),
    565: ("Blood rune", 380),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_path, n_items=1000):
    device = get_device()
    config = ModelConfig(n_items=n_items)
    model = PatchTSTModel(config)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def run_inference(model, recent, medium, long, item_id, device):
    recent_t = torch.tensor(recent[np.newaxis], dtype=torch.float32, device=device)
    medium_t = torch.tensor(medium[np.newaxis], dtype=torch.float32, device=device)
    long_t = torch.tensor(long[np.newaxis], dtype=torch.float32, device=device)
    item_ids_t = torch.tensor([item_id], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(recent_t, medium_t, long_t, item_ids_t)
    return outputs['high_quantiles'].cpu().numpy()[0], outputs['low_quantiles'].cpu().numpy()[0]


def visualize_gp(recent_features, high_quantiles, low_quantiles, targets,
                 osrs_item_id, sample_idx=0, save_path=None):
    """Visualize prediction in GP values."""
    item_name, typical_price = OSRS_ITEMS.get(osrs_item_id, (f"Item {osrs_item_id}", 1000))
    current_gp = typical_price

    hist_high = recent_features[:, 0] * current_gp
    hist_low = recent_features[:, 1] * current_gp
    hist_mid = (hist_high + hist_low) / 2
    hist_time = np.linspace(-24, 0, 288)
    horizons = np.array([1, 2, 4, 8, 12, 24, 48])

    fig, ax = plt.subplots(figsize=(16, 9))

    # Historical
    ax.fill_between(hist_time, hist_low, hist_high, alpha=0.2, color='#404040',
                    label='Historical High-Low Range')
    ax.plot(hist_time, hist_mid, color='#606060', linewidth=1.2, alpha=0.8,
            label='Historical Mid Price')

    # Current price
    ax.scatter([0], [current_gp], color='black', s=200, zorder=10, marker='D',
               edgecolors='white', linewidths=2)
    ax.annotate(f'{current_gp:,.0f} GP', xy=(0, current_gp), xytext=(2, current_gp * 1.05),
                fontsize=11, fontweight='bold')

    # Predictions
    pred_high_p50 = (1.0 + high_quantiles[:, 2]) * current_gp
    pred_low_p50 = (1.0 + low_quantiles[:, 2]) * current_gp
    pred_high_p90 = (1.0 + high_quantiles[:, 4]) * current_gp
    pred_low_p10 = (1.0 + low_quantiles[:, 0]) * current_gp

    ext_horizons = np.concatenate([[0], horizons])
    smooth_h = np.linspace(0, 48, 100)

    def smooth(arr):
        ext = np.concatenate([[current_gp], arr])
        return interp1d(ext_horizons, ext, kind='quadratic')(smooth_h)

    high_p50_s, low_p50_s = smooth(pred_high_p50), smooth(pred_low_p50)
    high_p90_s, low_p10_s = smooth(pred_high_p90), smooth(pred_low_p10)

    ax.fill_between(smooth_h, low_p10_s, high_p90_s, alpha=0.15, color='#2196F3',
                    label='Predicted p10-p90 Range')
    ax.fill_between(smooth_h, low_p50_s, high_p50_s, alpha=0.35, color='#2196F3',
                    label='Predicted Median Channel')
    ax.plot(smooth_h, high_p50_s, '#1976D2', linewidth=2.5, alpha=0.9)
    ax.plot(smooth_h, low_p50_s, '#1976D2', linewidth=2.5, alpha=0.9)
    ax.scatter(horizons, pred_high_p50, color='#1976D2', s=80, zorder=6, marker='^')
    ax.scatter(horizons, pred_low_p50, color='#1976D2', s=80, zorder=6, marker='v')

    # Actuals
    actual_high = (1.0 + targets[:, 0]) * current_gp
    actual_low = (1.0 + targets[:, 1]) * current_gp
    ax.plot(horizons, actual_high, '#E53935', linewidth=2.5, marker='s', markersize=10,
            label='Actual Max High', alpha=0.9)
    ax.plot(horizons, actual_low, '#E53935', linewidth=2.5, marker='s', markersize=10,
            label='Actual Min Low', alpha=0.9)
    ax.fill_between(horizons, actual_low, actual_high, alpha=0.1, color='#E53935')

    ax.axhline(y=current_gp, color='black', linestyle=':', alpha=0.4)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.6, linewidth=2)
    ax.set_xlabel('Time (hours from prediction)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (GP)', fontsize=13, fontweight='bold')
    ax.set_title(f'{item_name} - PatchTST Prediction vs Actual (Sample {sample_idx})',
                 fontsize=15, fontweight='bold', pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-25, 50)

    info = f"Current: {current_gp:,.0f} GP | 1h: {pred_high_p50[0]:,.0f}-{pred_low_p50[0]:,.0f}"
    ax.text(0.98, 0.02, info, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='right', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved to {save_path}")
    plt.show()
    return fig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--save', default=None)
    args = parser.parse_args()

    model, device = load_model(args.model)
    with open(args.mapping) as f:
        idx_to_osrs = {int(k): int(v) for k, v in json.load(f).items()}

    data = np.load(args.data)
    recent = data['recent'][args.sample]
    medium = data['medium'][args.sample]
    long = data['long'][args.sample]
    item_idx = data['item_ids'][args.sample]
    targets = data['targets'][args.sample]

    osrs_id = idx_to_osrs.get(item_idx, item_idx)
    logger.info(f"Sample {args.sample}: idx={item_idx}, osrs_id={osrs_id}")

    high_q, low_q = run_inference(model, recent, medium, long, item_idx, device)
    visualize_gp(recent, high_q, low_q, targets, osrs_id, args.sample, args.save)


if __name__ == '__main__':
    main()
