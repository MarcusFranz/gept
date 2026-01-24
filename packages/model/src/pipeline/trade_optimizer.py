"""
Trade Timing Optimizer
======================

Learns optimal buy/sell timing from price predictions.

Given high/low quantile predictions across 7 horizons, this module
determines the best timeframe to:
1. Place a buy offer (catch the low)
2. Place a sell offer (catch the high)

Architecture:
    Price Predictions (high/low quantiles)
        → Trade Optimizer MLP
        → Buy Window + Sell Window + Expected Margin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Horizons in hours
HORIZONS = [1, 2, 4, 8, 12, 24, 48]
N_HORIZONS = len(HORIZONS)
N_QUANTILES = 5  # p10, p30, p50, p70, p90


class TradeTimingHead(nn.Module):
    """
    Learns optimal buy/sell timing from quantile predictions.

    Input: Concatenated high and low quantile predictions
           Shape: (batch, 7 horizons * 5 quantiles * 2) = (batch, 70)

    Output:
        - buy_logits: (batch, 7) - probability of each horizon being best to buy
        - sell_logits: (batch, 7) - probability of each horizon being best to sell
        - margin_pred: (batch, 1) - predicted profit margin
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        input_dim = N_HORIZONS * N_QUANTILES * 2  # 70

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for buy timing, sell timing, and margin
        self.buy_head = nn.Linear(hidden_dim, N_HORIZONS)
        self.sell_head = nn.Linear(hidden_dim, N_HORIZONS)
        self.margin_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        high_quantiles: torch.Tensor,
        low_quantiles: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            high_quantiles: (batch, 7, 5) - predicted max high per horizon
            low_quantiles: (batch, 7, 5) - predicted min low per horizon

        Returns:
            Dict with buy_probs, sell_probs, margin_pred
        """
        batch_size = high_quantiles.shape[0]

        # Flatten and concatenate
        high_flat = high_quantiles.view(batch_size, -1)  # (batch, 35)
        low_flat = low_quantiles.view(batch_size, -1)    # (batch, 35)
        x = torch.cat([high_flat, low_flat], dim=1)      # (batch, 70)

        # Shared representation
        features = self.shared(x)

        # Predictions
        buy_logits = self.buy_head(features)
        sell_logits = self.sell_head(features)
        margin_pred = self.margin_head(features)

        return {
            'buy_logits': buy_logits,
            'sell_logits': sell_logits,
            'buy_probs': F.softmax(buy_logits, dim=-1),
            'sell_probs': F.softmax(sell_logits, dim=-1),
            'margin_pred': margin_pred.squeeze(-1),
        }


def compute_optimal_timing(
    high_quantiles: np.ndarray,
    low_quantiles: np.ndarray,
    quantile_idx: int = 2,  # p50 by default
) -> Dict[str, any]:
    """
    Compute optimal buy/sell timing using simple heuristics.

    For each combination of buy_horizon and sell_horizon (where sell >= buy),
    compute the expected margin and find the best combination.

    Args:
        high_quantiles: (7, 5) predictions for max high
        low_quantiles: (7, 5) predictions for min low
        quantile_idx: which quantile to use (0=p10, 2=p50, 4=p90)

    Returns:
        Dict with optimal timing recommendations
    """
    n_horizons = len(HORIZONS)

    # Extract the selected quantile
    high_prices = high_quantiles[:, quantile_idx]  # (7,)
    low_prices = low_quantiles[:, quantile_idx]    # (7,)

    best_margin = -np.inf
    best_buy_idx = 0
    best_sell_idx = 0

    margins_matrix = np.zeros((n_horizons, n_horizons))

    for buy_idx in range(n_horizons):
        for sell_idx in range(buy_idx, n_horizons):
            # Buy at the low, sell at the high
            buy_price = 1.0 + low_prices[buy_idx]   # relative to current
            sell_price = 1.0 + high_prices[sell_idx]

            margin = (sell_price - buy_price) / buy_price
            margins_matrix[buy_idx, sell_idx] = margin

            # Time-adjusted margin (penalize longer holds slightly)
            hold_hours = HORIZONS[sell_idx]
            time_factor = 1.0 / np.sqrt(hold_hours)  # sqrt decay
            adjusted_margin = margin * time_factor

            if adjusted_margin > best_margin:
                best_margin = adjusted_margin
                best_buy_idx = buy_idx
                best_sell_idx = sell_idx

    raw_margin = margins_matrix[best_buy_idx, best_sell_idx]

    return {
        'buy_horizon_idx': best_buy_idx,
        'sell_horizon_idx': best_sell_idx,
        'buy_horizon_hours': HORIZONS[best_buy_idx],
        'sell_horizon_hours': HORIZONS[best_sell_idx],
        'expected_margin': raw_margin,
        'time_adjusted_margin': best_margin,
        'margins_matrix': margins_matrix,
        'buy_price_pct': low_prices[best_buy_idx],
        'sell_price_pct': high_prices[best_sell_idx],
    }


def compute_training_labels(targets: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute optimal timing labels from actual price movements.

    Args:
        targets: (N, 7, 2) actual [max_high, min_low] per horizon

    Returns:
        Dict with buy_labels, sell_labels, margin_labels
    """
    n_samples = len(targets)

    buy_labels = np.zeros(n_samples, dtype=np.int64)
    sell_labels = np.zeros(n_samples, dtype=np.int64)
    margin_labels = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        high_actual = targets[i, :, 0]  # (7,) max high per horizon
        low_actual = targets[i, :, 1]   # (7,) min low per horizon

        best_margin = -np.inf

        for buy_idx in range(N_HORIZONS):
            for sell_idx in range(buy_idx, N_HORIZONS):
                buy_price = 1.0 + low_actual[buy_idx]
                sell_price = 1.0 + high_actual[sell_idx]
                margin = (sell_price - buy_price) / buy_price

                # Time adjustment
                hold_hours = HORIZONS[sell_idx]
                adjusted = margin / np.sqrt(hold_hours)

                if adjusted > best_margin:
                    best_margin = adjusted
                    buy_labels[i] = buy_idx
                    sell_labels[i] = sell_idx
                    margin_labels[i] = margin

    return {
        'buy_labels': buy_labels,
        'sell_labels': sell_labels,
        'margin_labels': margin_labels,
    }


class TradeOptimizerLoss(nn.Module):
    """Combined loss for trade timing optimization."""

    def __init__(self, margin_weight: float = 0.5):
        super().__init__()
        self.margin_weight = margin_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        buy_labels: torch.Tensor,
        sell_labels: torch.Tensor,
        margin_labels: torch.Tensor,
    ) -> torch.Tensor:
        buy_loss = self.ce_loss(outputs['buy_logits'], buy_labels)
        sell_loss = self.ce_loss(outputs['sell_logits'], sell_labels)
        margin_loss = self.mse_loss(outputs['margin_pred'], margin_labels)

        return buy_loss + sell_loss + self.margin_weight * margin_loss


def interpret_recommendation(
    high_quantiles: np.ndarray,
    low_quantiles: np.ndarray,
    current_price: float,
    item_name: str = "Item"
) -> str:
    """
    Generate human-readable trade recommendation.
    """
    timing = compute_optimal_timing(high_quantiles, low_quantiles)

    buy_price = current_price * (1.0 + timing['buy_price_pct'])
    sell_price = current_price * (1.0 + timing['sell_price_pct'])
    profit_gp = sell_price - buy_price

    rec = f"""
╔══════════════════════════════════════════════════════════════╗
║  TRADE RECOMMENDATION: {item_name:<37} ║
╠══════════════════════════════════════════════════════════════╣
║  Current Price: {current_price:>10,.0f} GP                              ║
╠══════════════════════════════════════════════════════════════╣
║  BUY WINDOW:  Within {timing['buy_horizon_hours']:>2}h                                   ║
║  → Target buy price: {buy_price:>10,.0f} GP ({timing['buy_price_pct']*100:>+.1f}%)              ║
╠══════════════════════════════════════════════════════════════╣
║  SELL WINDOW: Within {timing['sell_horizon_hours']:>2}h                                   ║
║  → Target sell price: {sell_price:>9,.0f} GP ({timing['sell_price_pct']*100:>+.1f}%)              ║
╠══════════════════════════════════════════════════════════════╣
║  EXPECTED PROFIT: {profit_gp:>8,.0f} GP ({timing['expected_margin']*100:.1f}%)                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    return rec
