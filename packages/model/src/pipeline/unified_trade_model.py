"""
Unified Trade Model
===================

Combines price prediction, volume estimation, and timing optimization
into a single coherent architecture.

Key insight: Volume should INFORM timing, not just validate it.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Input Features                        │
    │  Recent (288×6), Medium (168×6), Long (30×6), Item ID   │
    └───────────────────────────┬─────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │              PatchTST Base (frozen)                      │
    │              → Temporal embeddings                       │
    └───────────────────────────┬─────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Per-Item Head                               │
    │              → Price quantiles (7×5×2)                   │
    │              → Volume predictions (7×2)  ← NEW           │
    └───────────────────────────┬─────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────┐
    │         Joint Timing Optimizer                           │
    │         Input: prices + volumes + user prefs             │
    │         Output: buy/sell timing with expected value      │
    └─────────────────────────────────────────────────────────┘
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

HORIZONS = [1, 2, 4, 8, 12, 24, 48]
N_HORIZONS = len(HORIZONS)
N_QUANTILES = 5


@dataclass
class UserConstraints:
    """User trading constraints."""
    risk_tolerance: float = 0.5      # 0=conservative, 1=aggressive
    max_hold_hours: int = 48         # Maximum time willing to wait
    min_margin_pct: float = 0.02     # Minimum acceptable margin
    capital_gp: int = 10_000_000     # Available capital
    max_position_pct: float = 0.10   # Max % of capital per trade


@dataclass
class TradeRecommendation:
    """Complete trade recommendation with expected value."""
    recommend_trade: bool
    reason: str = ""

    # Timing
    buy_horizon_hours: int = 0
    sell_horizon_hours: int = 0

    # Prices
    buy_price_gp: int = 0
    sell_price_gp: int = 0

    # Volume-aware metrics
    buy_fill_probability: float = 0.0
    sell_fill_probability: float = 0.0
    combined_fill_probability: float = 0.0

    # Position sizing
    recommended_quantity: int = 0
    volume_limited: bool = False

    # Profitability
    margin_pct: float = 0.0
    expected_margin_pct: float = 0.0  # margin × fill_prob
    profit_per_item_gp: int = 0
    expected_total_profit_gp: int = 0

    # Confidence
    price_confidence: float = 0.0
    volume_confidence: float = 0.0
    overall_confidence: float = 0.0


class VolumeHead(nn.Module):
    """
    Predicts expected volume at each horizon.

    Output: (batch, 7, 2) - [buy_volume, sell_volume] per horizon
    Volume is in log scale (always positive after exp).
    """

    def __init__(self, d_model: int = 256, hidden_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Predict log volume for numerical stability
        self.volume_proj = nn.Linear(hidden_dim, N_HORIZONS * 2)

        # Predict volume uncertainty (for confidence)
        self.uncertainty_proj = nn.Linear(hidden_dim, N_HORIZONS * 2)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (batch, d_model) from base model

        Returns:
            Dict with volume predictions and uncertainties
        """
        features = self.mlp(embeddings)

        # Log volume predictions
        log_volume = self.volume_proj(features)
        log_volume = log_volume.view(-1, N_HORIZONS, 2)

        # Uncertainty
        log_uncertainty = self.uncertainty_proj(features)
        log_uncertainty = log_uncertainty.view(-1, N_HORIZONS, 2)

        return {
            'log_volume': log_volume,
            'volume': torch.exp(log_volume),  # Actual volume
            'volume_uncertainty': torch.exp(log_uncertainty),
        }


class UnifiedItemHead(nn.Module):
    """
    Per-item head that predicts BOTH prices AND volumes.

    This ensures price and volume predictions are consistent
    and can be learned jointly.
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 64,
        n_horizons: int = N_HORIZONS,
        n_quantiles: int = N_QUANTILES,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared feature extraction
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Price prediction heads
        self.high_proj = nn.Linear(hidden_dim, n_horizons * n_quantiles)
        self.low_proj = nn.Linear(hidden_dim, n_horizons * n_quantiles)

        # Volume prediction heads
        self.buy_volume_proj = nn.Linear(hidden_dim, n_horizons)
        self.sell_volume_proj = nn.Linear(hidden_dim, n_horizons)

        # Fill probability heads (learned, not heuristic!)
        self.buy_fill_proj = nn.Linear(hidden_dim, n_horizons)
        self.sell_fill_proj = nn.Linear(hidden_dim, n_horizons)

        self.n_horizons = n_horizons
        self.n_quantiles = n_quantiles

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: (batch, d_model) from PatchTST

        Returns:
            Dict with price quantiles, volumes, and fill probabilities
        """
        batch_size = embeddings.shape[0]

        # Shared features
        features = self.shared_mlp(embeddings)

        # Price predictions (percentage change from current)
        high_q = self.high_proj(features).view(batch_size, self.n_horizons, self.n_quantiles)
        low_q = self.low_proj(features).view(batch_size, self.n_horizons, self.n_quantiles)

        # Volume predictions (log scale)
        buy_log_vol = self.buy_volume_proj(features)
        sell_log_vol = self.sell_volume_proj(features)

        # Fill probability predictions (sigmoid for [0,1])
        buy_fill = torch.sigmoid(self.buy_fill_proj(features))
        sell_fill = torch.sigmoid(self.sell_fill_proj(features))

        return {
            # Prices
            'high_quantiles': high_q,
            'low_quantiles': low_q,

            # Volumes
            'buy_log_volume': buy_log_vol,
            'sell_log_volume': sell_log_vol,
            'buy_volume': torch.exp(buy_log_vol),
            'sell_volume': torch.exp(sell_log_vol),

            # Fill probabilities (learned!)
            'buy_fill_prob': buy_fill,
            'sell_fill_prob': sell_fill,
        }


class JointTimingOptimizer(nn.Module):
    """
    Optimizes trade timing using BOTH price AND volume information.

    Key difference from original: considers fill probability in the objective.

    Objective: maximize expected_value = margin × buy_fill × sell_fill
    Subject to: sell_horizon > buy_horizon (sequential constraint)
    """

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        # Input: prices (70) + volumes (14) + fill probs (14) + user prefs (5) = 103
        input_dim = (N_HORIZONS * N_QUANTILES * 2) + (N_HORIZONS * 4) + 5

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.buy_head = nn.Linear(hidden_dim, N_HORIZONS)
        self.sell_head = nn.Linear(hidden_dim, N_HORIZONS)
        self.expected_value_head = nn.Linear(hidden_dim, 1)
        self.skip_head = nn.Linear(hidden_dim, 1)  # Should we skip this trade?

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
        low_quantiles: torch.Tensor,
        buy_volume: torch.Tensor,
        sell_volume: torch.Tensor,
        buy_fill_prob: torch.Tensor,
        sell_fill_prob: torch.Tensor,
        user_constraints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            high_quantiles: (batch, 7, 5) price predictions
            low_quantiles: (batch, 7, 5) price predictions
            buy_volume: (batch, 7) expected buy volume per horizon
            sell_volume: (batch, 7) expected sell volume per horizon
            buy_fill_prob: (batch, 7) fill probability per horizon
            sell_fill_prob: (batch, 7) fill probability per horizon
            user_constraints: (batch, 5) [risk, max_hours_norm, min_margin, capital_norm, max_pos]

        Returns:
            Dict with timing recommendations
        """
        batch_size = high_quantiles.shape[0]

        # Flatten all inputs
        high_flat = high_quantiles.view(batch_size, -1)
        low_flat = low_quantiles.view(batch_size, -1)

        # Normalize volumes (log scale)
        buy_vol_norm = torch.log1p(buy_volume)
        sell_vol_norm = torch.log1p(sell_volume)

        # Concatenate everything
        x = torch.cat([
            high_flat,
            low_flat,
            buy_vol_norm,
            sell_vol_norm,
            buy_fill_prob,
            sell_fill_prob,
            user_constraints,
        ], dim=1)

        # Encode
        features = self.encoder(x)

        # Predictions
        buy_logits = self.buy_head(features)
        sell_logits = self.sell_head(features)
        expected_value = self.expected_value_head(features).squeeze(-1)
        skip_logit = self.skip_head(features).squeeze(-1)

        return {
            'buy_logits': buy_logits,
            'sell_logits': sell_logits,
            'buy_probs': F.softmax(buy_logits, dim=-1),
            'sell_probs': F.softmax(sell_logits, dim=-1),
            'expected_value': expected_value,
            'skip_prob': torch.sigmoid(skip_logit),
        }


def encode_user_constraints(constraints: UserConstraints) -> np.ndarray:
    """Convert UserConstraints to neural network input."""
    return np.array([
        constraints.risk_tolerance,
        constraints.max_hold_hours / 48.0,
        constraints.min_margin_pct / 0.20,
        np.log10(constraints.capital_gp) / 9.0,
        constraints.max_position_pct,
    ], dtype=np.float32)


def compute_joint_labels(
    targets: np.ndarray,
    volumes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute optimal timing labels considering BOTH price AND volume.

    Objective: maximize expected_value = margin × fill_probability

    Args:
        targets: (N, 7, 2) actual [max_high, min_low] per horizon
        volumes: (N, 7, 2) actual [buy_volume, sell_volume] per horizon

    Returns:
        Labels for training
    """
    n_samples = len(targets)

    buy_labels = np.zeros(n_samples, dtype=np.int64)
    sell_labels = np.zeros(n_samples, dtype=np.int64)
    margin_labels = np.zeros(n_samples, dtype=np.float32)
    expected_value_labels = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        high_actual = targets[i, :, 0]
        low_actual = targets[i, :, 1]
        buy_vol = volumes[i, :, 0] if volumes is not None else np.ones(N_HORIZONS)
        sell_vol = volumes[i, :, 1] if volumes is not None else np.ones(N_HORIZONS)

        best_ev = -np.inf

        for buy_idx in range(N_HORIZONS - 1):
            for sell_idx in range(buy_idx + 1, N_HORIZONS):
                buy_price = 1.0 + low_actual[buy_idx]
                sell_price = 1.0 + high_actual[sell_idx]
                margin = (sell_price - buy_price) / buy_price

                # Estimate fill probability from volume
                # Higher volume = higher fill probability
                total_vol = buy_vol[buy_idx] + sell_vol[sell_idx]
                vol_factor = 1 - np.exp(-total_vol / 10000)  # Saturates around 10k volume

                # Time factor
                total_hours = HORIZONS[sell_idx]
                time_factor = 1 - np.exp(-total_hours / 4)

                fill_prob = vol_factor * time_factor
                fill_prob = np.clip(fill_prob, 0.1, 0.95)

                # Expected value
                ev = margin * fill_prob / np.sqrt(total_hours)

                if ev > best_ev:
                    best_ev = ev
                    buy_labels[i] = buy_idx
                    sell_labels[i] = sell_idx
                    margin_labels[i] = margin
                    expected_value_labels[i] = margin * fill_prob

    return {
        'buy_labels': buy_labels,
        'sell_labels': sell_labels,
        'margin_labels': margin_labels,
        'expected_value_labels': expected_value_labels,
    }


def generate_recommendation(
    high_quantiles: np.ndarray,
    low_quantiles: np.ndarray,
    buy_fill_prob: np.ndarray,
    sell_fill_prob: np.ndarray,
    buy_volume: np.ndarray,
    sell_volume: np.ndarray,
    current_price: float,
    constraints: UserConstraints,
) -> TradeRecommendation:
    """
    Generate a complete trade recommendation.

    This is the main inference function that combines all predictions
    with user constraints to produce an actionable recommendation.
    """
    # Map risk tolerance to quantile index
    if constraints.risk_tolerance < 0.33:
        buy_q, sell_q = 1, 3  # p30/p70 - conservative
    elif constraints.risk_tolerance < 0.66:
        buy_q, sell_q = 2, 2  # p50/p50 - moderate
    else:
        buy_q, sell_q = 0, 4  # p10/p90 - aggressive

    # Find max horizon index based on user constraint
    max_h = N_HORIZONS - 1
    for i, h in enumerate(HORIZONS):
        if h >= constraints.max_hold_hours:
            max_h = i
            break

    # Find best trade considering BOTH margin AND fill probability
    best_ev = -np.inf
    best_buy_idx = None
    best_sell_idx = None

    for buy_idx in range(min(max_h, N_HORIZONS - 1)):
        for sell_idx in range(buy_idx + 1, max_h + 1):
            buy_pct = low_quantiles[buy_idx, buy_q]
            sell_pct = high_quantiles[sell_idx, sell_q]

            buy_price = current_price * (1 + buy_pct)
            sell_price = current_price * (1 + sell_pct)
            margin = (sell_price - buy_price) / buy_price

            if margin < constraints.min_margin_pct:
                continue

            # Combined fill probability
            combined_fill = buy_fill_prob[buy_idx] * sell_fill_prob[sell_idx]

            # Expected value (what we actually care about!)
            total_hours = HORIZONS[sell_idx]
            ev = margin * combined_fill / np.sqrt(total_hours)

            if ev > best_ev:
                best_ev = ev
                best_buy_idx = buy_idx
                best_sell_idx = sell_idx

    # No valid trade found
    if best_buy_idx is None:
        return TradeRecommendation(
            recommend_trade=False,
            reason=f"No trade meets {constraints.min_margin_pct*100:.1f}% min margin "
                   f"within {constraints.max_hold_hours}h with acceptable fill probability"
        )

    # Calculate final recommendation
    buy_pct = low_quantiles[best_buy_idx, buy_q]
    sell_pct = high_quantiles[best_sell_idx, sell_q]
    buy_price = current_price * (1 + buy_pct)
    sell_price = current_price * (1 + sell_pct)
    margin = (sell_price - buy_price) / buy_price

    b_fill = buy_fill_prob[best_buy_idx]
    s_fill = sell_fill_prob[best_sell_idx]
    combined_fill = b_fill * s_fill

    # Position sizing (volume-aware)
    max_position_gp = constraints.capital_gp * constraints.max_position_pct
    capital_based_qty = int(max_position_gp / buy_price)

    # Limit by volume (use 15% of expected volume for reliable fills)
    volume_based_qty = int(min(buy_volume[best_buy_idx], sell_volume[best_sell_idx]) * 0.15)

    final_qty = min(capital_based_qty, volume_based_qty)
    volume_limited = volume_based_qty < capital_based_qty

    # Profit calculations
    profit_per_item = int(sell_price - buy_price)
    expected_profit = int(profit_per_item * final_qty * combined_fill)

    # Confidence
    price_spread = (high_quantiles[best_sell_idx, 4] - high_quantiles[best_sell_idx, 0] +
                    low_quantiles[best_buy_idx, 4] - low_quantiles[best_buy_idx, 0])
    price_confidence = 1.0 / (1.0 + price_spread)
    volume_confidence = min(b_fill, s_fill)
    overall_confidence = price_confidence * combined_fill

    return TradeRecommendation(
        recommend_trade=True,
        buy_horizon_hours=HORIZONS[best_buy_idx],
        sell_horizon_hours=HORIZONS[best_sell_idx],
        buy_price_gp=int(buy_price),
        sell_price_gp=int(sell_price),
        buy_fill_probability=b_fill,
        sell_fill_probability=s_fill,
        combined_fill_probability=combined_fill,
        recommended_quantity=final_qty,
        volume_limited=volume_limited,
        margin_pct=margin,
        expected_margin_pct=margin * combined_fill,
        profit_per_item_gp=profit_per_item,
        expected_total_profit_gp=expected_profit,
        price_confidence=price_confidence,
        volume_confidence=volume_confidence,
        overall_confidence=overall_confidence,
    )


def format_recommendation(rec: TradeRecommendation, item_name: str) -> str:
    """Format recommendation for display."""

    if not rec.recommend_trade:
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  {item_name:^66}  ║
╠══════════════════════════════════════════════════════════════════════╣
║  ❌ NO TRADE RECOMMENDED                                              ║
║                                                                      ║
║  {rec.reason:<66}  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    vol_warning = "⚠️  VOLUME LIMITED" if rec.volume_limited else ""
    fill_warning = "⚠️  LOW FILL PROB" if rec.combined_fill_probability < 0.5 else ""

    return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  {item_name:^66}  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📥 BUY ORDER                                                        ║
║     Price:         {rec.buy_price_gp:>10,} GP                                     ║
║     Fill window:   {rec.buy_horizon_hours:>10}h                                        ║
║     Fill prob:     {rec.buy_fill_probability*100:>10.0f}%                                       ║
║                                                                      ║
║  📤 SELL ORDER (after buy fills)                                     ║
║     Price:         {rec.sell_price_gp:>10,} GP                                     ║
║     Complete by:   {rec.sell_horizon_hours:>10}h total                                 ║
║     Fill prob:     {rec.sell_fill_probability*100:>10.0f}%                                       ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  📊 POSITION                                                         ║
║     Quantity:      {rec.recommended_quantity:>10,} units  {vol_warning:<20}         ║
║     Combined fill: {rec.combined_fill_probability*100:>10.0f}%  {fill_warning:<20}         ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  💰 EXPECTED PROFIT                                                  ║
║     Raw margin:    {rec.margin_pct*100:>10.1f}%                                       ║
║     Expected margin: {rec.expected_margin_pct*100:>8.1f}%  (margin × fill prob)            ║
║     Per item:      {rec.profit_per_item_gp:>10,} GP                                     ║
║     Expected total:{rec.expected_total_profit_gp:>10,} GP                                     ║
║                                                                      ║
║  📈 Confidence:    {rec.overall_confidence*100:>10.0f}%                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""


class JointTimingLoss(nn.Module):
    """
    Loss function for joint timing optimizer.

    Combines:
    - Cross entropy for buy/sell horizon classification
    - MSE for expected value prediction
    - Binary cross entropy for skip prediction
    """

    def __init__(
        self,
        ev_weight: float = 1.0,
        skip_weight: float = 0.5,
    ):
        super().__init__()
        self.ev_weight = ev_weight
        self.skip_weight = skip_weight
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        buy_labels: torch.Tensor,
        sell_labels: torch.Tensor,
        ev_labels: torch.Tensor,
        skip_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        buy_loss = self.ce(outputs['buy_logits'], buy_labels)
        sell_loss = self.ce(outputs['sell_logits'], sell_labels)
        ev_loss = self.mse(outputs['expected_value'], ev_labels)

        total = buy_loss + sell_loss + self.ev_weight * ev_loss

        if skip_labels is not None:
            skip_loss = self.bce(
                outputs['skip_prob'].view(-1),
                skip_labels.float()
            )
            total += self.skip_weight * skip_loss

        return total
