"""
User Preference Layer
=====================

Adapts trade recommendations to individual user preferences.

User Preferences:
- risk_tolerance: conservative (0) to aggressive (1)
- max_hold_hours: maximum time willing to hold a flip
- min_margin_pct: minimum profit margin to consider
- capital_gp: available capital (affects position sizing)
- active_hours: when user can check trades

This layer sits on top of the Trade Optimizer and filters/adjusts
recommendations based on user constraints.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

HORIZONS = [1, 2, 4, 8, 12, 24, 48]
QUANTILE_NAMES = ['p10', 'p30', 'p50', 'p70', 'p90']


class RiskLevel(Enum):
    CONSERVATIVE = 0.2   # Use p30/p70 quantiles
    MODERATE = 0.5       # Use p50 (median)
    AGGRESSIVE = 0.8     # Use p10/p90 quantiles


@dataclass
class UserPreferences:
    """User-configurable trading preferences."""

    # Risk tolerance: 0 = conservative, 1 = aggressive
    risk_tolerance: float = 0.5

    # Maximum hours willing to hold a position
    max_hold_hours: int = 48

    # Minimum acceptable profit margin (0.05 = 5%)
    min_margin_pct: float = 0.02

    # Available capital in GP
    capital_gp: int = 10_000_000

    # Maximum GP to risk on single trade (% of capital)
    max_position_pct: float = 0.1

    # Hours user is active (for timing recommendations)
    active_hours: Tuple[int, int] = (8, 22)  # 8am to 10pm

    def quantile_for_buy(self) -> int:
        """Which quantile to use for buy price (lower = more aggressive)."""
        if self.risk_tolerance < 0.33:
            return 1  # p30 - conservative, higher buy price
        elif self.risk_tolerance < 0.66:
            return 2  # p50 - median
        else:
            return 0  # p10 - aggressive, targets lowest price

    def quantile_for_sell(self) -> int:
        """Which quantile to use for sell price (higher = more aggressive)."""
        if self.risk_tolerance < 0.33:
            return 3  # p70 - conservative, lower sell price
        elif self.risk_tolerance < 0.66:
            return 2  # p50 - median
        else:
            return 4  # p90 - aggressive, targets highest price

    def max_horizon_idx(self) -> int:
        """Get maximum horizon index based on max_hold_hours."""
        for i, h in enumerate(HORIZONS):
            if h >= self.max_hold_hours:
                return i
        return len(HORIZONS) - 1


def apply_user_preferences(
    high_quantiles: np.ndarray,
    low_quantiles: np.ndarray,
    current_price: float,
    prefs: UserPreferences,
) -> Dict:
    """
    Generate personalized trade recommendation.

    Args:
        high_quantiles: (7, 5) predicted max highs
        low_quantiles: (7, 5) predicted min lows
        current_price: current item price in GP
        prefs: user preferences

    Returns:
        Personalized recommendation dict
    """
    buy_q = prefs.quantile_for_buy()
    sell_q = prefs.quantile_for_sell()
    max_h = prefs.max_horizon_idx()

    best_margin = -np.inf
    best_buy_idx = None
    best_sell_idx = None

    # Find best trade within user constraints
    for buy_idx in range(min(max_h, len(HORIZONS) - 1)):
        for sell_idx in range(buy_idx + 1, max_h + 1):
            buy_pct = low_quantiles[buy_idx, buy_q]
            sell_pct = high_quantiles[sell_idx, sell_q]

            buy_price = current_price * (1 + buy_pct)
            sell_price = current_price * (1 + sell_pct)
            margin = (sell_price - buy_price) / buy_price

            # Skip if below minimum margin
            if margin < prefs.min_margin_pct:
                continue

            # Time-adjusted margin
            total_hours = HORIZONS[sell_idx]
            adjusted = margin / np.sqrt(total_hours)

            if adjusted > best_margin:
                best_margin = adjusted
                best_buy_idx = buy_idx
                best_sell_idx = sell_idx

    # No valid trade found
    if best_buy_idx is None:
        return {
            'recommend_trade': False,
            'reason': f'No trade meets {prefs.min_margin_pct*100:.1f}% min margin within {prefs.max_hold_hours}h',
        }

    # Calculate trade details
    buy_pct = low_quantiles[best_buy_idx, buy_q]
    sell_pct = high_quantiles[best_sell_idx, sell_q]
    buy_price = current_price * (1 + buy_pct)
    sell_price = current_price * (1 + sell_pct)
    margin = (sell_price - buy_price) / buy_price
    profit_per_item = sell_price - buy_price

    # Position sizing
    max_position_gp = prefs.capital_gp * prefs.max_position_pct
    quantity = int(max_position_gp / buy_price)
    total_profit = profit_per_item * quantity

    # Confidence based on quantile spread
    buy_spread = low_quantiles[best_buy_idx, 4] - low_quantiles[best_buy_idx, 0]
    sell_spread = high_quantiles[best_sell_idx, 4] - high_quantiles[best_sell_idx, 0]
    confidence = 1.0 / (1.0 + buy_spread + sell_spread)  # Higher = more confident

    return {
        'recommend_trade': True,
        'buy_horizon_hours': HORIZONS[best_buy_idx],
        'sell_horizon_hours': HORIZONS[best_sell_idx],
        'buy_price_gp': int(buy_price),
        'sell_price_gp': int(sell_price),
        'margin_pct': margin,
        'profit_per_item_gp': int(profit_per_item),
        'recommended_quantity': quantity,
        'total_profit_gp': int(total_profit),
        'confidence': confidence,
        'risk_level': 'aggressive' if prefs.risk_tolerance > 0.66 else 'moderate' if prefs.risk_tolerance > 0.33 else 'conservative',
        'quantiles_used': {
            'buy': QUANTILE_NAMES[buy_q],
            'sell': QUANTILE_NAMES[sell_q],
        },
    }


def format_recommendation(rec: Dict, item_name: str, prefs: UserPreferences) -> str:
    """Format recommendation for display."""

    if not rec['recommend_trade']:
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  {item_name:^56}  ║
╠══════════════════════════════════════════════════════════════╣
║  ❌ NO TRADE RECOMMENDED                                     ║
║                                                              ║
║  Reason: {rec['reason']:<49} ║
╚══════════════════════════════════════════════════════════════╝
"""

    return f"""
╔══════════════════════════════════════════════════════════════╗
║  {item_name:^56}  ║
╠══════════════════════════════════════════════════════════════╣
║  User Profile: {rec['risk_level'].upper():^12} | Max Hold: {prefs.max_hold_hours:>2}h | Min Margin: {prefs.min_margin_pct*100:.0f}%  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📥 BUY                                                      ║
║     Price:    {rec['buy_price_gp']:>10,} GP                              ║
║     Window:   Fill within {rec['buy_horizon_hours']:>2}h                             ║
║     Quantity: {rec['recommended_quantity']:>10,} units                           ║
║     Using:    {rec['quantiles_used']['buy']:>4} quantile                            ║
║                                                              ║
║  📤 SELL (after buy fills)                                   ║
║     Price:    {rec['sell_price_gp']:>10,} GP                              ║
║     Window:   Complete by {rec['sell_horizon_hours']:>2}h total                       ║
║     Using:    {rec['quantiles_used']['sell']:>4} quantile                            ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  💰 EXPECTED PROFIT                                          ║
║     Per Item: {rec['profit_per_item_gp']:>10,} GP ({rec['margin_pct']*100:>5.1f}%)                     ║
║     Total:    {rec['total_profit_gp']:>10,} GP                              ║
║     Confidence: {rec['confidence']*100:>5.1f}%                                   ║
╚══════════════════════════════════════════════════════════════╝
"""


class PersonalizedTradeHead(nn.Module):
    """
    Neural network that learns to optimize for user preferences.

    Takes price predictions + user preference embedding and outputs
    personalized buy/sell recommendations.
    """

    def __init__(
        self,
        n_pref_features: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()

        # Price features: 7 horizons * 5 quantiles * 2 (high/low) = 70
        price_dim = 70

        # Preference encoder
        self.pref_encoder = nn.Sequential(
            nn.Linear(n_pref_features, 16),
            nn.GELU(),
            nn.Linear(16, 32),
        )

        # Combined processor
        self.combined = nn.Sequential(
            nn.Linear(price_dim + 32, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Outputs
        self.buy_head = nn.Linear(hidden_dim, 7)
        self.sell_head = nn.Linear(hidden_dim, 7)
        self.margin_head = nn.Linear(hidden_dim, 1)
        self.skip_head = nn.Linear(hidden_dim, 1)  # Should we skip this trade?

    def forward(
        self,
        high_quantiles: torch.Tensor,
        low_quantiles: torch.Tensor,
        preferences: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            high_quantiles: (batch, 7, 5)
            low_quantiles: (batch, 7, 5)
            preferences: (batch, 5) - [risk, max_hours_norm, min_margin, capital_norm, max_pos]
        """
        batch_size = high_quantiles.shape[0]

        # Flatten price predictions
        high_flat = high_quantiles.view(batch_size, -1)
        low_flat = low_quantiles.view(batch_size, -1)
        prices = torch.cat([high_flat, low_flat], dim=1)

        # Encode preferences
        pref_embed = self.pref_encoder(preferences)

        # Combine
        combined = torch.cat([prices, pref_embed], dim=1)
        features = self.combined(combined)

        return {
            'buy_logits': self.buy_head(features),
            'sell_logits': self.sell_head(features),
            'margin_pred': self.margin_head(features).squeeze(-1),
            'skip_logit': self.skip_head(features).squeeze(-1),
        }


def encode_preferences(prefs: UserPreferences) -> np.ndarray:
    """Convert UserPreferences to neural network input."""
    return np.array([
        prefs.risk_tolerance,
        prefs.max_hold_hours / 48.0,  # Normalize to [0, 1]
        prefs.min_margin_pct / 0.20,  # Normalize (assuming max 20%)
        np.log10(prefs.capital_gp) / 9.0,  # Log scale, normalized
        prefs.max_position_pct,
    ], dtype=np.float32)
