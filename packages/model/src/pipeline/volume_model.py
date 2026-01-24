"""
Volume & Liquidity Model
========================

Predicts expected fill volume at different price levels.

Key questions this layer answers:
1. How much volume trades at price X within Y hours?
2. How long to fill an order of size N at price X?
3. What's the realistic max position given liquidity?

Architecture:
    Recent volume features → Volume MLP → Fill estimates per price/horizon
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

HORIZONS = [1, 2, 4, 8, 12, 24, 48]
N_HORIZONS = len(HORIZONS)


@dataclass
class VolumeEstimate:
    """Volume and fill time estimates for a trade."""

    # Expected volume at target price within horizon
    expected_volume: int

    # Estimated time to fill order (hours)
    fill_time_hours: float

    # Probability of complete fill within horizon
    fill_probability: float

    # Recommended max quantity given liquidity
    recommended_max_qty: int

    # Volume confidence (based on historical variance)
    confidence: float


class VolumeHead(nn.Module):
    """
    Predicts volume available at different price offsets.

    Input: Volume features from recent/medium/long timeframes
    Output: Expected volume at each horizon for buy/sell sides
    """

    def __init__(
        self,
        recent_vol_features: int = 4,  # high_vol, low_vol, avg_vol, vol_ratio
        medium_vol_features: int = 4,
        long_vol_features: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()

        total_features = (
            recent_vol_features * 24 +   # Last 24 hours of 5-min volume
            medium_vol_features * 7 +    # Last 7 days hourly
            long_vol_features * 7        # Last 30 days daily (weekly aggregates)
        )

        self.encoder = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Predict log volume (always positive) for each horizon
        self.buy_volume_head = nn.Linear(hidden_dim, N_HORIZONS)
        self.sell_volume_head = nn.Linear(hidden_dim, N_HORIZONS)

        # Predict volume uncertainty (for confidence)
        self.uncertainty_head = nn.Linear(hidden_dim, N_HORIZONS * 2)

    def forward(self, volume_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            volume_features: (batch, total_features) concatenated volume data

        Returns:
            Dict with buy_volume, sell_volume, uncertainty estimates
        """
        features = self.encoder(volume_features)

        # Log volume predictions (exp to get actual volume)
        buy_log_vol = self.buy_volume_head(features)
        sell_log_vol = self.sell_volume_head(features)

        # Uncertainty (log scale)
        uncertainty = self.uncertainty_head(features)
        buy_unc = uncertainty[:, :N_HORIZONS]
        sell_unc = uncertainty[:, N_HORIZONS:]

        return {
            'buy_log_volume': buy_log_vol,
            'sell_log_volume': sell_log_vol,
            'buy_volume': torch.exp(buy_log_vol),
            'sell_volume': torch.exp(sell_log_vol),
            'buy_uncertainty': torch.exp(buy_unc),
            'sell_uncertainty': torch.exp(sell_unc),
        }


def extract_volume_features(recent: np.ndarray) -> np.ndarray:
    """
    Extract volume-related features from recent data.

    Args:
        recent: (288, 6) recent features where:
            [:, 2] = log high volume
            [:, 3] = log low volume

    Returns:
        Volume feature vector for the volume model
    """
    # Recent has 288 5-min periods = 24 hours
    # Extract hourly volume aggregates

    high_vol = np.exp(recent[:, 2])  # Convert from log
    low_vol = np.exp(recent[:, 3])

    # Hourly aggregates (12 periods per hour)
    hourly_high = []
    hourly_low = []
    hourly_total = []
    hourly_ratio = []

    for h in range(24):
        start = h * 12
        end = (h + 1) * 12
        h_high = high_vol[start:end].sum()
        h_low = low_vol[start:end].sum()
        hourly_high.append(h_high)
        hourly_low.append(h_low)
        hourly_total.append(h_high + h_low)
        hourly_ratio.append(h_high / (h_low + 1e-6))

    # Build feature vector
    features = np.concatenate([
        np.log1p(hourly_high),   # 24 features
        np.log1p(hourly_low),    # 24 features
        np.log1p(hourly_total),  # 24 features
        np.clip(hourly_ratio, 0, 10),  # 24 features (clipped buy/sell ratio)
    ])

    return features.astype(np.float32)


def estimate_fill_volume(
    recent_features: np.ndarray,
    price_offset_pct: float,
    horizon_hours: int,
    side: str = 'buy',
    item_liquidity: str = 'high',
) -> VolumeEstimate:
    """
    Estimate available volume at a given price offset.

    This is a heuristic model based on observed OSRS GE patterns.
    Can be replaced with learned model.

    Args:
        recent_features: (288, 6) recent data
        price_offset_pct: How far from current price (-0.05 = 5% below)
        horizon_hours: Time window
        side: 'buy' or 'sell'
        item_liquidity: 'high' (cannonball), 'medium', 'low' (rare items)

    Returns:
        VolumeEstimate with fill predictions
    """
    # Extract base volume from recent data
    high_vol = np.exp(recent_features[:, 2])
    low_vol = np.exp(recent_features[:, 3])

    # Average hourly volume
    total_vol = high_vol.sum() + low_vol.sum()
    hourly_vol = total_vol / 24

    # Volume at target horizon
    horizon_vol = hourly_vol * horizon_hours

    # --- FIXED FILL PROBABILITY MODEL ---
    # Base fill rate at market price (scales with item liquidity)
    base_rates = {
        'high': 0.95,    # Very liquid items (cannonball, runes)
        'medium': 0.80,  # Medium liquidity
        'low': 0.50,     # Rare items (bandos, etc.)
    }
    base_rate = base_rates.get(item_liquidity, 0.80)

    # Time factor: longer horizon = higher fill probability
    # ~39% at 1h, ~63% at 2h, ~86% at 4h, ~95% at 8h
    time_factor = 1 - np.exp(-horizon_hours / 2)

    # Price deviation penalty (gentler curve than before)
    # At ±2%: ~87% of base | At ±5%: ~70% | At ±10%: ~50% | At ±15%: ~35%
    price_factor = np.exp(-abs(price_offset_pct) * 7)

    # Side factor: buying below market or selling above is harder
    if side == 'buy' and price_offset_pct < 0:
        side_factor = 0.9  # Slight penalty for below-market bids
    elif side == 'sell' and price_offset_pct > 0:
        side_factor = 0.9  # Slight penalty for above-market asks
    else:
        side_factor = 1.0  # At or better than market = easy fill

    # Combined fill probability
    fill_prob = base_rate * time_factor * price_factor * side_factor
    fill_prob = min(0.95, max(0.05, fill_prob))

    # Expected volume at our price level
    volume_at_price = horizon_vol * price_factor
    expected_vol = int(volume_at_price)

    # Recommended max quantity based on liquidity
    # Target: order size that can fill with 80% probability
    # Rule of thumb: 10-20% of expected volume for reliable fills
    reliable_fraction = 0.15 if item_liquidity == 'high' else 0.10
    recommended_qty = int(expected_vol * reliable_fraction)

    # Confidence based on volume consistency
    vol_std = np.std([high_vol.sum(), low_vol.sum()])
    vol_mean = np.mean([high_vol.sum(), low_vol.sum()])
    confidence = 1.0 / (1.0 + vol_std / (vol_mean + 1))

    # Fill time estimate (assumes uniform distribution within window)
    fill_time = horizon_hours * (1 - fill_prob * 0.5)  # Higher prob = faster fill

    return VolumeEstimate(
        expected_volume=max(0, expected_vol),
        fill_time_hours=fill_time,
        fill_probability=fill_prob,
        recommended_max_qty=max(1, recommended_qty),
        confidence=confidence,
    )


def adjust_recommendation_for_volume(
    recommendation: Dict,
    recent_features: np.ndarray,
    current_price: float,
) -> Dict:
    """
    Adjust a trade recommendation based on volume constraints.

    Args:
        recommendation: Output from apply_user_preferences
        recent_features: (288, 6) for volume estimation
        current_price: Current price in GP

    Returns:
        Updated recommendation with volume-adjusted quantities
    """
    if not recommendation.get('recommend_trade', False):
        return recommendation

    rec = recommendation.copy()

    # Get price offsets
    buy_price = rec['buy_price_gp']
    sell_price = rec['sell_price_gp']
    buy_offset = (buy_price - current_price) / current_price
    sell_offset = (sell_price - current_price) / current_price

    # Estimate volume for buy side
    buy_vol = estimate_fill_volume(
        recent_features,
        buy_offset,
        rec['buy_horizon_hours'],
        side='buy'
    )

    # Estimate volume for sell side
    sell_vol = estimate_fill_volume(
        recent_features,
        sell_offset,
        rec['sell_horizon_hours'] - rec['buy_horizon_hours'],  # Time after buy
        side='sell'
    )

    # Adjust quantity to minimum of user's preference and volume constraint
    original_qty = rec['recommended_quantity']
    volume_limited_qty = min(buy_vol.recommended_max_qty, sell_vol.recommended_max_qty)
    final_qty = min(original_qty, volume_limited_qty)

    # Update recommendation
    rec['recommended_quantity'] = final_qty
    rec['total_profit_gp'] = int(rec['profit_per_item_gp'] * final_qty)

    # Add volume info
    rec['volume_info'] = {
        'buy_expected_volume': buy_vol.expected_volume,
        'buy_fill_probability': buy_vol.fill_probability,
        'buy_fill_time_hours': buy_vol.fill_time_hours,
        'sell_expected_volume': sell_vol.expected_volume,
        'sell_fill_probability': sell_vol.fill_probability,
        'sell_fill_time_hours': sell_vol.fill_time_hours,
        'volume_limited': volume_limited_qty < original_qty,
        'original_quantity': original_qty,
        'volume_confidence': (buy_vol.confidence + sell_vol.confidence) / 2,
    }

    # Adjust overall confidence
    combined_fill_prob = buy_vol.fill_probability * sell_vol.fill_probability
    rec['confidence'] = rec['confidence'] * combined_fill_prob

    # Flag if volume is a concern
    if combined_fill_prob < 0.5:
        rec['volume_warning'] = f"Low fill probability ({combined_fill_prob:.0%}). Consider smaller position or wider prices."

    return rec


def format_volume_recommendation(rec: Dict, item_name: str) -> str:
    """Format recommendation with volume info."""

    if not rec.get('recommend_trade', False):
        return f"❌ No trade recommended: {rec.get('reason', 'Unknown')}"

    vol = rec.get('volume_info', {})
    warning = rec.get('volume_warning', '')

    return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  {item_name:^66}  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📥 BUY ORDER                                                        ║
║     Price:     {rec['buy_price_gp']:>10,} GP                                       ║
║     Quantity:  {rec['recommended_quantity']:>10,} units                                    ║
║     Window:    {rec['buy_horizon_hours']:>10}h                                         ║
║     Est. Volume: {vol.get('buy_expected_volume', 0):>8,} units available                   ║
║     Fill Prob:   {vol.get('buy_fill_probability', 0)*100:>7.0f}%                                     ║
║                                                                      ║
║  📤 SELL ORDER (after buy fills)                                     ║
║     Price:     {rec['sell_price_gp']:>10,} GP                                       ║
║     Window:    Complete by {rec['sell_horizon_hours']:>2}h total                            ║
║     Est. Volume: {vol.get('sell_expected_volume', 0):>8,} units available                   ║
║     Fill Prob:   {vol.get('sell_fill_probability', 0)*100:>7.0f}%                                     ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  💰 PROFIT (volume-adjusted)                                         ║
║     Per Item:  {rec['profit_per_item_gp']:>10,} GP  ({rec['margin_pct']*100:>5.1f}%)                      ║
║     Total:     {rec['total_profit_gp']:>10,} GP                                       ║
║     Confidence:{rec['confidence']*100:>10.0f}%                                       ║
║                                                                      ║
{'║  ⚠️  ' + warning[:60] + ' '*(60-len(warning[:60])) + '  ║' if warning else '║' + ' '*70 + '║'}
╚══════════════════════════════════════════════════════════════════════╝
"""
