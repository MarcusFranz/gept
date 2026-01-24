"""
Cached Predictions Architecture
===============================

Separates ML inference (expensive, scheduled) from user personalization (cheap, per-request).

Design:
    1. INFERENCE LAYER (GPU, runs every 5 min via cron)
       - Predicts ALL outcomes for each item
       - Writes to predictions cache (DB or memory)
       - No user-specific logic here

    2. SERVING LAYER (CPU, runs per-request)
       - Reads cached predictions
       - Applies user constraints (filtering, ranking)
       - Pure math, no ML

This means:
    - 1000 users with different preferences = 1 inference run
    - User can change preferences instantly (no re-inference)
    - Response time ~1ms instead of ~10ms
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

HORIZONS = [1, 2, 4, 8, 12, 24, 48]
N_HORIZONS = len(HORIZONS)
N_QUANTILES = 5
QUANTILE_NAMES = ['p10', 'p30', 'p50', 'p70', 'p90']


# =============================================================================
# CACHED PREDICTION (output of ML inference, stored in DB)
# =============================================================================

@dataclass
class CachedItemPrediction:
    """
    Everything the ML model predicts for ONE item at ONE point in time.

    This gets written to the predictions table and cached.
    Contains ALL information needed to serve ANY user's request.
    """
    item_id: int
    timestamp: datetime
    current_price: float

    # Price predictions: (7 horizons, 5 quantiles) for high and low
    # Stored as percentage change from current_price
    high_quantiles: np.ndarray  # (7, 5)
    low_quantiles: np.ndarray   # (7, 5)

    # Volume predictions: expected volume at each horizon
    buy_volume: np.ndarray      # (7,) units expected to trade
    sell_volume: np.ndarray     # (7,)

    # Fill probability: P(order fills) at each horizon
    # These are for market-price orders; adjusted for price offset at serving time
    buy_fill_prob: np.ndarray   # (7,) base fill probability
    sell_fill_prob: np.ndarray  # (7,)

    # Model metadata
    model_version: str = "v1"

    def to_dict(self) -> Dict:
        """Serialize for storage."""
        return {
            'item_id': self.item_id,
            'timestamp': self.timestamp.isoformat(),
            'current_price': self.current_price,
            'high_quantiles': self.high_quantiles.tolist(),
            'low_quantiles': self.low_quantiles.tolist(),
            'buy_volume': self.buy_volume.tolist(),
            'sell_volume': self.sell_volume.tolist(),
            'buy_fill_prob': self.buy_fill_prob.tolist(),
            'sell_fill_prob': self.sell_fill_prob.tolist(),
            'model_version': self.model_version,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'CachedItemPrediction':
        """Deserialize from storage."""
        return cls(
            item_id=d['item_id'],
            timestamp=datetime.fromisoformat(d['timestamp']),
            current_price=d['current_price'],
            high_quantiles=np.array(d['high_quantiles']),
            low_quantiles=np.array(d['low_quantiles']),
            buy_volume=np.array(d['buy_volume']),
            sell_volume=np.array(d['sell_volume']),
            buy_fill_prob=np.array(d['buy_fill_prob']),
            sell_fill_prob=np.array(d['sell_fill_prob']),
            model_version=d.get('model_version', 'v1'),
        )


# =============================================================================
# USER CONSTRAINTS (input at serving time)
# =============================================================================

@dataclass
class UserConstraints:
    """User preferences applied at serving time (not during inference)."""

    # Risk profile
    risk_tolerance: float = 0.5      # 0=conservative, 1=aggressive

    # Time constraints
    max_hold_hours: int = 48         # Maximum willing to hold
    min_hold_hours: int = 1          # Minimum hold (avoid ultra-short flips)

    # Profit requirements
    min_margin_pct: float = 0.02     # Minimum acceptable raw margin
    min_expected_margin_pct: float = 0.01  # Minimum expected margin (margin × fill_prob)

    # Position sizing
    capital_gp: int = 10_000_000     # Available capital
    max_position_pct: float = 0.10   # Max % of capital per trade
    max_quantity: Optional[int] = None  # Hard cap on quantity

    # Fill probability requirements
    min_fill_prob: float = 0.3       # Won't recommend if fill prob below this


# =============================================================================
# TRADE OPPORTUNITY (computed at serving time from cache + constraints)
# =============================================================================

@dataclass
class TradeOpportunity:
    """A specific trade opportunity for a user, computed from cached predictions."""

    item_id: int

    # Timing (which cached horizons were selected)
    buy_horizon_idx: int
    sell_horizon_idx: int
    buy_horizon_hours: int
    sell_horizon_hours: int

    # Prices (computed from cache using user's risk quantiles)
    buy_price_gp: int
    sell_price_gp: int

    # Fill probabilities (adjusted for price offset)
    buy_fill_prob: float
    sell_fill_prob: float
    combined_fill_prob: float

    # Position (sized by capital AND volume constraints)
    quantity: int
    position_gp: int
    volume_limited: bool

    # Profitability
    raw_margin_pct: float           # (sell - buy) / buy
    expected_margin_pct: float      # raw_margin × combined_fill_prob
    profit_per_item_gp: int
    raw_total_profit_gp: int        # If everything fills
    expected_profit_gp: int         # Probability-weighted

    # Ranking score (for sorting multiple opportunities)
    score: float

    # Which quantiles were used (for transparency)
    buy_quantile: str
    sell_quantile: str


# =============================================================================
# SERVING LAYER (pure functions, no ML)
# =============================================================================

def get_quantile_indices(risk_tolerance: float) -> Tuple[int, int]:
    """
    Map risk tolerance to quantile indices.

    Conservative (0.0-0.33): Buy at p30 (higher), sell at p70 (lower) = safer
    Moderate (0.33-0.66):    Buy at p50 (median), sell at p50 (median)
    Aggressive (0.66-1.0):   Buy at p10 (lower), sell at p90 (higher) = riskier
    """
    if risk_tolerance < 0.33:
        return 1, 3  # p30, p70
    elif risk_tolerance < 0.66:
        return 2, 2  # p50, p50
    else:
        return 0, 4  # p10, p90


def adjust_fill_prob_for_price(
    base_fill_prob: float,
    price_offset_pct: float,
    side: str,
) -> float:
    """
    Adjust base fill probability for how far from market price we're bidding.

    If buying BELOW market or selling ABOVE market, fill prob decreases.
    If buying AT or ABOVE market, fill prob stays high.

    Args:
        base_fill_prob: Fill probability at market price (from cache)
        price_offset_pct: How far from current price (-0.05 = 5% below)
        side: 'buy' or 'sell'
    """
    # Determine if we're on the "hard" side of the spread
    if side == 'buy':
        # Buying below market is harder
        if price_offset_pct < 0:
            penalty = np.exp(price_offset_pct * 7)  # e.g., -5% → 0.70
        else:
            penalty = 1.0  # Buying at/above market = easy
    else:  # sell
        # Selling above market is harder
        if price_offset_pct > 0:
            penalty = np.exp(-price_offset_pct * 7)  # e.g., +5% → 0.70
        else:
            penalty = 1.0  # Selling at/below market = easy

    adjusted = base_fill_prob * penalty
    return np.clip(adjusted, 0.05, 0.95)


def compute_trade_opportunities(
    prediction: CachedItemPrediction,
    constraints: UserConstraints,
) -> List[TradeOpportunity]:
    """
    Given cached predictions and user constraints, find all valid trade opportunities.

    This is the main serving function. It's pure math, no ML.
    Runs in <1ms on CPU.
    """
    opportunities = []

    # Get quantile indices for this risk level
    buy_q, sell_q = get_quantile_indices(constraints.risk_tolerance)

    # Determine horizon range from constraints
    min_h = 0
    max_h = N_HORIZONS - 1
    for i, h in enumerate(HORIZONS):
        if h >= constraints.min_hold_hours and min_h == 0:
            min_h = i
        if h >= constraints.max_hold_hours:
            max_h = i
            break

    # Try all valid (buy_horizon, sell_horizon) combinations
    for buy_idx in range(min_h, min(max_h, N_HORIZONS - 1)):
        for sell_idx in range(buy_idx + 1, max_h + 1):

            # Get prices from cached quantiles
            buy_pct = prediction.low_quantiles[buy_idx, buy_q]
            sell_pct = prediction.high_quantiles[sell_idx, sell_q]

            buy_price = prediction.current_price * (1 + buy_pct)
            sell_price = prediction.current_price * (1 + sell_pct)

            # Calculate margin
            raw_margin = (sell_price - buy_price) / buy_price

            # Skip if below minimum margin
            if raw_margin < constraints.min_margin_pct:
                continue

            # Calculate fill probabilities (adjusted for price offset)
            buy_fill = adjust_fill_prob_for_price(
                prediction.buy_fill_prob[buy_idx],
                buy_pct,
                'buy'
            )
            sell_fill = adjust_fill_prob_for_price(
                prediction.sell_fill_prob[sell_idx],
                sell_pct,
                'sell'
            )
            combined_fill = buy_fill * sell_fill

            # Skip if fill probability too low
            if combined_fill < constraints.min_fill_prob:
                continue

            # Calculate expected margin
            expected_margin = raw_margin * combined_fill

            # Skip if expected margin too low
            if expected_margin < constraints.min_expected_margin_pct:
                continue

            # Position sizing
            max_position_gp = constraints.capital_gp * constraints.max_position_pct
            capital_qty = int(max_position_gp / buy_price)

            # Volume-based limit (15% of expected volume for reliable fills)
            volume_qty = int(min(
                prediction.buy_volume[buy_idx],
                prediction.sell_volume[sell_idx]
            ) * 0.15)

            # Final quantity
            qty = min(capital_qty, volume_qty)
            if constraints.max_quantity:
                qty = min(qty, constraints.max_quantity)

            if qty < 1:
                continue

            volume_limited = volume_qty < capital_qty

            # Calculate profits
            profit_per = int(sell_price - buy_price)
            raw_total = profit_per * qty
            expected_total = int(raw_total * combined_fill)

            # Score for ranking (expected profit per sqrt(hour))
            total_hours = HORIZONS[sell_idx]
            score = expected_margin / np.sqrt(total_hours)

            opportunities.append(TradeOpportunity(
                item_id=prediction.item_id,
                buy_horizon_idx=buy_idx,
                sell_horizon_idx=sell_idx,
                buy_horizon_hours=HORIZONS[buy_idx],
                sell_horizon_hours=HORIZONS[sell_idx],
                buy_price_gp=int(buy_price),
                sell_price_gp=int(sell_price),
                buy_fill_prob=buy_fill,
                sell_fill_prob=sell_fill,
                combined_fill_prob=combined_fill,
                quantity=qty,
                position_gp=int(buy_price * qty),
                volume_limited=volume_limited,
                raw_margin_pct=raw_margin,
                expected_margin_pct=expected_margin,
                profit_per_item_gp=profit_per,
                raw_total_profit_gp=raw_total,
                expected_profit_gp=expected_total,
                score=score,
                buy_quantile=QUANTILE_NAMES[buy_q],
                sell_quantile=QUANTILE_NAMES[sell_q],
            ))

    # Sort by score (best first)
    opportunities.sort(key=lambda x: x.score, reverse=True)

    return opportunities


def get_best_opportunity(
    prediction: CachedItemPrediction,
    constraints: UserConstraints,
) -> Optional[TradeOpportunity]:
    """Get the single best trade opportunity for this user."""
    opps = compute_trade_opportunities(prediction, constraints)
    return opps[0] if opps else None


def format_opportunity(opp: TradeOpportunity, item_name: str) -> str:
    """Format a trade opportunity for display."""

    vol_flag = " ⚠️ VOLUME LIMITED" if opp.volume_limited else ""
    fill_flag = " ⚠️ LOW FILL" if opp.combined_fill_prob < 0.5 else ""

    return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  {item_name:^66}  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Strategy: Buy at {opp.buy_quantile}, Sell at {opp.sell_quantile}  (score: {opp.score:.4f})               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📥 BUY                                                              ║
║     Price:       {opp.buy_price_gp:>12,} GP                                   ║
║     Window:      within {opp.buy_horizon_hours:>2}h                                        ║
║     Fill prob:   {opp.buy_fill_prob*100:>11.0f}%                                       ║
║                                                                      ║
║  📤 SELL (after buy fills)                                           ║
║     Price:       {opp.sell_price_gp:>12,} GP                                   ║
║     Complete by: {opp.sell_horizon_hours:>6}h total                                    ║
║     Fill prob:   {opp.sell_fill_prob*100:>11.0f}%                                       ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  📊 POSITION{vol_flag:<20}                                    ║
║     Quantity:    {opp.quantity:>12,} units                                  ║
║     Investment:  {opp.position_gp:>12,} GP                                   ║
║     Combined fill: {opp.combined_fill_prob*100:>9.0f}%{fill_flag:<20}                ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  💰 PROFIT                                                           ║
║     Raw margin:      {opp.raw_margin_pct*100:>8.2f}%                                    ║
║     Expected margin: {opp.expected_margin_pct*100:>8.2f}%  (× fill prob)                 ║
║     Per item:        {opp.profit_per_item_gp:>8,} GP                                   ║
║     If all fills:    {opp.raw_total_profit_gp:>8,} GP                                   ║
║     Expected:        {opp.expected_profit_gp:>8,} GP                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# DEMO: Show same cache, different users
# =============================================================================

def demo_cached_serving():
    """
    Demonstrate serving different users from the same cached prediction.
    """

    # Simulate a cached prediction (would come from DB in production)
    cached = CachedItemPrediction(
        item_id=0,
        timestamp=datetime.now(),
        current_price=180.0,
        high_quantiles=np.array([
            [0.005, 0.008, 0.012, 0.018, 0.025],  # 1h
            [0.008, 0.012, 0.018, 0.025, 0.035],  # 2h
            [0.012, 0.018, 0.025, 0.035, 0.050],  # 4h
            [0.018, 0.025, 0.035, 0.050, 0.070],  # 8h
            [0.025, 0.035, 0.050, 0.070, 0.090],  # 12h
            [0.035, 0.050, 0.070, 0.090, 0.120],  # 24h
            [0.050, 0.070, 0.090, 0.120, 0.150],  # 48h
        ]),
        low_quantiles=np.array([
            [-0.025, -0.018, -0.012, -0.008, -0.005],  # 1h
            [-0.035, -0.025, -0.018, -0.012, -0.008],  # 2h
            [-0.050, -0.035, -0.025, -0.018, -0.012],  # 4h
            [-0.070, -0.050, -0.035, -0.025, -0.018],  # 8h
            [-0.090, -0.070, -0.050, -0.035, -0.025],  # 12h
            [-0.120, -0.090, -0.070, -0.050, -0.035],  # 24h
            [-0.150, -0.120, -0.090, -0.070, -0.050],  # 48h
        ]),
        buy_volume=np.array([5000, 8000, 15000, 25000, 35000, 60000, 100000]),
        sell_volume=np.array([5000, 8000, 15000, 25000, 35000, 60000, 100000]),
        buy_fill_prob=np.array([0.40, 0.55, 0.70, 0.82, 0.88, 0.93, 0.97]),
        sell_fill_prob=np.array([0.40, 0.55, 0.70, 0.82, 0.88, 0.93, 0.97]),
    )

    print("=" * 74)
    print("CACHED PREDICTION (computed once, serves all users)")
    print("=" * 74)
    print(f"Item: Cannonball | Price: {cached.current_price} GP")
    print(f"Timestamp: {cached.timestamp}")
    print()

    # Define different user profiles
    users = [
        ("Conservative Carl", UserConstraints(
            risk_tolerance=0.2,
            max_hold_hours=24,
            min_margin_pct=0.03,
            min_fill_prob=0.5,
            capital_gp=5_000_000,
        )),
        ("Moderate Mary", UserConstraints(
            risk_tolerance=0.5,
            max_hold_hours=48,
            min_margin_pct=0.02,
            min_fill_prob=0.3,
            capital_gp=20_000_000,
        )),
        ("Aggressive Alex", UserConstraints(
            risk_tolerance=0.9,
            max_hold_hours=48,
            min_margin_pct=0.01,
            min_fill_prob=0.2,
            capital_gp=100_000_000,
        )),
    ]

    for name, constraints in users:
        print("=" * 74)
        print(f"USER: {name}")
        print(f"  Risk: {constraints.risk_tolerance}, Max hold: {constraints.max_hold_hours}h")
        print(f"  Min margin: {constraints.min_margin_pct*100}%, Min fill: {constraints.min_fill_prob*100}%")
        print(f"  Capital: {constraints.capital_gp:,} GP")
        print("=" * 74)

        opp = get_best_opportunity(cached, constraints)

        if opp:
            print(format_opportunity(opp, "Cannonball"))
        else:
            print("\n  ❌ No trade meets this user's constraints\n")

        # Show all opportunities for this user
        all_opps = compute_trade_opportunities(cached, constraints)
        if len(all_opps) > 1:
            print(f"  ({len(all_opps)} total opportunities found, showing best)")
        print()


if __name__ == '__main__':
    demo_cached_serving()
