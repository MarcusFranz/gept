"""
End-to-End Backtest
===================

Simulates trading using the full pipeline on historical data.

Steps:
1. Fetch 6 months of historical data from PostgreSQL (via SSH tunnel)
2. Walk through time, generating predictions at each point
3. Simulate trade execution with volume-based fills
4. Track P&L across different user profiles
5. Report comprehensive metrics

Usage:
    # First, start SSH tunnel to Ampere:
    ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128 -N &

    # Then run backtest:
    python -m src.pipeline.backtest --item-id 2 --months 6

    # Or use API for recent data only (limited to ~1 day):
    python -m src.pipeline.backtest --item-id 2 --months 1 --source api
"""

import argparse
import logging
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HORIZONS = [1, 2, 4, 8, 12, 24, 48]
N_HORIZONS = len(HORIZONS)
N_QUANTILES = 5
GE_TAX_RATE = 0.01  # 1% GE tax on sells

ITEM_INFO = {
    2: ("Cannonball", 180),
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_5min_data_api(item_id: int, months: int = 6) -> Dict:
    """
    Fetch historical 5-minute data from OSRS Wiki API.

    NOTE: API only returns ~1 day of 5-minute data. For longer backtests,
    use fetch_5min_data_db() with SSH tunnel to Ampere.

    Returns dict with timestamps and OHLC data.
    """
    logger.info(f"Fetching 5-min data from API for item {item_id}...")
    logger.warning("API only provides ~1 day of 5-min data. Use --source db for longer backtests.")

    url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries"
    headers = {"User-Agent": "GePT-Backtest/1.0 (contact@gept.gg)"}

    params = {
        "timestep": "5m",
        "id": item_id,
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()

    # Parse the data
    timestamps = []
    high_prices = []
    low_prices = []
    high_volumes = []
    low_volumes = []

    for entry in data.get("data", []):
        ts = entry.get("timestamp")
        if ts is None:
            continue

        timestamps.append(ts)
        high_prices.append(entry.get("avgHighPrice") or entry.get("avgLowPrice") or 0)
        low_prices.append(entry.get("avgLowPrice") or entry.get("avgHighPrice") or 0)
        high_volumes.append(entry.get("highPriceVolume") or 0)
        low_volumes.append(entry.get("lowPriceVolume") or 0)

    logger.info(f"Fetched {len(timestamps)} data points from API")

    return {
        "timestamps": np.array(timestamps),
        "high_prices": np.array(high_prices, dtype=np.float32),
        "low_prices": np.array(low_prices, dtype=np.float32),
        "high_volumes": np.array(high_volumes, dtype=np.float32),
        "low_volumes": np.array(low_volumes, dtype=np.float32),
    }


def fetch_5min_data_db(item_id: int, months: int = 6) -> Dict:
    """
    Fetch historical 5-minute data from PostgreSQL database.

    Requires SSH tunnel to Ampere server:
        ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128 -N &

    Returns dict with timestamps and OHLC data.
    """
    try:
        from src.db_utils import get_simple_connection
    except ImportError:
        logger.error("Could not import db_utils. Make sure you're in the packages/model directory.")
        raise

    logger.info(f"Fetching {months} months of 5-min data from database for item {item_id}...")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=months * 30)

    conn = get_simple_connection()
    try:
        query = """
            SELECT
                EXTRACT(EPOCH FROM timestamp)::bigint as ts,
                avg_high_price,
                avg_low_price,
                high_price_volume,
                low_price_volume
            FROM price_data_5min
            WHERE item_id = %s
              AND timestamp >= %s
              AND timestamp < %s
            ORDER BY timestamp ASC
        """

        with conn.cursor() as cur:
            cur.execute(query, (item_id, start_date, end_date))
            rows = cur.fetchall()

        logger.info(f"Fetched {len(rows)} data points from database")

        if len(rows) == 0:
            return {
                "timestamps": np.array([]),
                "high_prices": np.array([], dtype=np.float32),
                "low_prices": np.array([], dtype=np.float32),
                "high_volumes": np.array([], dtype=np.float32),
                "low_volumes": np.array([], dtype=np.float32),
            }

        timestamps = np.array([r[0] for r in rows])
        high_prices = np.array([r[1] or 0 for r in rows], dtype=np.float32)
        low_prices = np.array([r[2] or 0 for r in rows], dtype=np.float32)
        high_volumes = np.array([r[3] or 0 for r in rows], dtype=np.float32)
        low_volumes = np.array([r[4] or 0 for r in rows], dtype=np.float32)

        return {
            "timestamps": timestamps,
            "high_prices": high_prices,
            "low_prices": low_prices,
            "high_volumes": high_volumes,
            "low_volumes": low_volumes,
        }

    finally:
        conn.close()


def fetch_5min_data_csv(item_id: int, months: int = 6, csv_path: str = None) -> Dict:
    """
    Fetch historical 5-minute data from local CSV export.

    Args:
        item_id: Item ID to fetch
        months: Number of months of history
        csv_path: Path to CSV file (default: data/hydra_export/price_data_5min.csv)

    Returns:
        Dict with timestamps and OHLC data
    """
    import pandas as pd

    if csv_path is None:
        csv_path = "data/hydra_export/price_data_5min.csv"

    logger.info(f"Loading data from CSV: {csv_path}")
    logger.info(f"Filtering for item {item_id}, last {months} months...")

    # Read CSV in chunks to handle large file
    chunk_size = 1_000_000
    chunks = []

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=months * 30)

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Filter by item_id
        item_chunk = chunk[chunk['item_id'] == item_id]

        if len(item_chunk) == 0:
            continue

        # Parse timestamp
        item_chunk = item_chunk.copy()
        item_chunk['timestamp'] = pd.to_datetime(item_chunk['timestamp'])

        # Filter by date range
        item_chunk = item_chunk[
            (item_chunk['timestamp'] >= start_date) &
            (item_chunk['timestamp'] < end_date)
        ]

        if len(item_chunk) > 0:
            chunks.append(item_chunk)

        # Progress
        if len(chunks) % 10 == 0:
            total_rows = sum(len(c) for c in chunks)
            logger.info(f"  Loaded {total_rows:,} rows so far...")

    if len(chunks) == 0:
        logger.warning(f"No data found for item {item_id}")
        return {
            "timestamps": np.array([]),
            "high_prices": np.array([], dtype=np.float32),
            "low_prices": np.array([], dtype=np.float32),
            "high_volumes": np.array([], dtype=np.float32),
            "low_volumes": np.array([], dtype=np.float32),
        }

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values('timestamp')

    logger.info(f"Loaded {len(df):,} data points from CSV")

    # Convert to unix timestamps
    timestamps = df['timestamp'].astype(np.int64) // 10**9

    return {
        "timestamps": timestamps.values,
        "high_prices": df['avg_high_price'].fillna(0).values.astype(np.float32),
        "low_prices": df['avg_low_price'].fillna(0).values.astype(np.float32),
        "high_volumes": df['high_price_volume'].fillna(0).values.astype(np.float32),
        "low_volumes": df['low_price_volume'].fillna(0).values.astype(np.float32),
    }


def fetch_5min_data(item_id: int, months: int = 6, source: str = "csv", csv_path: str = None) -> Dict:
    """
    Fetch historical 5-minute data.

    Args:
        item_id: Item ID to fetch
        months: Number of months of history
        source: "csv" for local CSV, "db" for database (requires SSH tunnel), "api" for Wiki API
        csv_path: Path to CSV file (for source="csv")

    Returns:
        Dict with timestamps and OHLC data
    """
    if source == "api":
        return fetch_5min_data_api(item_id, months)
    elif source == "csv":
        return fetch_5min_data_csv(item_id, months, csv_path)
    else:
        return fetch_5min_data_db(item_id, months)


# =============================================================================
# FEATURE COMPUTATION (simplified for backtest)
# =============================================================================

def compute_features_at_time(
    data: Dict,
    time_idx: int,
    lookback_5min: int = 288,  # 24 hours
) -> Optional[np.ndarray]:
    """
    Compute features at a specific time index.

    Returns (288, 6) array of recent features or None if not enough history.
    """
    if time_idx < lookback_5min:
        return None

    start_idx = time_idx - lookback_5min

    high = data["high_prices"][start_idx:time_idx]
    low = data["low_prices"][start_idx:time_idx]
    high_vol = data["high_volumes"][start_idx:time_idx]
    low_vol = data["low_volumes"][start_idx:time_idx]

    # Handle zeros
    high = np.where(high > 0, high, np.nan)
    low = np.where(low > 0, low, np.nan)

    # Forward fill NaN
    mask = ~np.isnan(high)
    if not mask.any():
        return None
    idx = np.where(mask, np.arange(len(high)), 0)
    np.maximum.accumulate(idx, out=idx)
    high = high[idx]

    mask = ~np.isnan(low)
    if not mask.any():
        return None
    idx = np.where(mask, np.arange(len(low)), 0)
    np.maximum.accumulate(idx, out=idx)
    low = low[idx]

    # Normalize to current price
    current_high = high[-1]
    current_low = low[-1]
    current_mid = (current_high + current_low) / 2

    if current_mid <= 0:
        return None

    # Features: (high_norm, low_norm, log_high_vol, log_low_vol, hour_sin, hour_cos)
    features = np.zeros((lookback_5min, 6), dtype=np.float32)
    features[:, 0] = (high - current_mid) / current_mid  # Normalized high
    features[:, 1] = (low - current_mid) / current_mid   # Normalized low
    features[:, 2] = np.log1p(high_vol)                  # Log high volume
    features[:, 3] = np.log1p(low_vol)                   # Log low volume

    # Time features (hour of day)
    timestamps = data["timestamps"][start_idx:time_idx]
    hours = np.array([datetime.fromtimestamp(ts, tz=timezone.utc).hour for ts in timestamps])
    features[:, 4] = np.sin(2 * np.pi * hours / 24)
    features[:, 5] = np.cos(2 * np.pi * hours / 24)

    return features


def compute_actual_outcomes(
    data: Dict,
    time_idx: int,
) -> Optional[np.ndarray]:
    """
    Compute actual price movements for each horizon.

    Returns (7, 2) array of [max_high_pct, min_low_pct] per horizon.
    """
    current_high = data["high_prices"][time_idx]
    current_low = data["low_prices"][time_idx]
    current_mid = (current_high + current_low) / 2

    if current_mid <= 0:
        return None

    outcomes = np.zeros((N_HORIZONS, 2), dtype=np.float32)

    for h_idx, hours in enumerate(HORIZONS):
        periods = hours * 12  # 12 periods per hour
        end_idx = min(time_idx + periods, len(data["high_prices"]))

        if end_idx <= time_idx:
            return None

        future_high = data["high_prices"][time_idx:end_idx]
        future_low = data["low_prices"][time_idx:end_idx]

        # Filter zeros
        future_high = future_high[future_high > 0]
        future_low = future_low[future_low > 0]

        if len(future_high) == 0 or len(future_low) == 0:
            return None

        max_high = future_high.max()
        min_low = future_low.min()

        outcomes[h_idx, 0] = (max_high - current_mid) / current_mid
        outcomes[h_idx, 1] = (min_low - current_mid) / current_mid

    return outcomes


def compute_actual_volume(
    data: Dict,
    time_idx: int,
) -> np.ndarray:
    """
    Compute actual volume at each horizon.

    Returns (7, 2) array of [buy_volume, sell_volume] per horizon.
    """
    volumes = np.zeros((N_HORIZONS, 2), dtype=np.float32)

    for h_idx, hours in enumerate(HORIZONS):
        periods = hours * 12
        end_idx = min(time_idx + periods, len(data["high_volumes"]))

        if end_idx > time_idx:
            volumes[h_idx, 0] = data["high_volumes"][time_idx:end_idx].sum()
            volumes[h_idx, 1] = data["low_volumes"][time_idx:end_idx].sum()

    return volumes


# =============================================================================
# SIMPLE PREDICTION MODEL (heuristic for backtest)
# =============================================================================

def generate_predictions(
    features: np.ndarray,
    current_price: float,
) -> Dict[str, np.ndarray]:
    """
    Generate price predictions from features.

    In production this would use the trained PatchTST model.
    For backtest, we use a simpler heuristic based on recent volatility.
    """
    # Extract recent price movements
    recent_high = features[:, 0]  # Normalized high prices
    recent_low = features[:, 1]   # Normalized low prices

    # Compute volatility metrics
    volatility = np.std(recent_high - recent_low)
    trend = np.mean(recent_high[-24:]) - np.mean(recent_high[:24])  # 2-hour trend

    # Generate quantile predictions based on historical patterns
    # More volatile = wider spread, trending = shifted center
    high_quantiles = np.zeros((N_HORIZONS, N_QUANTILES), dtype=np.float32)
    low_quantiles = np.zeros((N_HORIZONS, N_QUANTILES), dtype=np.float32)

    # Base quantile offsets (calibrated to typical Cannonball behavior)
    base_high = np.array([0.005, 0.010, 0.015, 0.025, 0.035])  # p10 to p90
    base_low = np.array([-0.035, -0.025, -0.015, -0.010, -0.005])

    for h_idx, hours in enumerate(HORIZONS):
        # Scale by sqrt(time) - longer horizons = more movement
        time_scale = np.sqrt(hours / 4)  # Normalized to 4h
        vol_scale = 1 + volatility * 5  # More volatile = wider predictions

        high_quantiles[h_idx] = base_high * time_scale * vol_scale + trend * 0.5
        low_quantiles[h_idx] = base_low * time_scale * vol_scale + trend * 0.5

    # Clip to reasonable ranges
    high_quantiles = np.clip(high_quantiles, 0.001, 0.20)
    low_quantiles = np.clip(low_quantiles, -0.20, -0.001)

    return {
        "high_quantiles": high_quantiles,
        "low_quantiles": low_quantiles,
    }


# =============================================================================
# USER PROFILES
# =============================================================================

@dataclass
class UserProfile:
    name: str
    risk_tolerance: float
    max_hold_hours: int
    min_margin_pct: float
    min_fill_prob: float
    capital_gp: int
    max_position_pct: float

    def get_quantile_indices(self) -> Tuple[int, int]:
        """Map risk to quantile indices."""
        if self.risk_tolerance < 0.33:
            return 1, 3  # p30/p70
        elif self.risk_tolerance < 0.66:
            return 2, 2  # p50/p50
        else:
            return 0, 4  # p10/p90


PROFILES = [
    UserProfile(
        name="Conservative",
        risk_tolerance=0.2,
        max_hold_hours=24,
        min_margin_pct=0.03,
        min_fill_prob=0.5,
        capital_gp=100_000_000,
        max_position_pct=0.10,
    ),
    UserProfile(
        name="Moderate",
        risk_tolerance=0.5,
        max_hold_hours=48,
        min_margin_pct=0.02,
        min_fill_prob=0.3,
        capital_gp=100_000_000,
        max_position_pct=0.10,
    ),
    UserProfile(
        name="Aggressive",
        risk_tolerance=0.9,
        max_hold_hours=48,
        min_margin_pct=0.01,
        min_fill_prob=0.2,
        capital_gp=100_000_000,
        max_position_pct=0.10,
    ),
]


# =============================================================================
# TRADE SIMULATION
# =============================================================================

@dataclass
class Trade:
    """A single trade from entry to exit."""
    entry_time: int  # timestamp
    exit_time: Optional[int] = None

    buy_price: float = 0.0
    sell_price: float = 0.0
    quantity: int = 0

    buy_filled: bool = False
    sell_filled: bool = False

    buy_horizon_hours: int = 0
    sell_horizon_hours: int = 0

    actual_buy_price: Optional[float] = None
    actual_sell_price: Optional[float] = None

    profit_gp: float = 0.0
    status: str = "pending"  # pending, active, completed, expired


@dataclass
class Portfolio:
    """Track portfolio state for a user profile."""
    profile: UserProfile
    capital_gp: float
    trades: List[Trade] = field(default_factory=list)
    completed_trades: List[Trade] = field(default_factory=list)

    # Metrics
    total_profit_gp: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    expired_trades: int = 0

    # Track capital over time
    capital_history: List[Tuple[int, float]] = field(default_factory=list)


def estimate_fill_probability(
    price_offset_pct: float,
    horizon_hours: int,
    actual_volume: float,
    side: str,
) -> float:
    """
    Estimate fill probability based on price offset and volume.

    Uses actual historical volume for realistic simulation.
    """
    # Base fill rate from volume (higher volume = easier fill)
    # Assume we want to fill 10% of volume for reliable execution
    volume_factor = 1 - np.exp(-actual_volume / 5000)

    # Time factor
    time_factor = 1 - np.exp(-horizon_hours / 2)

    # Price offset penalty
    price_factor = np.exp(-abs(price_offset_pct) * 7)

    # Side factor (buying below market or selling above is harder)
    if side == 'buy' and price_offset_pct < 0:
        side_factor = 0.9
    elif side == 'sell' and price_offset_pct > 0:
        side_factor = 0.9
    else:
        side_factor = 1.0

    fill_prob = volume_factor * time_factor * price_factor * side_factor
    return np.clip(fill_prob, 0.05, 0.95)


def check_fill(
    target_price: float,
    actual_prices: np.ndarray,
    side: str,
    volume: float,
) -> Tuple[bool, Optional[float]]:
    """
    Check if an order would have filled and at what price.

    Args:
        target_price: Our limit order price
        actual_prices: Array of actual prices during the window
        side: 'buy' or 'sell'
        volume: Total volume during window

    Returns:
        (filled: bool, fill_price: Optional[float])
    """
    if len(actual_prices) == 0 or volume < 100:
        return False, None

    if side == 'buy':
        # Buy fills if price drops to or below our bid
        min_price = actual_prices.min()
        if min_price <= target_price:
            # Fill at our price (limit order)
            return True, target_price
    else:
        # Sell fills if price rises to or above our ask
        max_price = actual_prices.max()
        if max_price >= target_price:
            return True, target_price

    return False, None


def generate_trade_recommendation(
    predictions: Dict[str, np.ndarray],
    current_price: float,
    profile: UserProfile,
    actual_volume: np.ndarray,
) -> Optional[Trade]:
    """
    Generate a trade recommendation for a user profile.
    """
    buy_q, sell_q = profile.get_quantile_indices()

    # Find max horizon index
    max_h = N_HORIZONS - 1
    for i, h in enumerate(HORIZONS):
        if h >= profile.max_hold_hours:
            max_h = i
            break

    best_ev = -np.inf
    best_trade = None

    for buy_idx in range(min(max_h, N_HORIZONS - 1)):
        for sell_idx in range(buy_idx + 1, max_h + 1):
            buy_pct = predictions["low_quantiles"][buy_idx, buy_q]
            sell_pct = predictions["high_quantiles"][sell_idx, sell_q]

            buy_price = current_price * (1 + buy_pct)
            sell_price = current_price * (1 + sell_pct)

            # Account for GE tax
            sell_price_after_tax = sell_price * (1 - GE_TAX_RATE)
            margin = (sell_price_after_tax - buy_price) / buy_price

            if margin < profile.min_margin_pct:
                continue

            # Estimate fill probabilities
            buy_vol = actual_volume[buy_idx, 0]
            sell_vol = actual_volume[sell_idx, 1]

            buy_fill = estimate_fill_probability(buy_pct, HORIZONS[buy_idx], buy_vol, 'buy')
            sell_fill = estimate_fill_probability(sell_pct, HORIZONS[sell_idx], sell_vol, 'sell')
            combined_fill = buy_fill * sell_fill

            if combined_fill < profile.min_fill_prob:
                continue

            # Expected value
            ev = margin * combined_fill / np.sqrt(HORIZONS[sell_idx])

            if ev > best_ev:
                best_ev = ev

                # Position sizing
                max_pos = profile.capital_gp * profile.max_position_pct
                quantity = int(max_pos / buy_price)

                # Volume limit (15% of expected volume)
                vol_limit = int(min(buy_vol, sell_vol) * 0.15)
                quantity = min(quantity, vol_limit)

                if quantity < 1:
                    continue

                best_trade = Trade(
                    entry_time=0,  # Set later
                    buy_price=buy_price,
                    sell_price=sell_price,
                    quantity=quantity,
                    buy_horizon_hours=HORIZONS[buy_idx],
                    sell_horizon_hours=HORIZONS[sell_idx],
                )

    return best_trade


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    data: Dict,
    profiles: List[UserProfile],
    eval_interval: int = 12 * 4,  # Every 4 hours (12 periods/hour * 4)
    min_trade_gap: int = 12 * 2,  # Minimum 2 hours between new trades
) -> Dict[str, Portfolio]:
    """
    Run the full backtest simulation.

    Args:
        data: Historical price data
        profiles: User profiles to simulate
        eval_interval: How often to evaluate new trades (in 5-min periods)
        min_trade_gap: Minimum gap between starting new trades

    Returns:
        Dict mapping profile name to Portfolio results
    """
    n_points = len(data["timestamps"])
    logger.info(f"Running backtest on {n_points} data points...")

    # Initialize portfolios
    portfolios = {
        p.name: Portfolio(profile=p, capital_gp=p.capital_gp)
        for p in profiles
    }

    # Track last trade time per profile
    last_trade_time = {p.name: -min_trade_gap for p in profiles}

    # Walk through time
    lookback = 288  # 24 hours of history needed
    max_future = max(HORIZONS) * 12  # Need future data for outcomes

    trades_evaluated = 0

    for t_idx in range(lookback, n_points - max_future, eval_interval):
        timestamp = data["timestamps"][t_idx]

        # Compute features
        features = compute_features_at_time(data, t_idx)
        if features is None:
            continue

        # Get current price
        current_high = data["high_prices"][t_idx]
        current_low = data["low_prices"][t_idx]
        current_price = (current_high + current_low) / 2

        if current_price <= 0:
            continue

        # Get actual future outcomes and volume
        actual_outcomes = compute_actual_outcomes(data, t_idx)
        actual_volume = compute_actual_volume(data, t_idx)

        if actual_outcomes is None:
            continue

        # Generate predictions
        predictions = generate_predictions(features, current_price)

        trades_evaluated += 1

        # Process each profile
        for profile in profiles:
            portfolio = portfolios[profile.name]

            # Update existing trades
            for trade in portfolio.trades[:]:  # Copy list to allow removal
                if trade.status == "pending":
                    # Check if buy filled
                    buy_periods = trade.buy_horizon_hours * 12
                    buy_end = min(t_idx + buy_periods, n_points)

                    if buy_end > t_idx:
                        buy_prices = data["low_prices"][t_idx:buy_end]
                        buy_vol = data["low_volumes"][t_idx:buy_end].sum()

                        filled, fill_price = check_fill(
                            trade.buy_price, buy_prices, 'buy', buy_vol
                        )

                        if filled:
                            trade.buy_filled = True
                            trade.actual_buy_price = fill_price
                            trade.status = "active"
                        elif buy_end >= t_idx + buy_periods:
                            # Buy window expired
                            trade.status = "expired"
                            portfolio.expired_trades += 1
                            portfolio.trades.remove(trade)
                            portfolio.completed_trades.append(trade)

                elif trade.status == "active":
                    # Check if sell filled
                    sell_periods = trade.sell_horizon_hours * 12
                    sell_end = min(trade.entry_time + sell_periods, n_points)

                    # Find current position in sell window
                    sell_start = trade.entry_time + trade.buy_horizon_hours * 12

                    if t_idx >= sell_start:
                        sell_prices = data["high_prices"][sell_start:min(t_idx + eval_interval, sell_end)]
                        sell_vol = data["high_volumes"][sell_start:min(t_idx + eval_interval, sell_end)].sum()

                        filled, fill_price = check_fill(
                            trade.sell_price, sell_prices, 'sell', sell_vol
                        )

                        if filled:
                            trade.sell_filled = True
                            trade.actual_sell_price = fill_price
                            trade.exit_time = t_idx
                            trade.status = "completed"

                            # Calculate profit (with GE tax)
                            sell_after_tax = fill_price * (1 - GE_TAX_RATE)
                            profit_per = sell_after_tax - trade.actual_buy_price
                            trade.profit_gp = profit_per * trade.quantity

                            portfolio.total_profit_gp += trade.profit_gp
                            portfolio.total_trades += 1

                            if trade.profit_gp > 0:
                                portfolio.winning_trades += 1
                            else:
                                portfolio.losing_trades += 1

                            portfolio.capital_gp += trade.profit_gp
                            portfolio.trades.remove(trade)
                            portfolio.completed_trades.append(trade)

                        elif t_idx >= sell_end:
                            # Sell window expired - force sell at current price
                            trade.sell_filled = True
                            trade.actual_sell_price = current_price
                            trade.exit_time = t_idx
                            trade.status = "completed"

                            sell_after_tax = current_price * (1 - GE_TAX_RATE)
                            profit_per = sell_after_tax - trade.actual_buy_price
                            trade.profit_gp = profit_per * trade.quantity

                            portfolio.total_profit_gp += trade.profit_gp
                            portfolio.total_trades += 1

                            if trade.profit_gp > 0:
                                portfolio.winning_trades += 1
                            else:
                                portfolio.losing_trades += 1

                            portfolio.capital_gp += trade.profit_gp
                            portfolio.trades.remove(trade)
                            portfolio.completed_trades.append(trade)

            # Generate new trade if no active trades and enough time passed
            active_trades = [t for t in portfolio.trades if t.status in ["pending", "active"]]
            time_since_last = t_idx - last_trade_time[profile.name]

            if len(active_trades) == 0 and time_since_last >= min_trade_gap:
                trade = generate_trade_recommendation(
                    predictions, current_price, profile, actual_volume
                )

                if trade is not None:
                    trade.entry_time = t_idx
                    portfolio.trades.append(trade)
                    last_trade_time[profile.name] = t_idx

            # Record capital history
            portfolio.capital_history.append((timestamp, portfolio.capital_gp))

        # Progress logging
        if trades_evaluated % 100 == 0:
            logger.info(f"Evaluated {trades_evaluated} time points...")

    logger.info(f"Backtest complete. Evaluated {trades_evaluated} time points.")

    return portfolios


# =============================================================================
# REPORTING
# =============================================================================

def print_backtest_report(portfolios: Dict[str, Portfolio], item_name: str):
    """Print comprehensive backtest results."""

    print("\n" + "=" * 80)
    print(f"  BACKTEST RESULTS: {item_name}")
    print("=" * 80)

    for name, portfolio in portfolios.items():
        profile = portfolio.profile

        # Calculate metrics
        total_trades = portfolio.total_trades
        win_rate = portfolio.winning_trades / max(total_trades, 1) * 100

        # Average profit per trade
        avg_profit = portfolio.total_profit_gp / max(total_trades, 1)

        # Calculate max drawdown from capital history
        peak = portfolio.profile.capital_gp
        max_drawdown = 0
        for _, capital in portfolio.capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Profit factor
        gross_profit = sum(t.profit_gp for t in portfolio.completed_trades if t.profit_gp > 0)
        gross_loss = abs(sum(t.profit_gp for t in portfolio.completed_trades if t.profit_gp < 0))
        profit_factor = gross_profit / max(gross_loss, 1)

        # ROI
        roi = (portfolio.capital_gp - profile.capital_gp) / profile.capital_gp * 100

        print(f"\n┌{'─' * 76}┐")
        print(f"│  {name.upper():^72}  │")
        print(f"├{'─' * 76}┤")
        print(f"│  Risk: {profile.risk_tolerance:.1f} | Max Hold: {profile.max_hold_hours}h | Min Margin: {profile.min_margin_pct*100:.0f}%  {' '*30}│")
        print(f"├{'─' * 76}┤")
        print(f"│  {'TRADE STATISTICS':<72}  │")
        print(f"│    Total Trades:      {total_trades:>10}                                      │")
        print(f"│    Winning Trades:    {portfolio.winning_trades:>10}  ({win_rate:>5.1f}%)                          │")
        print(f"│    Losing Trades:     {portfolio.losing_trades:>10}                                      │")
        print(f"│    Expired (no fill): {portfolio.expired_trades:>10}                                      │")
        print(f"├{'─' * 76}┤")
        print(f"│  {'PROFITABILITY':<72}  │")
        print(f"│    Starting Capital:  {profile.capital_gp:>14,} GP                              │")
        print(f"│    Ending Capital:    {portfolio.capital_gp:>14,.0f} GP                              │")
        print(f"│    Total Profit/Loss: {portfolio.total_profit_gp:>+14,.0f} GP                              │")
        print(f"│    ROI:               {roi:>13.2f}%                                 │")
        print(f"├{'─' * 76}┤")
        print(f"│  {'RISK METRICS':<72}  │")
        print(f"│    Avg Profit/Trade:  {avg_profit:>+14,.0f} GP                              │")
        print(f"│    Max Drawdown:      {max_drawdown*100:>13.2f}%                                 │")
        print(f"│    Profit Factor:     {profit_factor:>14.2f}                                 │")
        print(f"└{'─' * 76}┘")

    # Comparison table
    print("\n" + "=" * 80)
    print("  PROFILE COMPARISON")
    print("=" * 80)
    print(f"\n{'Profile':<15} {'Trades':>8} {'Win Rate':>10} {'ROI':>10} {'Total P/L':>15} {'Avg P/L':>12}")
    print("-" * 70)

    for name, portfolio in portfolios.items():
        total = portfolio.total_trades
        win_rate = portfolio.winning_trades / max(total, 1) * 100
        roi = (portfolio.capital_gp - portfolio.profile.capital_gp) / portfolio.profile.capital_gp * 100
        avg_pl = portfolio.total_profit_gp / max(total, 1)

        print(f"{name:<15} {total:>8} {win_rate:>9.1f}% {roi:>9.2f}% {portfolio.total_profit_gp:>+14,.0f} {avg_pl:>+11,.0f}")

    print()


def save_results(portfolios: Dict[str, Portfolio], output_path: str):
    """Save backtest results to JSON."""
    results = {}

    for name, portfolio in portfolios.items():
        results[name] = {
            "profile": {
                "risk_tolerance": portfolio.profile.risk_tolerance,
                "max_hold_hours": portfolio.profile.max_hold_hours,
                "min_margin_pct": portfolio.profile.min_margin_pct,
                "capital_gp": portfolio.profile.capital_gp,
            },
            "metrics": {
                "total_trades": portfolio.total_trades,
                "winning_trades": portfolio.winning_trades,
                "losing_trades": portfolio.losing_trades,
                "expired_trades": portfolio.expired_trades,
                "total_profit_gp": portfolio.total_profit_gp,
                "final_capital_gp": portfolio.capital_gp,
                "roi_pct": (portfolio.capital_gp - portfolio.profile.capital_gp) / portfolio.profile.capital_gp * 100,
            },
            "trades": [
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "buy_price": t.buy_price,
                    "sell_price": t.sell_price,
                    "actual_buy_price": t.actual_buy_price,
                    "actual_sell_price": t.actual_sell_price,
                    "quantity": t.quantity,
                    "profit_gp": t.profit_gp,
                    "status": t.status,
                }
                for t in portfolio.completed_trades
            ],
        }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run backtest simulation")
    parser.add_argument("--item-id", type=int, default=2, help="Item ID (default: 2 = Cannonball)")
    parser.add_argument("--months", type=int, default=6, help="Months of history (default: 6)")
    parser.add_argument("--output", default="/tmp/backtest_results.json", help="Output JSON path")
    parser.add_argument("--eval-interval", type=int, default=48, help="Evaluation interval in 5-min periods (default: 48 = 4 hours)")
    parser.add_argument("--source", choices=["csv", "db", "api"], default="csv", help="Data source: csv (local), db (SSH tunnel), api (limited)")
    parser.add_argument("--csv-path", default="data/hydra_export/price_data_5min.csv", help="Path to CSV file")
    args = parser.parse_args()

    item_name = ITEM_INFO.get(args.item_id, (f"Item {args.item_id}", 100))[0]

    # Fetch data
    try:
        data = fetch_5min_data(args.item_id, args.months, source=args.source, csv_path=args.csv_path)
    except Exception as e:
        if args.source == "db":
            logger.error(f"Database connection failed: {e}")
            logger.error("Make sure SSH tunnel is running:")
            logger.error("  ssh -i .secrets/oracle_key.pem -L 5432:localhost:5432 ubuntu@150.136.170.128 -N &")
        elif args.source == "csv":
            logger.error(f"CSV load failed: {e}")
            logger.error(f"Make sure CSV file exists at: {args.csv_path}")
        raise

    if len(data["timestamps"]) < 1000:
        logger.error(f"Not enough data points for backtest ({len(data['timestamps'])} found, need 1000+)")
        if args.source == "api":
            logger.error("Wiki API only provides ~1 day of 5-min data. Use --source csv or --source db for longer backtests.")
        return

    # Run backtest
    portfolios = run_backtest(
        data,
        PROFILES,
        eval_interval=args.eval_interval,
    )

    # Report results
    print_backtest_report(portfolios, item_name)

    # Save results
    save_results(portfolios, args.output)


if __name__ == "__main__":
    main()
