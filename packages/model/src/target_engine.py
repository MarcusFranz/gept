"""
Target Variable Engineering for GE Flipping Predictions

Computes fill probability targets:
- Did a buy order at X% below current low fill within Y hours?
- Did a sell order at X% above current high fill within Y hours?
- Did both buy and sell fill within the window (round-trip)?

This is the critical target for profitable flipping - predicting when
limit orders will actually fill.
"""

import math

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TargetConfig:
    """Configuration for target variable computation."""
    # Offset percentages to try (buy X% below, sell X% above)
    offsets: List[float] = None
    # Time windows in hours
    windows_hours: List[int] = None
    # Tax rate (affects profitability)
    tax_rate: float = 0.02

    def __post_init__(self):
        if self.offsets is None:
            self.offsets = [0.01, 0.015, 0.02, 0.025, 0.03]  # 1% to 3%
        if self.windows_hours is None:
            self.windows_hours = [4, 8, 12, 24, 48]  # 4h to 48h


@dataclass
class DiscreteHourConfig:
    """Configuration for discrete hour target computation."""
    # Offset percentages
    offsets: List[float] = None
    # Discrete hours to predict (1-24 means "fills in hour 1", "fills in hour 2", etc.)
    discrete_hours: List[int] = None
    # Tax rate
    tax_rate: float = 0.02

    def __post_init__(self):
        if self.offsets is None:
            self.offsets = [0.02, 0.025]  # 2% and 2.5%
        if self.discrete_hours is None:
            self.discrete_hours = list(range(1, 25))  # Hours 1-24


class TargetEngine:
    """
    Computes target variables for fill probability prediction.

    The key insight: Profit comes from limit orders filling, not price direction.
    We predict "will my buy order at X% below current low fill within Y hours?"
    """

    def __init__(self, granularity: str = '5m', config: Optional[TargetConfig] = None):
        """
        Initialize target engine.

        Args:
            granularity: '5m' for 5-minute data, '1m' for 1-minute
            config: Target configuration
        """
        self.granularity = granularity
        self.config = config or TargetConfig()

        # Calculate periods per hour
        if granularity == '5m':
            self.periods_per_hour = 12
        elif granularity == '1m':
            self.periods_per_hour = 60
        else:
            self.periods_per_hour = 1

        # Convert windows to periods
        self.window_periods = {
            h: h * self.periods_per_hour for h in self.config.windows_hours
        }

    def compute_targets(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Compute all target variables.

        Args:
            df: DataFrame with high/low price columns
            copy: Whether to copy input

        Returns:
            DataFrame with original columns plus target columns
        """
        if copy:
            df = df.copy()

        # Ensure sorted
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Get price columns
        high_col = 'avg_high_price' if 'avg_high_price' in df.columns else 'high_price'
        low_col = 'avg_low_price' if 'avg_low_price' in df.columns else 'low_price'

        # Forward fill NaN prices for computation
        df['_high'] = df[high_col].ffill()
        df['_low'] = df[low_col].ffill()

        # Compute fill targets for each offset and window
        for offset in self.config.offsets:
            for hours, periods in self.window_periods.items():
                df = self._compute_fill_targets(df, offset, periods, hours)

        # Compute direction target (secondary)
        df = self._compute_direction_targets(df)

        # Clean up temp columns
        df = df.drop(columns=['_high', '_low'], errors='ignore')

        return df

    def _compute_fill_targets(self, df: pd.DataFrame, offset: float,
                              periods: int, hours: int) -> pd.DataFrame:
        """
        Compute fill probability targets for a specific offset and window.

        A buy fills if future_min_low <= buy_target
        A sell fills if future_max_high >= sell_target
        """
        offset_str = f"{int(offset*100)}pct"  # e.g., "2pct"

        # Buy target: current low minus offset
        buy_target = df['_low'] * (1 - offset)

        # Sell target: current high plus offset
        sell_target = df['_high'] * (1 + offset)

        # Future min low within window (shift to look forward)
        # We use negative shift to look into the future
        future_min_low = df['_low'].rolling(periods, min_periods=1).min().shift(-periods)

        # Future max high within window
        future_max_high = df['_high'].rolling(periods, min_periods=1).max().shift(-periods)

        # Fill conditions
        buy_fills = (future_min_low <= buy_target).astype(int)
        sell_fills = (future_max_high >= sell_target).astype(int)
        both_fill = (buy_fills & sell_fills).astype(int)

        # Store targets
        df[f'buy_fills_{offset_str}_{hours}h'] = buy_fills
        df[f'sell_fills_{offset_str}_{hours}h'] = sell_fills
        df[f'roundtrip_{offset_str}_{hours}h'] = both_fill

        # Compute profit if round-trip fills
        # Gross profit = offset on buy + offset on sell = 2*offset
        # Net profit = gross - tax (1% on buy, 1% on sell = 2%)
        net_profit_pct = 2 * offset - self.config.tax_rate
        df[f'profit_{offset_str}_{hours}h'] = both_fill * net_profit_pct

        return df

    def _compute_direction_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price direction targets (secondary targets)."""
        mid = (df['_high'] + df['_low']) / 2

        # Direction at various horizons
        for hours in [4, 8, 24, 48]:
            periods = hours * self.periods_per_hour
            future_mid = mid.shift(-periods)
            df[f'direction_{hours}h'] = (future_mid > mid).astype(int)

            # Price change magnitude
            df[f'price_change_{hours}h'] = (future_mid - mid) / mid

        return df

    def get_target_names(self) -> List[str]:
        """Return list of all target column names."""
        targets = []

        for offset in self.config.offsets:
            offset_str = f"{int(offset*100)}pct"
            for hours in self.config.windows_hours:
                targets.extend([
                    f'buy_fills_{offset_str}_{hours}h',
                    f'sell_fills_{offset_str}_{hours}h',
                    f'roundtrip_{offset_str}_{hours}h',
                    f'profit_{offset_str}_{hours}h',
                ])

        for hours in [4, 8, 24, 48]:
            targets.extend([
                f'direction_{hours}h',
                f'price_change_{hours}h',
            ])

        return targets

    def get_primary_targets(self) -> List[str]:
        """Return list of primary targets for modeling (roundtrip fills)."""
        targets = []
        for offset in self.config.offsets:
            offset_str = f"{int(offset*100)}pct"
            for hours in self.config.windows_hours:
                targets.append(f'roundtrip_{offset_str}_{hours}h')
        return targets

    def analyze_target_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze base rates for all targets.

        Returns dict mapping target name to fill rate.
        """
        rates = {}
        for target in self.get_target_names():
            if target in df.columns:
                # Only count non-NaN rows
                valid = df[target].dropna()
                if len(valid) > 0:
                    rates[target] = valid.mean()
        return rates


class DiscreteHourTargetEngine:
    """
    Computes discrete hour target variables for fill probability prediction.

    Unlike TargetEngine which computes "did fill WITHIN X hours" (cumulative),
    this computes "did fill IN hour X" (discrete).

    Example:
    - roundtrip_2pct_hour3 = filled specifically in hour 3 (not before)
    - This means: filled by hour 3 AND NOT filled by hour 2
    """

    def __init__(self, granularity: str = '5m', config: Optional[DiscreteHourConfig] = None):
        """
        Initialize discrete hour target engine.

        Args:
            granularity: '5m' for 5-minute data, '1m' for 1-minute
            config: Discrete hour configuration
        """
        self.granularity = granularity
        self.config = config or DiscreteHourConfig()

        # Calculate periods per hour
        if granularity == '5m':
            self.periods_per_hour = 12
        elif granularity == '1m':
            self.periods_per_hour = 60
        else:
            self.periods_per_hour = 1

    def compute_targets(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Compute discrete hour targets for all offsets and hours.

        Args:
            df: DataFrame with high/low price columns
            copy: Whether to copy input

        Returns:
            DataFrame with original columns plus discrete hour target columns
        """
        if copy:
            df = df.copy()

        # Ensure sorted
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Get price columns
        high_col = 'avg_high_price' if 'avg_high_price' in df.columns else 'high_price'
        low_col = 'avg_low_price' if 'avg_low_price' in df.columns else 'low_price'

        # Forward fill NaN prices for computation
        df['_high'] = df[high_col].ffill()
        df['_low'] = df[low_col].ffill()

        # Compute targets for each offset
        for offset in self.config.offsets:
            df = self._compute_cumulative_fills(df, offset)
            df = self._compute_discrete_hour_targets(df, offset)

        # Clean up temp columns
        cols_to_drop = ['_high', '_low'] + [c for c in df.columns if c.startswith('_cum_')]
        df = df.drop(columns=cols_to_drop, errors='ignore')

        return df

    def _compute_cumulative_fills(self, df: pd.DataFrame, offset: float) -> pd.DataFrame:
        """
        Compute cumulative fill targets for each hour.

        These are intermediate targets used to derive discrete hour targets.
        Uses vectorized operations for performance.
        """
        offset_str = self._offset_str(offset)

        # Buy target: current low minus offset
        buy_target = df['_low'].values * (1 - offset)
        # Sell target: current high plus offset
        sell_target = df['_high'].values * (1 + offset)

        max_hour = max(self.config.discrete_hours)

        # Pre-compute all columns then concat at once to avoid fragmentation
        new_cols = {}

        for hour in range(1, max_hour + 1):
            periods = hour * self.periods_per_hour

            # Future min low within window (cumulative by hour)
            future_min_low = df['_low'].rolling(periods, min_periods=1).min().shift(-periods).values
            # Future max high within window
            future_max_high = df['_high'].rolling(periods, min_periods=1).max().shift(-periods).values

            # Cumulative fill: did it fill BY this hour?
            buy_fills_by_hour = (future_min_low <= buy_target).astype(int)
            sell_fills_by_hour = (future_max_high >= sell_target).astype(int)
            roundtrip_by_hour = (buy_fills_by_hour & sell_fills_by_hour).astype(int)

            # Store in dict
            new_cols[f'_cum_buy_{offset_str}_h{hour}'] = buy_fills_by_hour
            new_cols[f'_cum_sell_{offset_str}_h{hour}'] = sell_fills_by_hour
            new_cols[f'_cum_rt_{offset_str}_h{hour}'] = roundtrip_by_hour

        # Concat all new columns at once
        new_df = pd.DataFrame(new_cols, index=df.index)
        return pd.concat([df, new_df], axis=1)

    def _compute_discrete_hour_targets(self, df: pd.DataFrame, offset: float) -> pd.DataFrame:
        """
        Compute discrete hour targets from cumulative fills.

        fills_in_hour_x = fills_by_hour_x AND NOT fills_by_hour_(x-1)
        """
        offset_str = self._offset_str(offset)

        # Pre-compute all discrete targets
        new_cols = {}

        for hour in self.config.discrete_hours:
            cum_rt_this = df[f'_cum_rt_{offset_str}_h{hour}'].values

            if hour == 1:
                # First hour: discrete = cumulative (nothing to subtract)
                discrete_rt = cum_rt_this
            else:
                cum_rt_prev = df[f'_cum_rt_{offset_str}_h{hour - 1}'].values
                # Discrete: filled this hour but NOT in previous hours
                discrete_rt = ((cum_rt_this == 1) & (cum_rt_prev == 0)).astype(int)

            new_cols[f'roundtrip_{offset_str}_hour{hour}'] = discrete_rt

        # Concat all at once
        new_df = pd.DataFrame(new_cols, index=df.index)
        return pd.concat([df, new_df], axis=1)

    def _offset_str(self, offset: float) -> str:
        """Convert offset to string like '2pct' or '2.5pct'."""
        if offset * 100 == int(offset * 100):
            return f"{int(offset * 100)}pct"
        else:
            return f"{offset * 100:.1f}pct".replace('.', '_')

    def get_target_names(self) -> List[str]:
        """Return list of all discrete hour target column names."""
        targets = []
        for offset in self.config.offsets:
            offset_str = self._offset_str(offset)
            for hour in self.config.discrete_hours:
                targets.append(f'roundtrip_{offset_str}_hour{hour}')
        return targets

    def get_target_for_offset_hour(self, offset: float, hour: int) -> str:
        """Get target column name for specific offset and hour."""
        offset_str = self._offset_str(offset)
        return f'roundtrip_{offset_str}_hour{hour}'

    def analyze_target_rates(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze fill rates for all discrete hour targets."""
        rates = {}
        for target in self.get_target_names():
            if target in df.columns:
                valid = df[target].dropna()
                if len(valid) > 0:
                    rates[target] = valid.mean()
        return rates

    def validate_discrete_targets(self, df: pd.DataFrame) -> Dict:
        """
        Validate discrete hour targets.

        Checks:
        1. Sum of discrete hours <= cumulative 24h (should be close but not exceed)
        2. Later hours should have lower fill rates (generally)
        3. No NaN values in computed targets
        """
        results = {
            'is_valid': True,
            'issues': [],
            'hourly_rates': {}
        }

        for offset in self.config.offsets:
            offset_str = self._offset_str(offset)

            # Check each hour has valid data
            for hour in self.config.discrete_hours:
                target = f'roundtrip_{offset_str}_hour{hour}'
                if target not in df.columns:
                    results['issues'].append(f"Missing target: {target}")
                    results['is_valid'] = False
                else:
                    rate = df[target].dropna().mean()
                    results['hourly_rates'][target] = rate

            # Check sum of discrete hours approximates cumulative
            total_discrete = sum(
                df[f'roundtrip_{offset_str}_hour{h}'].dropna().mean()
                for h in self.config.discrete_hours
                if f'roundtrip_{offset_str}_hour{h}' in df.columns
            )
            results[f'total_discrete_{offset_str}'] = total_discrete

        return results


class TargetValidator:
    """Validates target computation for sanity."""

    @staticmethod
    def validate_targets(df: pd.DataFrame, engine: TargetEngine) -> Dict:
        """
        Validate computed targets.

        Returns validation results dict.
        """
        results = {
            'is_valid': True,
            'issues': [],
            'target_rates': {}
        }

        target_names = engine.get_target_names()

        # Check all targets exist
        missing = [t for t in target_names if t not in df.columns]
        if missing:
            results['issues'].append(f"Missing targets: {missing[:5]}...")
            results['is_valid'] = False
            return results

        # Analyze rates
        rates = engine.analyze_target_rates(df)
        results['target_rates'] = rates

        # Check for suspicious rates
        for target, rate in rates.items():
            if 'roundtrip' in target:
                # Round-trip fill rates should generally be 10-70%
                if rate < 0.01:
                    results['issues'].append(f"{target}: suspiciously low rate ({rate:.2%})")
                elif rate > 0.95:
                    results['issues'].append(f"{target}: suspiciously high rate ({rate:.2%})")

        # Check monotonicity: larger offsets should fill less often
        for hours in engine.config.windows_hours:
            prev_rate = None
            for offset in sorted(engine.config.offsets):
                offset_str = f"{int(offset*100)}pct"
                target = f'roundtrip_{offset_str}_{hours}h'
                rate = rates.get(target, 0)

                if prev_rate is not None and rate > prev_rate * 1.1:  # Allow 10% tolerance
                    results['issues'].append(
                        f"Non-monotonic: {target} ({rate:.2%}) > previous offset ({prev_rate:.2%})"
                    )
                prev_rate = rate

        # Check monotonicity: longer windows should fill more often
        for offset in engine.config.offsets:
            offset_str = f"{int(offset*100)}pct"
            prev_rate = None
            for hours in sorted(engine.config.windows_hours):
                target = f'roundtrip_{offset_str}_{hours}h'
                rate = rates.get(target, 0)

                if prev_rate is not None and rate < prev_rate * 0.9:  # Allow 10% tolerance
                    results['issues'].append(
                        f"Non-monotonic: {target} ({rate:.2%}) < shorter window ({prev_rate:.2%})"
                    )
                prev_rate = rate

        if results['issues']:
            results['is_valid'] = False

        return results


def compute_ge_tax(sale_price: float) -> int:
    """
    Compute OSRS Grand Exchange tax for a sale.

    OSRS GE tax rules:
    - 2% of sale price, rounded UP to nearest GP
    - Minimum: 1 GP
    - Maximum: 5,000,000 GP

    Args:
        sale_price: The price at which the item is sold (in GP)

    Returns:
        Tax amount in GP (integer)
    """
    if pd.isna(sale_price) or sale_price <= 0:
        return 0

    raw_tax = sale_price * 0.02
    tax = math.ceil(raw_tax)

    # Enforce min/max bounds
    tax = max(1, tax)  # Minimum 1 GP
    tax = min(5_000_000, tax)  # Maximum 5M GP cap

    return tax


def compute_expected_value_gp(
    fill_prob: float,
    buy_price: float,
    sell_price: float
) -> float:
    """
    Compute expected value of a trade with accurate GP-based tax calculation.

    Uses the actual OSRS GE tax rules:
    - Tax is 2% of sale price (not profit)
    - Tax has min 1 GP and max 5M GP

    Args:
        fill_prob: Probability that round-trip fills (0-1)
        buy_price: The price we pay to buy the item (GP)
        sell_price: The price we receive when selling (GP)

    Returns:
        Expected value in GP (absolute, not percentage)
    """
    if pd.isna(buy_price) or pd.isna(sell_price) or buy_price <= 0 or sell_price <= 0:
        return 0.0

    gross_profit = sell_price - buy_price
    tax = compute_ge_tax(sell_price)
    net_profit = gross_profit - tax

    # EV = P(fill) * profit + P(no_fill) * 0
    return fill_prob * net_profit


def compute_expected_value_pct(
    fill_prob: float,
    buy_price: float,
    sell_price: float
) -> float:
    """
    Compute expected value as a percentage of capital invested.

    Args:
        fill_prob: Probability that round-trip fills (0-1)
        buy_price: The price we pay to buy the item (GP)
        sell_price: The price we receive when selling (GP)

    Returns:
        Expected value as decimal (e.g., 0.02 = 2% return on capital)
    """
    if pd.isna(buy_price) or buy_price <= 0:
        return 0.0

    ev_gp = compute_expected_value_gp(fill_prob, buy_price, sell_price)
    return ev_gp / buy_price


def compute_expected_value(
    fill_prob: float,
    offset: float,
    tax_rate: float = 0.02,
    *,
    current_high: float = None,
    current_low: float = None
) -> float:
    """
    Compute expected value of a trade.

    If current_high and current_low are provided, uses accurate GP-based
    tax calculation with OSRS GE rules (2% rounded up, 1 GP min, 5M max).
    Otherwise falls back to percentage approximation for backward compatibility.

    Args:
        fill_prob: Probability that round-trip fills
        offset: Price offset used (e.g., 0.02 for 2%)
        tax_rate: GE tax rate (ignored if prices provided)
        current_high: Current high price (for accurate calculation)
        current_low: Current low price (for accurate calculation)

    Returns:
        Expected value as decimal (e.g., 0.02 = 2% EV)
    """
    # If prices provided, use accurate calculation
    if current_high is not None and current_low is not None:
        buy_price = current_low * (1 - offset)
        sell_price = current_high * (1 + offset)
        return compute_expected_value_pct(fill_prob, buy_price, sell_price)

    # Fallback: percentage-based approximation (original behavior)
    gross_profit = 2 * offset  # Buy at -offset, sell at +offset
    net_profit = gross_profit - tax_rate

    # EV = P(fill) * profit + P(no_fill) * 0
    return fill_prob * net_profit


def find_optimal_offset(fill_probs: Dict[float, float], tax_rate: float = 0.02) -> Tuple[float, float]:
    """
    Find the offset that maximizes expected value.

    Args:
        fill_probs: Dict mapping offset to fill probability
        tax_rate: GE tax rate

    Returns:
        Tuple of (optimal_offset, expected_value)
    """
    best_offset = 0
    best_ev = 0

    for offset, prob in fill_probs.items():
        ev = compute_expected_value(prob, offset, tax_rate)
        if ev > best_ev:
            best_ev = ev
            best_offset = offset

    return best_offset, best_ev


if __name__ == "__main__":
    # Test with dummy data
    print("Testing TargetEngine...")

    np.random.seed(42)
    n = 2000

    # Create realistic price data with some volatility
    base_price = 1000
    returns = np.random.randn(n) * 0.002  # 0.2% per period
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'avg_high_price': prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        'avg_low_price': prices * (1 - np.abs(np.random.randn(n) * 0.005)),
    })

    # Compute targets
    engine = TargetEngine(granularity='5m')
    result = engine.compute_targets(df)

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Target columns: {len(engine.get_target_names())}")

    # Analyze rates
    print("\nTarget fill rates:")
    rates = engine.analyze_target_rates(result)
    for target in engine.get_primary_targets()[:10]:
        print(f"  {target}: {rates.get(target, 0):.1%}")

    # Validate
    validator = TargetValidator()
    validation = validator.validate_targets(result, engine)
    print(f"\nValidation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues'][:5]:
            print(f"  - {issue}")

    # Test EV calculation
    print("\nExpected Value Analysis (24h window):")
    for offset in [0.01, 0.015, 0.02, 0.025, 0.03]:
        offset_str = f"{int(offset*100)}pct"
        target = f'roundtrip_{offset_str}_24h'
        prob = rates.get(target, 0)
        ev = compute_expected_value(prob, offset)
        print(f"  {offset_str}: fill rate={prob:.1%}, EV={ev*100:.2f}%")
