"""
Feature Engineering Pipeline for GE Flipping Predictions

Computes features from price data at both 5-min (training) and 1-min (inference) granularities.
All features are designed to capture fill probability signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Granularity(Enum):
    """Data granularity options."""
    FIVE_MIN = '5m'
    ONE_MIN = '1m'
    HOURLY = '1h'


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""
    # Moving average windows (in hours)
    ma_hours: List[int] = None
    # Return lookback periods (in hours)
    return_hours: List[float] = None
    # Volatility windows (in hours)
    volatility_hours: List[int] = None
    # Range windows for fill prediction (in hours)
    range_hours: List[int] = None

    def __post_init__(self):
        if self.ma_hours is None:
            self.ma_hours = [1, 4, 8, 24, 48, 168]  # 1h, 4h, 8h, 24h, 48h, 1week
        if self.return_hours is None:
            self.return_hours = [0.25, 0.5, 1, 2, 4, 8, 24]
        if self.volatility_hours is None:
            self.volatility_hours = [1, 4, 24]
        if self.range_hours is None:
            self.range_hours = [1, 4, 8, 12, 24, 48]


class FeatureEngine:
    """
    Computes features from price data for GE flipping predictions.

    Supports both 5-minute (training) and 1-minute (inference) data.
    Window sizes are automatically adjusted based on granularity.
    """

    def __init__(self, granularity: Granularity = Granularity.FIVE_MIN,
                 config: Optional[FeatureConfig] = None):
        """
        Initialize the feature engine.

        Args:
            granularity: Data granularity (5m for training, 1m for inference)
            config: Feature configuration (uses defaults if None)
        """
        self.granularity = granularity
        self.config = config or FeatureConfig()
        self._setup_windows()

    def _setup_windows(self):
        """Convert hour-based windows to period counts based on granularity."""
        if self.granularity == Granularity.FIVE_MIN:
            periods_per_hour = 12  # 60/5 = 12
        elif self.granularity == Granularity.ONE_MIN:
            periods_per_hour = 60
        else:  # hourly
            periods_per_hour = 1

        # Convert all windows to periods
        self.ma_windows = {h: int(h * periods_per_hour) for h in self.config.ma_hours}
        self.return_periods = {h: max(1, int(h * periods_per_hour)) for h in self.config.return_hours}
        self.volatility_windows = {h: int(h * periods_per_hour) for h in self.config.volatility_hours}
        self.range_windows = {h: int(h * periods_per_hour) for h in self.config.range_hours}

        # Minimum periods needed to compute all features
        self.min_history = max(
            max(self.ma_windows.values()),
            max(self.return_periods.values()),
            max(self.volatility_windows.values()),
            max(self.range_windows.values())
        )

    def compute_features(self, df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
        """
        Compute all features from price data.

        Args:
            df: DataFrame with columns: timestamp, avg_high_price, avg_low_price,
                high_price_volume, low_price_volume
            copy: Whether to copy input DataFrame

        Returns:
            DataFrame with original columns plus computed features
        """
        if copy:
            df = df.copy()

        # Ensure sorted by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Collect all new columns in a dictionary to avoid DataFrame fragmentation
        new_cols = {}

        # Compute derived price columns (these are needed for subsequent calculations)
        df, base_cols = self._compute_base_prices(df)
        new_cols.update(base_cols)

        # Compute feature groups - each returns a dict of new columns
        new_cols.update(self._compute_spread_features(df, new_cols))
        new_cols.update(self._compute_ma_features(df))
        new_cols.update(self._compute_return_features(df))
        new_cols.update(self._compute_volatility_features(df, new_cols))
        new_cols.update(self._compute_volume_features(df))
        new_cols.update(self._compute_range_features(df))
        new_cols.update(self._compute_time_features(df))
        new_cols.update(self._compute_momentum_features(df, new_cols))

        # Create all new columns at once to avoid fragmentation
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=df.index)
            df = pd.concat([df, new_df], axis=1)

        return df

    def _compute_base_prices(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Compute base price columns. Returns (df, new_cols_dict).

        Note: Base prices are added directly to df since they're needed by other methods.
        """
        cols = {}

        # Handle column name variations
        high_col = 'avg_high_price' if 'avg_high_price' in df.columns else 'high_price'
        low_col = 'avg_low_price' if 'avg_low_price' in df.columns else 'low_price'

        high = df[high_col].ffill()  # Forward fill NaN prices
        low = df[low_col].ffill()

        # Mid price
        mid = (high + low) / 2

        # Handle edge cases where both are NaN
        mid = mid.ffill().bfill()
        high = high.fillna(mid)
        low = low.fillna(mid)

        # Add to df for use by other methods, and return as cols dict
        df['high'] = high
        df['low'] = low
        df['mid'] = mid

        cols['high'] = high
        cols['low'] = low
        cols['mid'] = mid

        return df, cols

    def _compute_spread_features(self, df: pd.DataFrame, existing_cols: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """Compute spread-related features. Returns dict of new columns."""
        cols = {}

        # Basic spread
        cols['spread'] = df['high'] - df['low']
        cols['spread_pct'] = cols['spread'] / df['mid'].replace(0, np.nan)

        # Spread moving averages and ratios
        for hours, window in self.ma_windows.items():
            if window > 0:
                cols[f'spread_ma_{hours}h'] = cols['spread_pct'].rolling(window, min_periods=1).mean()

        # Spread ratio vs recent average (24h default)
        if 24 in self.ma_windows and self.ma_windows[24] > 0:
            cols['spread_ratio'] = cols['spread_pct'] / cols['spread_ma_24h'].replace(0, np.nan)
        elif self.ma_windows:
            # Use largest window as default
            max_h = max(self.ma_windows.keys())
            cols['spread_ratio'] = cols['spread_pct'] / cols[f'spread_ma_{max_h}h'].replace(0, np.nan)

        return cols

    def _compute_ma_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute moving average features. Returns dict of new columns."""
        cols = {}

        for hours, window in self.ma_windows.items():
            if window > 0:
                # Moving averages
                cols[f'mid_ma_{hours}h'] = df['mid'].rolling(window, min_periods=1).mean()
                cols[f'high_ma_{hours}h'] = df['high'].rolling(window, min_periods=1).mean()
                cols[f'low_ma_{hours}h'] = df['low'].rolling(window, min_periods=1).mean()

                # Price vs MA ratio (mean reversion signal)
                cols[f'mid_ma_ratio_{hours}h'] = df['mid'] / cols[f'mid_ma_{hours}h'].replace(0, np.nan)

        return cols

    def _compute_return_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute return features at various lookbacks. Returns dict of new columns."""
        cols = {}

        for hours, periods in self.return_periods.items():
            if periods > 0:
                # Returns
                cols[f'return_{hours}h'] = df['mid'].pct_change(periods)
                cols[f'high_return_{hours}h'] = df['high'].pct_change(periods)
                cols[f'low_return_{hours}h'] = df['low'].pct_change(periods)

        return cols

    def _compute_volatility_features(self, df: pd.DataFrame, existing_cols: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """Compute volatility features. Returns dict of new columns."""
        cols = {}
        existing_cols = existing_cols or {}

        # Get return_1h from existing cols or compute it
        if 'return_1h' in existing_cols:
            return_1h = existing_cols['return_1h']
        else:
            return_1h = df['mid'].pct_change(self.return_periods.get(1, 12))
            cols['return_1h'] = return_1h

        for hours, window in self.volatility_windows.items():
            if window > 1:
                # Standard deviation of returns
                cols[f'volatility_{hours}h'] = return_1h.rolling(window, min_periods=2).std()

                # Parkinson volatility (uses high/low range)
                log_range = np.log(df['high'] / df['low'].replace(0, np.nan))
                cols[f'parkinson_vol_{hours}h'] = np.sqrt(
                    log_range.pow(2).rolling(window, min_periods=2).mean() / (4 * np.log(2))
                )

        return cols

    def _compute_volume_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute volume features. Returns dict of new columns."""
        cols = {}

        # Total volume
        high_vol_col = 'high_price_volume' if 'high_price_volume' in df.columns else None
        low_vol_col = 'low_price_volume' if 'low_price_volume' in df.columns else None

        if high_vol_col and low_vol_col:
            volume = df[high_vol_col].fillna(0) + df[low_vol_col].fillna(0)
        elif 'volume' in df.columns:
            volume = df['volume']
        else:
            # No volume data available
            volume = pd.Series(0, index=df.index)

        cols['volume'] = volume

        # Volume moving averages
        for hours, window in self.ma_windows.items():
            if window > 0:
                cols[f'volume_ma_{hours}h'] = volume.rolling(window, min_periods=1).mean()

        # Volume ratio vs 24h average
        if 24 in self.ma_windows and self.ma_windows[24] > 0:
            cols['volume_ratio'] = volume / cols['volume_ma_24h'].replace(0, np.nan)

        # Log volume (for modeling)
        cols['log_volume'] = np.log1p(volume)

        return cols

    def _compute_range_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute price range features - critical for fill probability. Returns dict of new columns."""
        cols = {}

        for hours, window in self.range_windows.items():
            if window > 0:
                # Rolling high/low range
                rolling_high = df['high'].rolling(window, min_periods=1).max()
                rolling_low = df['low'].rolling(window, min_periods=1).min()
                rolling_mid = (rolling_high + rolling_low) / 2

                # Range as percentage of mid
                cols[f'range_{hours}h'] = (rolling_high - rolling_low) / rolling_mid.replace(0, np.nan)

                # Current position in range
                cols[f'range_position_{hours}h'] = (df['mid'] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

                # Distance from extremes
                cols[f'dist_from_high_{hours}h'] = (rolling_high - df['high']) / rolling_mid.replace(0, np.nan)
                cols[f'dist_from_low_{hours}h'] = (df['low'] - rolling_low) / rolling_mid.replace(0, np.nan)

        return cols

    def _compute_time_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Compute time-based features. Returns dict of new columns."""
        cols = {}

        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])

            # Hour of day (cyclical encoding)
            hour = ts.dt.hour
            cols['hour'] = hour
            cols['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            cols['hour_cos'] = np.cos(2 * np.pi * hour / 24)

            # Day of week (cyclical encoding)
            day_of_week = ts.dt.dayofweek
            cols['day_of_week'] = day_of_week
            cols['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            cols['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)

            # Binary flags
            cols['is_weekend'] = (day_of_week >= 5).astype(int)

            # Time of day bins (for activity patterns)
            cols['is_peak_hours'] = ((hour >= 14) & (hour <= 22)).astype(int)  # UK evening / US day

        return cols

    def _compute_momentum_features(self, df: pd.DataFrame, existing_cols: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """Compute momentum and divergence features. Returns dict of new columns."""
        cols = {}
        existing_cols = existing_cols or {}

        # High/low momentum divergence (indicates range expansion/contraction)
        for hours in [1, 4, 8]:
            periods = self.return_periods.get(hours, hours * 12)
            if periods > 0:
                high_mom = df['high'].pct_change(periods)
                low_mom = df['low'].pct_change(periods)
                cols[f'momentum_divergence_{hours}h'] = high_mom - low_mom

        # RSI-like indicator - get return_1h from existing cols
        return_1h = existing_cols.get('return_1h')
        if return_1h is not None:
            returns = return_1h.fillna(0)
            gains = returns.where(returns > 0, 0)
            losses = (-returns).where(returns < 0, 0)

            for hours, window in [(4, self.ma_windows.get(4, 48)), (24, self.ma_windows.get(24, 288))]:
                if window and window > 0:
                    avg_gain = gains.rolling(window, min_periods=1).mean()
                    avg_loss = losses.rolling(window, min_periods=1).mean()
                    rs = avg_gain / avg_loss.replace(0, np.nan)
                    cols[f'rsi_{hours}h'] = 100 - (100 / (1 + rs))

        return cols

    def get_feature_names(self) -> List[str]:
        """Return list of all computed feature names."""
        # Create a small dummy DataFrame to get column names
        dummy_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=500, freq='5min'),
            'avg_high_price': np.random.uniform(100, 110, 500),
            'avg_low_price': np.random.uniform(90, 100, 500),
            'high_price_volume': np.random.randint(100, 1000, 500),
            'low_price_volume': np.random.randint(100, 1000, 500),
        })
        result = self.compute_features(dummy_df)

        # Exclude non-feature columns
        exclude = {'timestamp', 'avg_high_price', 'avg_low_price',
                   'high_price_volume', 'low_price_volume', 'item_id'}
        return [c for c in result.columns if c not in exclude]

    def get_feature_columns(self) -> List[str]:
        """Return list of columns to use as model inputs (excludes derived columns like mid, high, low)."""
        all_features = self.get_feature_names()

        # Exclude intermediate computation columns
        exclude = {'high', 'low', 'mid', 'spread', 'volume', 'hour', 'day_of_week'}
        return [c for c in all_features if c not in exclude]


class FeatureValidator:
    """Validates computed features for quality issues."""

    @staticmethod
    def validate(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, any]:
        """
        Validate feature DataFrame for issues.

        Returns dict with validation results.
        """
        results = {
            'total_rows': len(df),
            'feature_count': len(feature_cols),
            'issues': []
        }

        for col in feature_cols:
            if col not in df.columns:
                results['issues'].append(f"Missing column: {col}")
                continue

            col_data = df[col]

            # Check for NaN
            nan_pct = col_data.isna().mean()
            if nan_pct > 0.5:
                results['issues'].append(f"{col}: {nan_pct*100:.1f}% NaN values")

            # Check for infinity
            inf_pct = np.isinf(col_data.replace([np.inf, -np.inf], np.nan).dropna()).mean() if col_data.dtype in [np.float64, np.float32] else 0
            if inf_pct > 0:
                results['issues'].append(f"{col}: contains infinity values")

            # Check for constant values (low variance)
            if col_data.std() < 1e-10 and col_data.notna().sum() > 10:
                results['issues'].append(f"{col}: constant/zero variance")

        results['is_valid'] = len(results['issues']) == 0
        return results


# Convenience function for quick feature computation
def compute_features(df: pd.DataFrame,
                     granularity: str = '5m') -> pd.DataFrame:
    """
    Convenience function to compute features.

    Args:
        df: Price DataFrame
        granularity: '5m' or '1m'

    Returns:
        DataFrame with computed features
    """
    gran = Granularity.FIVE_MIN if granularity == '5m' else Granularity.ONE_MIN
    engine = FeatureEngine(granularity=gran)
    return engine.compute_features(df)


if __name__ == "__main__":
    # Test with dummy data
    print("Testing FeatureEngine...")

    # Create dummy data
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'avg_high_price': 100 + np.cumsum(np.random.randn(n) * 0.1),
        'avg_low_price': 98 + np.cumsum(np.random.randn(n) * 0.1),
        'high_price_volume': np.random.randint(100, 1000, n),
        'low_price_volume': np.random.randint(100, 1000, n),
    })
    df['avg_low_price'] = df[['avg_high_price', 'avg_low_price']].min(axis=1)

    # Test feature computation
    engine = FeatureEngine(granularity=Granularity.FIVE_MIN)
    result = engine.compute_features(df)

    print(f"Input shape: {df.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Feature columns: {len(engine.get_feature_columns())}")

    # Validate
    validator = FeatureValidator()
    validation = validator.validate(result, engine.get_feature_columns())
    print(f"\nValidation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['issues']:
        print("Issues:")
        for issue in validation['issues'][:10]:
            print(f"  - {issue}")

    print("\nSample features:")
    sample_cols = ['spread_pct', 'mid_ma_ratio_24h', 'return_1h', 'volatility_24h',
                   'range_24h', 'volume_ratio', 'rsi_24h']
    print(result[sample_cols].tail(5).to_string())
