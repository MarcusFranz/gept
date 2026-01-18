"""Feature engineering for ML model predictions."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Computes ML features from price data.

    The models were trained on 5-minute data. When computing features from
    1-minute data, window sizes are scaled by 5x to maintain parity.
    """

    # Feature names in the order expected by the model
    FEATURE_NAMES = [
        "mid",
        "spread",
        "spread_pct",
        "ma_ratio_60",
        "ma_ratio_240",
        "ma_ratio_480",
        "ma_ratio_1440",
        "return_5",
        "return_15",
        "return_30",
        "return_60",
        "return_240",
        "volatility_60",
        "volatility_240",
        "volatility_1440",
        "volume",
        "volume_ma_1440",
        "volume_ratio",
        "spread_ma_240",
        "spread_ratio",
        "range_60",
        "range_240",
        "range_1440",
        "hour",
        "day_of_week",
        "is_weekend",
    ]

    def __init__(self, granularity: str = "1m"):
        """Initialize feature engine.

        Args:
            granularity: Data granularity ('1m' or '5m')
        """
        self.granularity = granularity
        # Models trained on 5-min data; scale windows for 1-min data
        self.window_multiplier = 1 if granularity == "5m" else 5

    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a price DataFrame.

        Args:
            prices: DataFrame with columns: high, low, high_volume, low_volume
                   Index should be timestamp

        Returns:
            DataFrame with computed features (same index as input)
        """
        if prices is None or prices.empty:
            logger.warning("Empty price data provided")
            return pd.DataFrame()

        df = prices.copy()
        wm = self.window_multiplier

        # Price basics
        df["mid"] = (df["high"] + df["low"]) / 2
        df["spread"] = df["high"] - df["low"]
        df["spread_pct"] = df["spread"] / df["mid"]

        # Moving average ratios
        # Window sizes are in original 5-min terms, scaled by multiplier
        df["ma_ratio_60"] = (
            df["mid"] / df["mid"].rolling(window=60 * wm, min_periods=1).mean()
        )
        df["ma_ratio_240"] = (
            df["mid"] / df["mid"].rolling(window=240 * wm, min_periods=1).mean()
        )
        df["ma_ratio_480"] = (
            df["mid"] / df["mid"].rolling(window=480 * wm, min_periods=1).mean()
        )
        df["ma_ratio_1440"] = (
            df["mid"] / df["mid"].rolling(window=1440 * wm, min_periods=1).mean()
        )

        # Returns at different lookbacks
        df["return_5"] = df["mid"].pct_change(periods=5 * wm)
        df["return_15"] = df["mid"].pct_change(periods=15 * wm)
        df["return_30"] = df["mid"].pct_change(periods=30 * wm)
        df["return_60"] = df["mid"].pct_change(periods=60 * wm)
        df["return_240"] = df["mid"].pct_change(periods=240 * wm)

        # Calculate returns for volatility
        returns = df["mid"].pct_change()

        # Volatility
        df["volatility_60"] = returns.rolling(window=60 * wm, min_periods=1).std()
        df["volatility_240"] = returns.rolling(window=240 * wm, min_periods=1).std()
        df["volatility_1440"] = returns.rolling(window=1440 * wm, min_periods=1).std()

        # Volume
        df["volume"] = df["high_volume"].fillna(0) + df["low_volume"].fillna(0)
        df["volume_ma_1440"] = (
            df["volume"].rolling(window=1440 * wm, min_periods=1).mean()
        )
        df["volume_ratio"] = df["volume"] / df["volume_ma_1440"].replace(0, np.nan)

        # Spread dynamics
        df["spread_ma_240"] = (
            df["spread_pct"].rolling(window=240 * wm, min_periods=1).mean()
        )
        df["spread_ratio"] = df["spread_pct"] / df["spread_ma_240"].replace(0, np.nan)

        # Price range (for fill probability)
        df["range_60"] = (
            df["high"].rolling(window=60 * wm, min_periods=1).max()
            - df["low"].rolling(window=60 * wm, min_periods=1).min()
        ) / df["mid"]

        df["range_240"] = (
            df["high"].rolling(window=240 * wm, min_periods=1).max()
            - df["low"].rolling(window=240 * wm, min_periods=1).min()
        ) / df["mid"]

        df["range_1440"] = (
            df["high"].rolling(window=1440 * wm, min_periods=1).max()
            - df["low"].rolling(window=1440 * wm, min_periods=1).min()
        ) / df["mid"]

        # Time features (from index)
        if isinstance(df.index, pd.DatetimeIndex):
            df["hour"] = df.index.hour
            df["day_of_week"] = df.index.dayofweek
            df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        else:
            # Fallback if index is not datetime
            df["hour"] = 0
            df["day_of_week"] = 0
            df["is_weekend"] = 0

        # Replace infinities with NaN, then forward fill
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        return df[self.FEATURE_NAMES]

    def get_latest_features(self, prices: pd.DataFrame) -> Optional[np.ndarray]:
        """Get feature vector for the most recent timestamp only.

        Args:
            prices: DataFrame with price history

        Returns:
            1D numpy array of features for the latest timestamp,
            or None if insufficient data
        """
        features = self.compute_features(prices)
        if features.empty:
            return None

        # Get the last row
        latest = features.iloc[-1].values

        # Check for NaN values
        if np.isnan(latest).any():
            logger.warning("NaN values in latest features, attempting to fill")
            latest = np.nan_to_num(latest, nan=0.0)

        return latest

    def get_feature_names(self) -> list[str]:
        """Return ordered list of feature names.

        Returns:
            List of feature names in the order expected by models
        """
        return self.FEATURE_NAMES.copy()

    def validate_features(
        self, features: np.ndarray, expected_names: Optional[list[str]] = None
    ) -> bool:
        """Validate feature vector.

        Args:
            features: Feature vector to validate
            expected_names: Optional list of expected feature names

        Returns:
            True if features are valid
        """
        if features is None:
            return False

        if len(features) != len(self.FEATURE_NAMES):
            logger.error(
                f"Feature count mismatch: got {len(features)}, "
                f"expected {len(self.FEATURE_NAMES)}"
            )
            return False

        if np.isnan(features).any():
            logger.warning("NaN values in feature vector")
            return False

        if np.isinf(features).any():
            logger.warning("Infinite values in feature vector")
            return False

        return True

    def get_minimum_history_required(self) -> int:
        """Get minimum number of data points required for feature computation.

        Returns:
            Minimum number of 1-minute data points needed
        """
        # 1440 minutes = 24 hours for the longest rolling window
        return 1440 * self.window_multiplier
