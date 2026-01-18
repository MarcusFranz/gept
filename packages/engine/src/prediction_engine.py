"""Main prediction engine orchestrating all components."""

import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd

from .config import Config
from .data_loader import DataLoader
from .feature_engine import FeatureEngine
from .model_runner import ModelRunner
from .output_formatter import OutputFormatter
from .store import RecommendationStore

logger = logging.getLogger(__name__)


# Trading styles and risk levels
TradingStyle = Literal["passive", "hybrid", "active"]
RiskLevel = Literal["low", "medium", "high"]


class PredictionCache:
    """LRU cache for predictions with bounded size and TTL expiry."""

    def __init__(self, ttl_seconds: int = 60, max_size: int = 1000):
        """Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached items
            max_size: Maximum number of items to store (LRU eviction when exceeded)
        """
        self.cache: OrderedDict[str, tuple[dict, datetime]] = OrderedDict()
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_size = max_size

    def get(self, key: str) -> Optional[dict]:
        """Get item from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now(timezone.utc) - timestamp < self.ttl:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: dict):
        """Set item in cache with LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing key, move to end
            self.cache.move_to_end(key)
        self.cache[key] = (value, datetime.now(timezone.utc))

        # Evict oldest items if over max size
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest (first) item

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()


class PredictionEngine:
    """Orchestrates the full prediction pipeline."""

    def __init__(
        self,
        db_connection_string: str,
        registry_path: str,
        config: Optional[Config] = None,
    ):
        """Initialize prediction engine.

        Args:
            db_connection_string: PostgreSQL connection string
            registry_path: Path to model registry JSON
            config: Optional configuration object
        """
        self.config = config or Config()
        self.data_loader = DataLoader(
            db_connection_string,
            pool_size=self.config.db_pool_size,
        )
        self.feature_engine = FeatureEngine(granularity="1m")
        self.model_runner = ModelRunner(registry_path)
        self.output_formatter = OutputFormatter(
            ev_threshold=self.config.min_ev_threshold,
            confidence_high_auc=self.config.confidence_high_auc,
            confidence_medium_auc=self.config.confidence_medium_auc,
            data_stale_seconds=self.config.data_stale_seconds,
        )
        self.cache = PredictionCache(
            ttl_seconds=self.config.prediction_cache_ttl,
            max_size=self.config.prediction_cache_max_size,
        )
        self.recommendation_store = RecommendationStore(ttl_seconds=900)  # 15 min

        # Track supported items
        self._supported_items: Optional[set[int]] = None

    @property
    def supported_items(self) -> set[int]:
        """Get set of supported item IDs."""
        if self._supported_items is None:
            self._supported_items = set(self.model_runner.get_supported_items())
        return self._supported_items

    def predict_item(self, item_id: int, use_cache: bool = True) -> Optional[dict]:
        """Generate full prediction for one item.

        Args:
            item_id: OSRS item ID
            use_cache: Whether to use cached predictions

        Returns:
            Prediction dictionary or None if item not supported
        """
        if item_id not in self.supported_items:
            logger.warning(f"Item {item_id} not in supported items")
            return None

        # Check cache
        cache_key = f"item_{item_id}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.debug(f"Cache hit for item {item_id}")
                return cached

        start_time = time.time()

        # Get price data
        prices = self.data_loader.get_recent_prices(
            item_id,
            minutes=self.feature_engine.get_minimum_history_required(),
        )

        if prices is None or len(prices) < 100:  # Need at least some data
            logger.warning(f"Insufficient price data for item {item_id}")
            return None

        # Calculate data age
        latest_timestamp = prices.index.max()
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
        data_age = (datetime.now(timezone.utc) - latest_timestamp).total_seconds()

        # Compute features
        features = self.feature_engine.get_latest_features(prices)
        if features is None:
            logger.warning(f"Failed to compute features for item {item_id}")
            return None

        # Get model predictions
        predictions = self.model_runner.predict_all_configs(item_id, features)
        if not predictions:
            logger.warning(f"No predictions available for item {item_id}")
            return None

        # Get metadata
        item_metadata = self.model_runner.get_item_metadata(item_id)
        item_name = (
            item_metadata.get("item_name") if item_metadata else f"Item {item_id}"
        )
        tier = item_metadata.get("tier", 3) if item_metadata else 3
        avg_auc = item_metadata.get("avg_auc", 0.5) if item_metadata else 0.5

        # Get current prices
        latest_row = prices.iloc[-1]
        current_prices = {
            "high": int(latest_row["high"]) if not pd.isna(latest_row["high"]) else 0,
            "low": int(latest_row["low"]) if not pd.isna(latest_row["low"]) else 0,
            "high_volume": (
                int(latest_row["high_volume"])
                if not pd.isna(latest_row["high_volume"])
                else 0
            ),
            "low_volume": (
                int(latest_row["low_volume"])
                if not pd.isna(latest_row["low_volume"])
                else 0
            ),
        }

        # Format output
        model_metadata = {
            "avg_auc": avg_auc,
            "model_version": "v1.0.0",
            "last_trained": self.model_runner.get_registry_metadata().get("created_at"),
        }

        result = self.output_formatter.format_prediction(
            item_id=item_id,
            item_name=item_name,
            tier=tier,
            current_prices=current_prices,
            predictions=predictions,
            model_metadata=model_metadata,
            data_age_seconds=data_age,
        )

        # Add timing
        result["_latency_ms"] = round((time.time() - start_time) * 1000, 2)

        # Cache result
        self.cache.set(cache_key, result)

        return result

    def predict_all(self) -> list[dict]:
        """Generate predictions for all supported items.

        This method fetches all price data in a single query for efficiency,
        then generates predictions for each item. Results are not cached
        since the batch operation is typically used for bulk updates.

        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        predictions = []

        # Get all price data in one query for efficiency
        all_prices = self.data_loader.get_all_recent_prices(
            minutes=self.feature_engine.get_minimum_history_required()
        )

        for item_id in self.supported_items:
            if item_id not in all_prices:
                logger.debug(f"No price data for item {item_id}")
                continue

            prices = all_prices[item_id]
            if len(prices) < 100:
                continue

            try:
                pred = self._predict_from_prices(item_id, prices)
                if pred:
                    predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting item {item_id}: {e}")

        total_time = time.time() - start_time
        logger.info(f"Generated {len(predictions)} predictions in {total_time:.2f}s")

        return predictions

    def _predict_from_prices(self, item_id: int, prices) -> Optional[dict]:
        """Generate prediction from pre-fetched price data.

        Args:
            item_id: OSRS item ID
            prices: DataFrame with price history

        Returns:
            Prediction dictionary or None
        """
        # Calculate data age
        latest_timestamp = prices.index.max()
        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(tzinfo=timezone.utc)
        data_age = (datetime.now(timezone.utc) - latest_timestamp).total_seconds()

        # Compute features
        features = self.feature_engine.get_latest_features(prices)
        if features is None:
            return None

        # Get predictions
        predictions = self.model_runner.predict_all_configs(item_id, features)
        if not predictions:
            return None

        # Get metadata
        item_metadata = self.model_runner.get_item_metadata(item_id)
        item_name = (
            item_metadata.get("item_name") if item_metadata else f"Item {item_id}"
        )
        tier = item_metadata.get("tier", 3) if item_metadata else 3
        avg_auc = item_metadata.get("avg_auc", 0.5) if item_metadata else 0.5

        # Get current prices
        latest_row = prices.iloc[-1]
        current_prices = {
            "high": int(latest_row["high"]) if not pd.isna(latest_row["high"]) else 0,
            "low": int(latest_row["low"]) if not pd.isna(latest_row["low"]) else 0,
            "high_volume": (
                int(latest_row["high_volume"])
                if not pd.isna(latest_row["high_volume"])
                else 0
            ),
            "low_volume": (
                int(latest_row["low_volume"])
                if not pd.isna(latest_row["low_volume"])
                else 0
            ),
        }

        model_metadata = {
            "avg_auc": avg_auc,
            "model_version": "v1.0.0",
            "last_trained": self.model_runner.get_registry_metadata().get("created_at"),
        }

        return self.output_formatter.format_prediction(
            item_id=item_id,
            item_name=item_name,
            tier=tier,
            current_prices=current_prices,
            predictions=predictions,
            model_metadata=model_metadata,
            data_age_seconds=data_age,
        )

    def predict_actionable(
        self,
        min_ev: float = 0.001,
        tiers: Optional[list[int]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Generate predictions filtered to actionable opportunities.

        Args:
            min_ev: Minimum expected value threshold
            tiers: Optional list of tiers to filter to
            limit: Maximum number of predictions to return

        Returns:
            List of actionable predictions sorted by EV
        """
        all_predictions = self.predict_all()

        # Filter by actionability
        actionable = []
        for pred in all_predictions:
            rec = pred.get("recommendation", {})

            # Must be tradeable
            if rec.get("action") != "trade":
                continue

            # Must meet EV threshold
            if rec.get("expected_value", 0) < min_ev:
                continue

            # Must have valid prices
            if rec.get("suggested_buy") is None:
                continue

            # Filter by tier if specified
            if tiers and pred.get("tier") not in tiers:
                continue

            actionable.append(pred)

        # Sort by EV descending
        actionable.sort(
            key=lambda p: p.get("recommendation", {}).get("expected_value", 0),
            reverse=True,
        )

        return actionable[:limit]

    def get_recommendations(
        self,
        style: TradingStyle,
        capital: int,
        risk: RiskLevel,
        slots: int,
    ) -> list[dict]:
        """Get recommendations formatted for Discord bot.

        This is the main interface for the Discord bot integration.

        Args:
            style: Trading style (passive, hybrid, active)
            capital: User's available capital in gp
            risk: Risk tolerance level
            slots: Number of available GE slots

        Returns:
            List of Recommendation objects for Discord bot
        """
        # Adjust parameters based on style and risk
        min_ev = self._get_min_ev_for_risk(risk)
        preferred_hours = self._get_hours_for_style(style)
        tiers = self._get_tiers_for_risk(risk)

        # Get actionable predictions
        predictions = self.predict_actionable(
            min_ev=min_ev, tiers=tiers, limit=slots * 3
        )

        # Format for Discord and filter by capital
        recommendations = []
        for pred in predictions:
            buy_limit = self.data_loader.get_item_buy_limit(pred.get("item_id"))
            rec = self.output_formatter.format_for_discord_bot(pred, capital, buy_limit)
            if rec and rec["capitalRequired"] <= capital:
                # Filter by preferred hour windows based on style
                best_hour = (
                    pred.get("recommendation", {})
                    .get("best_config", {})
                    .get("hour_window", 4)
                )
                if best_hour in preferred_hours or not preferred_hours:
                    # Store recommendation with stable ID
                    self.recommendation_store.store(rec)
                    recommendations.append(rec)

        return recommendations[:slots]

    def get_recommendation_by_id(self, rec_id: str) -> Optional[dict]:
        """Get recommendation by its ID.

        Used by Discord bot when user clicks "Mark Ordered".

        Args:
            rec_id: Recommendation ID

        Returns:
            Recommendation dict or None if not found/expired
        """
        return self.recommendation_store.get_by_id(rec_id)

    def get_recommendation_by_item_id(self, item_id: int) -> Optional[dict]:
        """Get recommendation by item ID.

        Used by Discord bot when showing active trade details.

        Args:
            item_id: OSRS item ID

        Returns:
            Recommendation dict or None if not found/expired
        """
        return self.recommendation_store.get_by_item_id(item_id)

    def _get_min_ev_for_risk(self, risk: RiskLevel) -> float:
        """Get minimum EV threshold based on risk level."""
        thresholds = {
            "low": 0.002,  # Higher EV requirement for low risk
            "medium": 0.001,
            "high": 0.0005,  # Accept lower EV for high risk
        }
        return thresholds.get(risk, 0.001)

    def _get_hours_for_style(self, style: TradingStyle) -> list[int]:
        """Get preferred hour windows based on trading style."""
        preferences = {
            "passive": [12, 24],  # Longer holds
            "hybrid": [4, 8, 12],  # Mix
            "active": [1, 2, 4],  # Quick flips
        }
        return preferences.get(style, [1, 2, 4, 8, 12, 24])

    def _get_tiers_for_risk(self, risk: RiskLevel) -> list[int]:
        """Get item tiers based on risk level."""
        tiers = {
            "low": [1],  # Only highest quality items
            "medium": [1, 2],  # Good items
            "high": [1, 2, 3],  # All items
        }
        return tiers.get(risk, [1, 2])

    def health_check(self) -> dict:
        """Check database connection, model loading, data freshness.

        Returns:
            Health status dictionary
        """
        checks = []

        # Database check
        db_health = self.data_loader.health_check()
        checks.append(db_health)

        # Model runner check
        model_health = self.model_runner.health_check()
        checks.append(model_health)

        # Prediction latency check
        try:
            test_item = next(iter(self.supported_items), None)
            if test_item:
                start = time.time()
                self.predict_item(test_item, use_cache=False)
                latency = (time.time() - start) * 1000

                checks.append(
                    {
                        "status": "ok" if latency < 500 else "warning",
                        "component": "prediction_latency",
                        "latency_ms": round(latency, 2),
                    }
                )
        except Exception as e:
            checks.append(
                {
                    "status": "error",
                    "component": "prediction_latency",
                    "error": str(e),
                }
            )

        # Overall status
        statuses = [c.get("status") for c in checks]
        if "error" in statuses:
            overall = "error"
        elif "warning" in statuses:
            overall = "warning"
        else:
            overall = "ok"

        return {
            "status": overall,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "supported_items": len(self.supported_items),
        }

    def close(self):
        """Clean up resources."""
        self.data_loader.close()
        self.model_runner.clear_cache()
        self.cache.clear()
        self.recommendation_store.clear()
