"""Recommendation engine that transforms raw predictions into actionable trades.

This engine reads pre-computed predictions from the Ampere server's database
and applies user constraints (capital, style, risk, slots) to generate
optimized trade recommendations for the Discord bot.
"""

import logging
import math
import random
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd

from .config import Config
from .crowding import create_crowding_tracker
from .ml_ranker import MLRanker, MLFeatureBuilder
from .prediction_loader import PredictionLoader
from .shadow_logger import ShadowLogger
from .store import RecommendationStore  # Re-exported for backward compatibility
from .tax_calculator import calculate_tax

logger = logging.getLogger(__name__)


# Trading styles and risk levels
TradingStyle = Literal["passive", "hybrid", "active"]
RiskLevel = Literal["low", "medium", "high"]
ConfidenceLevel = Literal["high", "medium", "low"]


class RecommendationEngine:
    """Transforms raw predictions into optimized trade recommendations."""

    # EV thresholds by risk level
    EV_THRESHOLDS = {
        "low": 0.008,  # Higher bar for low risk
        "medium": 0.005,  # Standard threshold
        "high": 0.003,  # Accept lower EV for high risk
    }

    # Hour offset ranges by trading style
    HOUR_RANGES = {
        "active": (1, 4),  # Quick flips
        "hybrid": (2, 12),  # Medium-term
        "passive": (8, 48),  # Overnight/long-term
    }

    # Minimum fill probability thresholds by risk
    # Note: Upper bounds removed per issue #64 - high fill probabilities
    # indicate model confidence and should not be discarded
    FILL_PROB_MINIMUMS = {
        "low": 0.08,  # Require higher confidence
        "medium": 0.05,  # Standard minimum
        "high": 0.03,  # Accept more uncertainty
    }

    # Offset percentage ranges by risk level
    OFFSET_RANGES = {
        "low": (0.0125, 0.0175),  # Conservative: 1.25% - 1.75%
        "medium": (0.0150, 0.0200),  # Moderate: 1.5% - 2.0%
        "high": (0.0175, 0.0250),  # Aggressive: 1.75% - 2.5%
    }

    # Risk penalty weights by risk level
    # Lower risk profiles receive HIGHER penalties (more conservative)
    EXIT_RISK_WEIGHTS = {
        "low": 0.40,  # 40% weight on exit risk penalty
        "medium": 0.25,  # 25% weight
        "high": 0.10,  # 10% weight (accept more exit risk)
    }

    FILL_RISK_WEIGHTS = {
        "low": 0.35,  # 35% weight on fill risk penalty
        "medium": 0.20,  # 20% weight
        "high": 0.08,  # 8% weight (accept more fill risk)
    }

    def __init__(
        self,
        db_connection_string: str,
        config: Optional[Config] = None,
        enable_ml_shadow: bool = False,
        ml_model_path: Optional[str] = None,
    ):
        """Initialize recommendation engine.

        Args:
            db_connection_string: PostgreSQL connection string to Ampere server
            config: Optional configuration object
            enable_ml_shadow: Whether to enable ML ranker shadow mode
            ml_model_path: Optional path to ML ranker model file
        """
        self.config = config or Config()
        self.loader = PredictionLoader(
            db_connection_string,
            pool_size=self.config.db_pool_size,
        )
        self.store = RecommendationStore(ttl_seconds=900)

        # Create crowding tracker (Redis if configured, otherwise in-memory)
        self.crowding_tracker = create_crowding_tracker(
            redis_url=self.config.redis_url or None,
            fallback_to_memory=self.config.redis_fallback_to_memory,
        )

        # ML ranker shadow mode
        self.enable_ml_shadow = enable_ml_shadow
        self.ml_ranker = None
        self.ml_feature_builder = None
        self.shadow_logger = None

        if enable_ml_shadow:
            self.ml_ranker = MLRanker(model_path=ml_model_path)
            self.ml_feature_builder = MLFeatureBuilder(self.loader)
            self.shadow_logger = ShadowLogger(enabled=True)
            if self.ml_ranker.is_available():
                logger.info("ML ranker shadow mode enabled")
            else:
                logger.warning("ML ranker model not loaded - shadow mode disabled")

    def _apply_price_buffer(self, buy_price: int, sell_price: int) -> tuple[int, int]:
        """Apply a random buffer to buy and sell prices to reduce price competition.

        The buffer is a random percentage of the margin (sell - buy) that moves
        prices toward the market: buy price moves UP, sell price moves DOWN.

        If 1% of margin < 1gp, no buffer is applied (prevents sub-gp adjustments).

        Args:
            buy_price: Original buy price in gp
            sell_price: Original sell price in gp

        Returns:
            Tuple of (buffered_buy_price, buffered_sell_price)
        """
        if not self.config.price_buffer_enabled:
            return buy_price, sell_price

        margin = sell_price - buy_price
        if margin <= 0:
            return buy_price, sell_price

        # Check if 1% of margin is at least 1gp
        min_buffer = margin * (self.config.price_buffer_min_pct / 100.0)
        if min_buffer < 1.0:
            # Margin too small for meaningful buffer, skip
            return buy_price, sell_price

        # Calculate random buffer as percentage of margin
        buffer_pct = random.uniform(
            self.config.price_buffer_min_pct, self.config.price_buffer_max_pct
        )
        buffer_gp = int(margin * (buffer_pct / 100.0))

        # Apply buffer: buy moves UP, sell moves DOWN (toward market)
        buffered_buy = buy_price + buffer_gp
        buffered_sell = sell_price - buffer_gp

        # Ensure we don't cross over (buy > sell)
        if buffered_buy >= buffered_sell:
            return buy_price, sell_price

        logger.debug(
            f"Applied price buffer: buy {buy_price} -> {buffered_buy}, "
            f"sell {sell_price} -> {buffered_sell} (buffer: {buffer_gp}gp, {buffer_pct:.1f}%)"
        )

        return buffered_buy, buffered_sell

    def get_recommendations(
        self,
        style: TradingStyle,
        capital: int,
        risk: RiskLevel,
        slots: int,
        active_trades: Optional[list[dict]] = None,
        exclude_ids: Optional[set[str]] = None,
        exclude_item_ids: Optional[set[int]] = None,
        user_id: Optional[str] = None,
        offset_pct: Optional[float] = None,
        min_offset_pct: Optional[float] = None,
        max_offset_pct: Optional[float] = None,
        max_hour_offset: Optional[int] = None,
        min_ev: Optional[float] = None,
    ) -> list[dict]:
        """Get optimized trade recommendations for user constraints.

        Args:
            style: Trading style (passive, hybrid, active)
            capital: Total capital in gp
            risk: Risk tolerance level
            slots: Total GE slots (1-8)
            active_trades: List of tracked trades with itemId, quantity, buyPrice
            exclude_ids: Set of recommendation IDs to exclude (already seen)
            exclude_item_ids: Set of item IDs to exclude (e.g., active trades)
            user_id: Optional user identifier for crowding tracking
            offset_pct: Optional specific offset percentage to filter by (takes precedence)
            min_offset_pct: Optional minimum offset percentage threshold
            max_offset_pct: Optional maximum offset percentage threshold
            max_hour_offset: Optional max time horizon override (1-48)
            min_ev: Optional minimum expected value threshold override

        Returns:
            List of Recommendation objects for Discord bot
        """
        active_trades = active_trades or []
        exclude_ids = exclude_ids or set()
        exclude_item_ids = exclude_item_ids or set()

        # Calculate remaining capital and slots after active trades
        capital_in_use = sum(t["quantity"] * t["buyPrice"] for t in active_trades)
        remaining_capital = max(0, capital - capital_in_use)
        slots_in_use = len(active_trades)
        available_slots = max(0, slots - slots_in_use)

        # Get item IDs to exclude (already tracking + explicit exclusions)
        excluded_items = {t["itemId"] for t in active_trades}
        excluded_items.update(exclude_item_ids)

        if remaining_capital < 1000 or available_slots < 1:
            logger.info(
                f"No capacity: remaining_capital={remaining_capital}, "
                f"available_slots={available_slots}"
            )
            return []

        # Get filtering parameters based on style/risk
        min_ev_threshold = (
            min_ev if min_ev is not None else self.EV_THRESHOLDS.get(risk, 0.005)
        )
        min_hour, max_hour = self.HOUR_RANGES.get(style, (1, 48))
        min_fill = self.FILL_PROB_MINIMUMS.get(risk, 0.05)

        # Allow max_hour_offset parameter to override style default
        if max_hour_offset is not None:
            max_hour = min(max_hour_offset, 48)

        # Determine offset range (priority: offset_pct > explicit min/max > risk-based)
        if offset_pct is not None:
            # Exact offset takes precedence
            min_offset, max_offset = offset_pct, offset_pct
        elif min_offset_pct is not None or max_offset_pct is not None:
            # Use explicit min/max if provided, fall back to risk-based defaults
            risk_min, risk_max = self.OFFSET_RANGES.get(risk, (0.0125, 0.0250))
            min_offset = min_offset_pct if min_offset_pct is not None else risk_min
            max_offset = max_offset_pct if max_offset_pct is not None else risk_max
        else:
            # Default to risk-based range
            min_offset, max_offset = self.OFFSET_RANGES.get(risk, (0.0125, 0.0250))

        # Calculate dynamic pool size based on capital
        # Large capital users need access to high-price items beyond top EV
        # log10(1M) = 6, log10(1B) = 9, log10(10B) = 10
        # capital_factor: 0 at 1M, 1 at 10M, 2 at 100M, 3 at 1B
        capital_factor = max(
            0, math.log10(max(remaining_capital, 1_000_000) / 1_000_000)
        )
        base_limit = max(available_slots * 10, 100)
        candidate_limit = int(base_limit * (1 + capital_factor * 0.5))
        candidate_limit = min(
            candidate_limit, 500
        )  # Hard cap to prevent excessive queries

        # Fetch predictions from database with hour/volume filtering at DB level
        predictions_df = self.loader.get_best_prediction_per_item(
            min_fill_prob=min_fill,
            min_ev=min_ev_threshold,
            min_hour_offset=min_hour,
            max_hour_offset=max_hour,
            min_offset_pct=min_offset,
            max_offset_pct=max_offset,
            limit=candidate_limit,
            min_volume_24h=self.config.min_volume_24h,
        )

        if predictions_df.empty:
            logger.warning("No predictions found matching criteria")
            return []

        # Exclude tracked items (hour filtering is now done at DB level)
        predictions_df = predictions_df[~predictions_df["item_id"].isin(excluded_items)]

        # Apply anti-manipulation filter
        predictions_df = self._apply_price_stability_filter(predictions_df)
        if predictions_df.empty:
            logger.info("All predictions filtered by stability check")
            return []

        # Apply anti-adverse selection filter
        predictions_df = self._apply_trend_entry_filter(predictions_df, style=style)
        if predictions_df.empty:
            logger.info("All predictions filtered by trend check")
            return []

        # Get prediction age for confidence adjustment
        pred_age = self.loader.get_prediction_age_seconds()

        # Batch fetch all per-item data upfront (eliminates N+1 queries)
        candidate_item_ids = predictions_df["item_id"].unique().tolist()
        buy_limits = self.loader.get_batch_buy_limits(candidate_item_ids)
        volumes_24h = self.loader.get_batch_volumes_24h(candidate_item_ids)
        volumes_1h = self.loader.get_batch_volumes_1h(candidate_item_ids)
        trends = self.loader.get_batch_trends(candidate_item_ids)

        # Build candidate recommendations with flexible sizing
        candidates = []
        for _, row in predictions_df.iterrows():
            item_id = int(row["item_id"])

            # Skip if excluded
            rec_id = self._generate_stable_id(item_id)
            if rec_id in exclude_ids:
                continue

            # Create candidate with full details for optimization
            candidate = self._build_candidate(
                row,
                remaining_capital,
                pred_age,
                style,
                buy_limits=buy_limits,
                volumes_24h=volumes_24h,
                volumes_1h=volumes_1h,
                trends=trends,
            )
            if candidate:
                candidates.append(candidate)

        if not candidates:
            logger.info("No viable candidates after filtering")
            return []

        # Filter out crowded items
        candidates = self.crowding_tracker.filter_crowded_items(candidates)

        if not candidates:
            logger.info("No candidates available after crowding filter")
            return []

        # ML ranker shadow mode: score candidates and log comparison
        if self.enable_ml_shadow and self.ml_ranker and self.ml_ranker.is_available():
            request_id = str(uuid.uuid4())[:8]
            try:
                # Build ML features for candidate items
                candidate_item_ids = [c.get("item_id") for c in candidates if c.get("item_id")]
                pred_features = self.ml_feature_builder.build_features_for_items(candidate_item_ids)

                # Score with ML ranker
                candidates = self.ml_ranker.score_candidates(candidates, pred_features)

                # Log comparison between heuristic and ML rankings
                if self.shadow_logger:
                    self.shadow_logger.log_rankings(
                        request_id=request_id,
                        user_id=user_id,
                        style=style,
                        risk=risk,
                        capital=capital,
                        slots=slots,
                        candidates=candidates,
                    )
            except Exception as e:
                logger.warning(f"ML ranker shadow scoring failed: {e}")

        # Optimize portfolio selection
        selected = self._optimize_portfolio(
            candidates=candidates,
            total_capital=remaining_capital,
            num_slots=available_slots,
            risk=risk,
            style=style,
        )

        # Store and format final recommendations
        recommendations = []
        for rec in selected:
            self.store.store(rec)
            recommendations.append(rec)

            # Track delivery for crowding limits
            if user_id:
                item_id = rec.get("itemId")
                if item_id:
                    self.crowding_tracker.record_delivery(item_id, user_id)

        logger.info(
            f"Generated {len(recommendations)} recommendations for "
            f"style={style}, capital={capital:,} (remaining={remaining_capital:,}), "
            f"risk={risk}, slots={slots} (available={available_slots}), "
            f"excluded={len(exclude_ids)}, candidates={len(candidates)}, "
            f"user_id={user_id or 'anonymous'}"
        )

        return recommendations

    def get_all_recommendations(
        self,
        style: TradingStyle,
        capital: int,
        risk: RiskLevel,
        active_trades: Optional[list[dict]] = None,
        exclude_item_ids: Optional[set[int]] = None,
        user_id: Optional[str] = None,
        offset_pct: Optional[float] = None,
        min_offset_pct: Optional[float] = None,
        max_offset_pct: Optional[float] = None,
        max_hour_offset: Optional[int] = None,
        min_ev: Optional[float] = None,
    ) -> list[dict]:
        """Get ALL viable recommendations sorted by score (no slot limit).

        Unlike get_recommendations(), this method returns all viable candidates
        without portfolio optimization or slot-based limiting. This is useful
        for clients that want to cache and browse recommendations locally.

        Args:
            style: Trading style (passive, hybrid, active)
            capital: Total capital in gp
            risk: Risk tolerance level
            active_trades: List of tracked trades with itemId, quantity, buyPrice
            exclude_item_ids: Set of item IDs to exclude (e.g., active trades)
            user_id: Optional user identifier for crowding tracking
            offset_pct: Optional specific offset percentage to filter by
            min_offset_pct: Optional minimum offset percentage threshold
            max_offset_pct: Optional maximum offset percentage threshold
            max_hour_offset: Optional max time horizon override (1-48)
            min_ev: Optional minimum expected value threshold override

        Returns:
            List of all viable recommendations sorted by composite score (descending)
        """
        active_trades = active_trades or []
        exclude_item_ids = exclude_item_ids or set()

        # Calculate remaining capital after active trades
        capital_in_use = sum(t["quantity"] * t["buyPrice"] for t in active_trades)
        remaining_capital = max(0, capital - capital_in_use)

        # Get item IDs to exclude (already tracking + explicit exclusions)
        excluded_items = {t["itemId"] for t in active_trades}
        excluded_items.update(exclude_item_ids)

        if remaining_capital < 1000:
            logger.info(f"No capacity: remaining_capital={remaining_capital}")
            return []

        # Get filtering parameters based on style/risk
        min_ev_threshold = (
            min_ev if min_ev is not None else self.EV_THRESHOLDS.get(risk, 0.005)
        )
        min_hour, max_hour = self.HOUR_RANGES.get(style, (1, 48))
        min_fill = self.FILL_PROB_MINIMUMS.get(risk, 0.05)

        # Allow max_hour_offset parameter to override style default
        if max_hour_offset is not None:
            max_hour = min(max_hour_offset, 48)

        # Determine offset range
        if offset_pct is not None:
            min_offset, max_offset = offset_pct, offset_pct
        elif min_offset_pct is not None or max_offset_pct is not None:
            risk_min, risk_max = self.OFFSET_RANGES.get(risk, (0.0125, 0.0250))
            min_offset = min_offset_pct if min_offset_pct is not None else risk_min
            max_offset = max_offset_pct if max_offset_pct is not None else risk_max
        else:
            min_offset, max_offset = self.OFFSET_RANGES.get(risk, (0.0125, 0.0250))

        # Fetch more predictions since we're not limiting by slots
        # Use a larger limit to get all viable candidates
        candidate_limit = 500

        # Fetch predictions from database
        predictions_df = self.loader.get_best_prediction_per_item(
            min_fill_prob=min_fill,
            min_ev=min_ev_threshold,
            min_hour_offset=min_hour,
            max_hour_offset=max_hour,
            min_offset_pct=min_offset,
            max_offset_pct=max_offset,
            limit=candidate_limit,
            min_volume_24h=self.config.min_volume_24h,
        )

        if predictions_df.empty:
            logger.warning("No predictions found matching criteria for get_all")
            return []

        # Exclude tracked items
        predictions_df = predictions_df[~predictions_df["item_id"].isin(excluded_items)]

        # Get prediction age for confidence adjustment
        pred_age = self.loader.get_prediction_age_seconds()

        # Batch fetch all per-item data upfront (eliminates N+1 queries)
        candidate_item_ids = predictions_df["item_id"].unique().tolist()
        buy_limits = self.loader.get_batch_buy_limits(candidate_item_ids)
        volumes_24h = self.loader.get_batch_volumes_24h(candidate_item_ids)
        volumes_1h = self.loader.get_batch_volumes_1h(candidate_item_ids)
        trends = self.loader.get_batch_trends(candidate_item_ids)

        # Build candidates
        candidates = []
        for _, row in predictions_df.iterrows():
            item_id = int(row["item_id"])
            candidate = self._build_candidate(
                row,
                remaining_capital,
                pred_age,
                style,
                buy_limits=buy_limits,
                volumes_24h=volumes_24h,
                volumes_1h=volumes_1h,
                trends=trends,
            )
            if candidate:
                candidates.append(candidate)

        if not candidates:
            logger.info("No viable candidates after filtering for get_all")
            return []

        # Filter out crowded items
        candidates = self.crowding_tracker.filter_crowded_items(candidates)

        if not candidates:
            logger.info("No candidates available after crowding filter for get_all")
            return []

        # ML ranker shadow mode: score candidates and log comparison
        if self.enable_ml_shadow and self.ml_ranker and self.ml_ranker.is_available():
            request_id = str(uuid.uuid4())[:8]
            try:
                # Build ML features for candidate items
                candidate_item_ids = [c.get("item_id") for c in candidates if c.get("item_id")]
                pred_features = self.ml_feature_builder.build_features_for_items(candidate_item_ids)

                # Score with ML ranker
                candidates = self.ml_ranker.score_candidates(candidates, pred_features)

                # Log comparison between heuristic and ML rankings
                if self.shadow_logger:
                    self.shadow_logger.log_rankings(
                        request_id=request_id,
                        user_id=user_id,
                        style=style,
                        risk=risk,
                        capital=capital,
                        slots=8,  # get_all_recommendations doesn't have slots param
                        candidates=candidates,
                    )
            except Exception as e:
                logger.warning(f"ML ranker shadow scoring failed in get_all: {e}")

        # Calculate composite score for each candidate and format as recommendation
        recommendations = []
        all_fill_probs = [c["fill_probability"] for c in candidates]

        for cand in candidates:
            fill_prob = cand["fill_probability"]
            ev = cand["expected_value"]
            buy_price = cand["buy_price"]
            profit_per_unit = cand["profit_per_unit"]

            # Calculate max quantity (simplified sizing - no portfolio optimization)
            max_quantity = min(cand["max_quantity"], remaining_capital // buy_price)
            if max_quantity < 1:
                continue

            capital_used = buy_price * max_quantity
            expected_profit = int(profit_per_unit * max_quantity * fill_prob)

            # Calculate margin percentage for scoring
            margin_pct = profit_per_unit / buy_price if buy_price > 0 else 0

            # Composite score formula:
            # 40% EV, 35% fill probability, 25% margin percentage
            composite_score = ev * 0.40 + fill_prob * 0.35 + margin_pct * 0.25

            rec = {
                "id": self._generate_stable_id(cand["item_id"]),
                "itemId": cand["item_id"],
                "item": cand["item_name"],
                "buyPrice": buy_price,
                "sellPrice": cand["sell_price"],
                "quantity": max_quantity,
                "capitalRequired": capital_used,
                "expectedProfit": expected_profit,
                "confidence": cand["confidence"],
                "volumeTier": cand["volume_tier"],
                "trend": cand["trend"],
                "expectedHours": cand["hour_offset"],
                "fillProbability": round(fill_prob, 4),
                "fillConfidence": self._determine_fill_confidence(
                    fill_prob, all_fill_probs
                ),
                "_expectedValue": round(ev, 6),
                "_score": round(composite_score, 6),
                "reason": self._build_reason(cand),
                "isRecommended": True,
            }

            # Add optional volume24h field if available
            if cand.get("volume_24h") is not None:
                rec["volume24h"] = cand["volume_24h"]

            recommendations.append(rec)

        # Sort by composite score descending
        recommendations.sort(key=lambda x: x.get("_score", 0), reverse=True)

        # Track delivery for crowding limits
        if user_id:
            for rec in recommendations:
                item_id = rec.get("itemId")
                if item_id:
                    self.crowding_tracker.record_delivery(item_id, user_id)

        logger.info(
            f"Generated {len(recommendations)} all-recommendations for "
            f"style={style}, capital={capital:,} (remaining={remaining_capital:,}), "
            f"risk={risk}, user_id={user_id or 'anonymous'}"
        )

        return recommendations

    def _generate_stable_id(self, item_id: int) -> str:
        """Generate stable ID for an item (matches RecommendationStore)."""
        now = datetime.now(timezone.utc)
        hour_bucket = now.strftime("%Y%m%d%H")
        return f"rec_{item_id}_{hour_bucket}"

    def get_effective_buy_limit(
        self,
        base_limit: int,
        style: TradingStyle,
        expected_hours: int,
    ) -> tuple[int, bool]:
        """Calculate effective buy limit based on trading style and expected hold time.

        For passive mode, traders can feasibly buy up to 4x the GE limit by purchasing
        one limit's worth, waiting for the 4-hour reset, and purchasing again.
        With 48h horizons now supported, up to 12 resets are possible, but we cap at 4x
        (16h+ of resets) to keep recommendations practical.

        Args:
            base_limit: Base GE buy limit for the item
            style: Trading style (passive, hybrid, active)
            expected_hours: Expected hold time in hours

        Returns:
            Tuple of (effective_limit, is_multi_limit_strategy)
        """
        if style != "passive":
            return base_limit, False

        # How many full 4-hour cycles fit in expected hold time?
        limit_resets = expected_hours // 4

        # Cap at 4x to keep recommendations reasonable for 48h horizons
        # (4x requires 16h of resets, practical for long passive trades)
        multiplier = min(limit_resets + 1, 4)

        if multiplier > 1:
            return base_limit * multiplier, True

        return base_limit, False

    def _calculate_tax_per_unit(self, sell_price: int) -> int:
        """Calculate GE tax per unit sold.

        Uses tax_calculator.calculate_tax() which implements:
        - 2% tax rate
        - Rounds DOWN to nearest whole number
        - Items below 50gp: 0 tax
        - Maximum 5,000,000gp tax per item

        Args:
            sell_price: Price per item when selling

        Returns:
            Tax amount per unit (integer)
        """
        return calculate_tax(sell_price, 1)

    def _build_candidate(
        self,
        row,
        max_capital: int,
        pred_age_seconds: float,
        style: TradingStyle = "hybrid",
        buy_limits: Optional[dict[int, int]] = None,
        volumes_24h: Optional[dict[int, int]] = None,
        volumes_1h: Optional[dict[int, int]] = None,
        trends: Optional[dict[int, str]] = None,
    ) -> Optional[dict]:
        """Build a candidate recommendation with sizing flexibility.

        Args:
            row: DataFrame row with prediction data
            max_capital: Maximum capital available for this recommendation
            pred_age_seconds: Age of prediction in seconds (for confidence adjustment)
            style: Trading style (passive, hybrid, active)
            buy_limits: Pre-fetched buy limits dict (item_id -> buy_limit).
                       If None, falls back to individual lookup (slower).
            volumes_24h: Pre-fetched 24h volumes dict (item_id -> volume).
                        If None, falls back to individual lookup (slower).
            volumes_1h: Pre-fetched 1h volumes dict (item_id -> volume).
                       If None, falls back to individual lookup (slower).
            trends: Pre-fetched trends dict (item_id -> trend string).
                   If None, falls back to individual lookup (slower).

        Returns:
            Candidate dict with buy_price, max_quantity, and per-unit metrics
            so the optimizer can adjust quantity, or None if candidate is invalid.
        """
        try:
            item_id = int(row["item_id"]) if row["item_id"] is not None else 0
            buy_price_raw = row["buy_price"]
            sell_price_raw = row["sell_price"]
            fill_prob_raw = row["fill_probability"]
            ev_raw = row["expected_value"]

            if buy_price_raw is None or (
                isinstance(buy_price_raw, float) and math.isnan(buy_price_raw)
            ):
                return None
            if sell_price_raw is None or (
                isinstance(sell_price_raw, float) and math.isnan(sell_price_raw)
            ):
                return None

            buy_price = int(buy_price_raw)
            sell_price = int(sell_price_raw)
            fill_prob = float(fill_prob_raw) if fill_prob_raw is not None else 0.0
            ev = float(ev_raw) if ev_raw is not None else 0.0
            hour_offset = int(row["hour_offset"])

            if buy_price <= 0 or sell_price <= 0:
                return None

            # Apply price buffer to reduce competition on exact prices
            # Buffer moves buy UP and sell DOWN (toward market)
            buy_price, sell_price = self._apply_price_buffer(buy_price, sell_price)

            # Block instant-fill: buy_price at or above current_high would instant-fill
            current_high_raw = row.get("current_high")
            if current_high_raw is None or (
                isinstance(current_high_raw, float) and math.isnan(current_high_raw)
            ):
                # Fail closed: skip candidate if current_high is missing/NaN
                logger.debug(
                    f"Skipping item {item_id}: missing current_high (fail closed)"
                )
                return None

            current_high = int(current_high_raw)
            if buy_price >= current_high:
                logger.debug(
                    f"Blocking instant-fill for item {item_id}: "
                    f"buy_price {buy_price} >= current_high {current_high}"
                )
                return None

            # Calculate max quantity based on capital and effective buy limit
            max_quantity = max_capital // buy_price

            # Use pre-fetched buy limits if available, otherwise fall back to individual lookup
            if buy_limits is not None:
                base_buy_limit = buy_limits.get(item_id)
            else:
                base_buy_limit = self.loader.get_item_buy_limit(item_id)

            is_multi_limit = False
            if base_buy_limit and base_buy_limit > 0:
                effective_limit, is_multi_limit = self.get_effective_buy_limit(
                    base_buy_limit, style, hour_offset
                )
                max_quantity = min(max_quantity, effective_limit)
            else:
                # Conservative fallback when buy limit unknown
                # 1000 is safer than 10000 as most valuable items have limits <1000
                logger.debug(f"No buy limit for item {item_id}, using fallback of 1000")
                max_quantity = min(max_quantity, 1000)

            if max_quantity < 1:
                return None

            # Per-unit profit accounting for GE tax (before fill probability)
            tax_per_unit = self._calculate_tax_per_unit(sell_price)
            profit_per_unit = sell_price - buy_price - tax_per_unit

            # Confidence and other metadata
            db_confidence = row.get("confidence", "medium") or "medium"
            confidence = self._adjust_confidence(db_confidence, pred_age_seconds)

            # current_high already validated and set above (instant-fill check)
            current_low = row.get("current_low") or buy_price
            if isinstance(current_low, float) and math.isnan(current_low):
                current_low = buy_price
            spread_pct = (
                (current_high - current_low) / current_low if current_low > 0 else 0
            )

            # Use pre-fetched volume data if available, otherwise fall back to individual lookup
            if volumes_24h is not None:
                volume_24h = volumes_24h.get(item_id)
            else:
                volume_24h = self.loader.get_item_volume_24h(item_id)

            if volumes_1h is not None:
                volume_1h = volumes_1h.get(item_id, 0)
            else:
                volume_1h = self.loader.get_item_volume_1h(item_id)

            # Check for manipulation signals - skip suspicious items
            is_suspicious, manipulation_risk, manipulation_reasons = (
                self._check_manipulation_signals(
                    item_id=item_id,
                    buy_price=buy_price,
                    spread_pct=spread_pct,
                    volume_24h=volume_24h,
                    volume_1h=volume_1h,
                )
            )
            if is_suspicious:
                logger.debug(
                    f"Skipping item {item_id} due to manipulation signals: "
                    f"{manipulation_reasons}"
                )
                return None

            volume_tier = self._determine_volume_tier(spread_pct)
            crowding_capacity = self._get_crowding_capacity(item_id, volume_1h)

            # Use pre-fetched trend if available, otherwise fall back to individual lookup
            if trends is not None:
                trend = trends.get(item_id, "Stable")
            else:
                trend = self.loader.get_item_trend(item_id)
            item_name = str(row["item_name"])

            return {
                "item_id": item_id,
                "item_name": item_name,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "max_quantity": max_quantity,
                "profit_per_unit": profit_per_unit,
                "fill_probability": fill_prob,
                "expected_value": ev,
                "confidence": confidence,
                "volume_tier": volume_tier,
                "crowding_capacity": crowding_capacity,
                "trend": trend,
                "volume_24h": volume_24h,
                "hour_offset": hour_offset,
                "is_multi_limit": is_multi_limit,
                "base_buy_limit": base_buy_limit if base_buy_limit else None,
                "_spread_pct": spread_pct,  # For exit risk calculation
            }

        except (ValueError, TypeError) as e:
            logger.debug(f"Error building candidate: {e}")
            return None

    def _calculate_exit_risk_penalty(
        self,
        candidate: dict,
        risk: RiskLevel,
    ) -> float:
        """Calculate exit risk penalty (0.0 to 1.0).

        Exit risk = risk that position fills but can't be sold for breakeven+

        Factors:
        - Required move relative to spread/volatility
        - Time horizon (longer = more uncertainty)

        Args:
            candidate: Candidate recommendation dict
            risk: User's risk level (unused but kept for API consistency)

        Returns:
            Penalty value between 0.0 and 1.0
        """
        penalty = 0.0

        # Factor 1: Required price move vs spread
        buy_price = candidate.get("buy_price", 0)
        sell_price = candidate.get("sell_price", 0)
        required_move_pct = (sell_price - buy_price) / buy_price if buy_price > 0 else 0

        # spread_pct roughly proxies recent volatility/range
        spread_pct = candidate.get("_spread_pct", 0.02)
        # If required move > typical spread * 2, add penalty
        if required_move_pct > spread_pct * 2:
            penalty += min(0.3, (required_move_pct - spread_pct * 2) * 5)

        # REMOVED: Trend is now an entry filter (_apply_trend_entry_filter),
        # not an exit penalty. Items in downtrends are filtered out before
        # reaching this point.

        # Factor 2: Long horizon penalty
        hour_offset = candidate.get("hour_offset", 4)
        if hour_offset >= 24:
            penalty += 0.15  # 15% penalty for very long holds
        elif hour_offset >= 12:
            penalty += 0.08  # 8% penalty for 12+ hour holds

        return max(0.0, min(1.0, penalty))

    def _calculate_fill_risk_penalty(
        self,
        candidate: dict,
        risk: RiskLevel,
    ) -> float:
        """Calculate fill risk penalty (0.0 to 1.0).

        Fill risk = risk that capital stays locked because order never fills

        Factors:
        - Inverse of fill probability
        - Volume (low volume = harder to fill)
        - Crowding (near capacity = harder to fill)
        - Hour offset (longer = more capital opportunity cost)

        Args:
            candidate: Candidate recommendation dict
            risk: User's risk level (unused but kept for API consistency)

        Returns:
            Penalty value between 0.0 and 1.0
        """
        penalty = 0.0

        # Factor 1: Inverse fill probability (most direct measure)
        fill_prob = candidate.get("fill_probability", 0.10)
        no_fill_prob = 1 - fill_prob
        penalty += no_fill_prob * 0.5  # 50% of (1 - fill_prob) as base penalty

        # Factor 2: Volume tier penalty
        volume_tier = candidate.get("volume_tier", "Medium")
        volume_penalties = {
            "Very High": 0.0,
            "High": 0.05,
            "Medium": 0.12,
            "Low": 0.25,
        }
        penalty += volume_penalties.get(volume_tier, 0.12)

        # Factor 3: Crowding capacity utilization
        crowding_capacity = candidate.get("crowding_capacity")
        if crowding_capacity is not None and crowding_capacity <= 20:
            penalty += 0.10  # Penalty for items with tight crowding limits

        # Factor 4: Hour offset opportunity cost
        hour_offset = candidate.get("hour_offset", 4)
        # Longer holds = more capital locked up
        penalty += min(0.15, hour_offset / 100)  # 0-15% based on hours

        return max(0.0, min(1.0, penalty))

    def _calculate_slot_hour_efficiency_baseline(
        self,
        candidates: list[dict],
        total_capital: int,
    ) -> float:
        """Calculate portfolio-level baseline for profit-per-slot-hour.

        Uses median of all candidates' profit_per_slot_hour values to create
        a meaningful baseline for penalizing low-efficiency trades in passive mode.

        Args:
            candidates: List of candidate recommendation dicts
            total_capital: Total available capital for allocation

        Returns:
            Median profit_per_slot_hour across candidates, or 0.0 if no valid values
        """
        efficiencies = []
        for cand in candidates:
            fill_prob = cand["fill_probability"]
            profit_per_unit = cand["profit_per_unit"]
            hour_offset = cand["hour_offset"]
            buy_price = cand["buy_price"]

            # Estimate expected profit at max allocation for this candidate
            max_qty = min(cand["max_quantity"], total_capital // buy_price)
            if max_qty < 1:
                continue

            expected_profit = profit_per_unit * max_qty * fill_prob
            profit_per_slot_hour = expected_profit / max(1, hour_offset)

            if profit_per_slot_hour > 0:
                efficiencies.append(profit_per_slot_hour)

        if not efficiencies:
            return 0.0

        # Return median (more robust to outliers than mean)
        efficiencies.sort()
        mid = len(efficiencies) // 2
        if len(efficiencies) % 2 == 0:
            return (efficiencies[mid - 1] + efficiencies[mid]) / 2
        return efficiencies[mid]

    def _calculate_risk_adjusted_score(
        self,
        candidate: dict,
        base_profit: float,
        risk: RiskLevel,
    ) -> float:
        """Calculate risk-adjusted profit after applying risk penalties.

        Combines exit risk and fill risk penalties, weighted by user's risk level.
        Low risk profiles apply higher weights (more conservative).

        Args:
            candidate: Candidate recommendation dict
            base_profit: Raw expected profit before risk adjustments
            risk: User's risk level

        Returns:
            Risk-adjusted profit (base_profit * (1 - total_penalty))
        """
        exit_penalty = self._calculate_exit_risk_penalty(candidate, risk)
        fill_penalty = self._calculate_fill_risk_penalty(candidate, risk)

        exit_weight = self.EXIT_RISK_WEIGHTS.get(risk, 0.25)
        fill_weight = self.FILL_RISK_WEIGHTS.get(risk, 0.20)

        total_penalty = (exit_penalty * exit_weight) + (fill_penalty * fill_weight)
        total_penalty = min(0.70, total_penalty)  # Cap at 70% max penalty

        return base_profit * (1 - total_penalty)

    def _optimize_portfolio(
        self,
        candidates: list[dict],
        total_capital: int,
        num_slots: int,
        risk: RiskLevel,
        style: Optional[TradingStyle] = None,
    ) -> list[dict]:
        """Select optimal portfolio of flips maximizing risk-adjusted returns.

        Strategy:
        - Low risk: Strongly prefer diversification (4-5 smaller flips)
        - Medium risk: Balance diversification and concentration
        - High risk: Allow concentration if EV justifies it
        - Available slots create pressure toward diversification
        - Passive mode: Filter by minimum capital per slot and profit per slot-hour
        """
        if not candidates or num_slots < 1 or total_capital < 1000:
            return []

        # Risk parameters with max single trade percentage caps
        # max_single_pct: hard cap on any single trade (per issue #4)
        # target_pct: ideal % of capital per slot
        # concentration_penalty: how much to penalize going over target
        # min_slots: minimum slots to target for diversification
        risk_params = {
            "low": {
                "max_single_pct": 0.15,
                "target_pct": 0.15,
                "concentration_penalty": 0.5,
                "min_slots": 3,
            },
            "medium": {
                "max_single_pct": 0.30,
                "target_pct": 0.25,
                "concentration_penalty": 0.25,
                "min_slots": 2,
            },
            "high": {
                "max_single_pct": 0.50,
                "target_pct": 0.40,
                "concentration_penalty": 0.1,
                "min_slots": 1,
            },
        }
        params = risk_params.get(risk, risk_params["medium"])

        # Adjust targets based on available slots to encourage diversification
        # More slots should encourage more positions with smaller capital per trade
        if num_slots >= 6:
            # Many slots: reduce max single trade and target significantly
            params["max_single_pct"] = min(params["max_single_pct"], 0.20)
            params["target_pct"] = min(params["target_pct"], 0.15)
            # Aim for at least 4 recommendations (unless high risk)
            params["min_slots"] = max(params["min_slots"], 4 if risk != "high" else 2)
        elif num_slots >= 4:
            # Moderate slots: reduce concentration moderately
            params["max_single_pct"] = min(params["max_single_pct"], 0.25)
            params["target_pct"] = min(params["target_pct"], 0.20)
            # Aim for at least 3 recommendations (unless high risk)
            params["min_slots"] = max(params["min_slots"], 3 if risk != "high" else 2)
        elif num_slots == 3:
            # Three slots: balanced approach between few and moderate
            params["max_single_pct"] = min(params["max_single_pct"], 0.35)
            params["target_pct"] = min(params["target_pct"], 0.30)
            # Aim for at least 2 recommendations (unless high risk)
            params["min_slots"] = max(params["min_slots"], 2 if risk != "high" else 1)
        elif num_slots <= 2:
            # Few slots: allow higher concentration per trade
            params["max_single_pct"] = min(params["max_single_pct"] * 1.5, 0.80)
            params["target_pct"] = min(params["target_pct"] * 1.3, 0.60)
            # With only 1-2 slots, aim to fill at least 1
            params["min_slots"] = 1

        min_slots_target = min(params["min_slots"], num_slots, len(candidates))
        max_capital_per_trade = int(total_capital * params["max_single_pct"])

        # Slot efficiency parameters for passive mode
        # Use logarithmic scaling so threshold grows slowly with capital
        # This prevents high-capital users from being filtered out of viable trades
        # (fixes issue #128: 2B capital returning 0 results while 500M works)
        if style == "passive":
            base_threshold = 5_000_000  # 5M gp minimum viable trade
            capital_factor = math.log10(max(total_capital, 1_000_000) / 1_000_000)
            min_capital_per_slot = base_threshold * (1 + capital_factor * 2)
        else:
            min_capital_per_slot = 0  # No minimum for active/hybrid

        # Filter candidates by minimum capital threshold for passive mode
        if style == "passive" and min_capital_per_slot > 0:
            filtered_candidates = []
            for cand in candidates:
                min_viable_capital = cand["buy_price"] * max(1, cand["max_quantity"])
                # For passive, require at least min_capital_per_slot
                if min_viable_capital >= min_capital_per_slot * 0.8:  # 20% tolerance
                    filtered_candidates.append(cand)
                else:
                    logger.debug(
                        f"Filtered {cand['item_name']} for passive mode: "
                        f"capital {min_viable_capital:,} < "
                        f"threshold {min_capital_per_slot:,.0f}"
                    )
            candidates = filtered_candidates

            if not candidates:
                logger.info(
                    "No candidates meet passive mode capital efficiency threshold "
                    f"(threshold={min_capital_per_slot:,.0f})"
                )
                return []

        # Compute portfolio-level profit-per-slot-hour baseline for passive mode
        slot_hour_baseline = 0.0
        if style == "passive":
            slot_hour_baseline = self._calculate_slot_hour_efficiency_baseline(
                candidates, total_capital
            )

        # Score each candidate at different capital allocations
        scored_options = []
        for cand in candidates:
            buy_price = cand["buy_price"]
            max_qty = cand["max_quantity"]
            profit_per_unit = cand["profit_per_unit"]
            fill_prob = cand["fill_probability"]
            hour_offset = cand["hour_offset"]

            # Try different allocation sizes, but cap at max_single_pct
            for alloc_pct in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                # Skip allocation percentages that exceed risk profile limits
                if alloc_pct > params["max_single_pct"]:
                    continue

                alloc_capital = min(
                    int(total_capital * alloc_pct),
                    buy_price * max_qty,
                    max_capital_per_trade,
                )
                quantity = alloc_capital // buy_price

                if quantity < 1:
                    continue

                capital_used = buy_price * quantity
                expected_profit = int(profit_per_unit * quantity * fill_prob)

                # Apply risk-adjusted scoring (exit risk + fill risk penalties)
                risk_adjusted_profit = self._calculate_risk_adjusted_score(
                    cand, expected_profit, risk
                )

                # Calculate profit per slot-hour for opportunity cost assessment
                profit_per_slot_hour = risk_adjusted_profit / max(1, hour_offset)

                # Concentration penalty: penalize using too much capital in one flip
                concentration_ratio = capital_used / total_capital
                if concentration_ratio > params["target_pct"]:
                    penalty = (concentration_ratio - params["target_pct"]) * params[
                        "concentration_penalty"
                    ]
                    adjusted_profit = risk_adjusted_profit * (1 - penalty)
                else:
                    adjusted_profit = risk_adjusted_profit

                # For passive mode, apply slot efficiency bonus/penalty
                if style == "passive":
                    # Bonus for high capital deployment
                    capital_efficiency = capital_used / min_capital_per_slot
                    if capital_efficiency >= 1.0:
                        # Reward trades that meet or exceed capital threshold
                        efficiency_bonus = min(0.2, (capital_efficiency - 1.0) * 0.1)
                        adjusted_profit *= 1 + efficiency_bonus
                    else:
                        # Penalize low capital deployment
                        efficiency_penalty = (1.0 - capital_efficiency) * 0.3
                        adjusted_profit *= 1 - efficiency_penalty

                    # Penalty for poor profit-per-slot-hour (uses portfolio-level baseline)
                    if slot_hour_baseline > 0:
                        if profit_per_slot_hour < slot_hour_baseline * 0.5:
                            # Penalize trades with poor slot-hour efficiency
                            adjusted_profit *= 0.7

                scored_options.append(
                    {
                        "candidate": cand,
                        "quantity": quantity,
                        "capital_used": capital_used,
                        "expected_profit": expected_profit,
                        "adjusted_profit": adjusted_profit,
                        "profit_per_capital": (
                            adjusted_profit / capital_used if capital_used > 0 else 0
                        ),
                        "profit_per_slot_hour": profit_per_slot_hour,
                    }
                )

        if not scored_options:
            return []

        # Try ILP optimizer first, fallback to greedy if it fails
        selected = self._solve_portfolio_ilp(scored_options, total_capital, num_slots)

        if not selected:
            logger.info("ILP solver returned no solution, falling back to greedy")
            selected = self._greedy_select(
                scored_options,
                total_capital,
                num_slots,
                min_slots_target,
                params,
                max_capital_per_trade,
            )

        # Convert selected options to final recommendation format
        # Collect all fill probabilities for relative confidence calculation
        all_fill_probs = [opt["candidate"]["fill_probability"] for opt in selected]

        recommendations = []
        for opt in selected:
            cand = opt["candidate"]
            fill_prob = cand["fill_probability"]
            rec = {
                "id": self._generate_stable_id(cand["item_id"]),
                "itemId": cand["item_id"],
                "item": cand["item_name"],
                "buyPrice": cand["buy_price"],
                "sellPrice": cand["sell_price"],
                "quantity": opt["quantity"],
                "capitalRequired": opt["capital_used"],
                "expectedProfit": opt["expected_profit"],
                "confidence": cand["confidence"],
                "volumeTier": cand["volume_tier"],
                "trend": cand["trend"],
                "expectedHours": cand["hour_offset"],
                "fillProbability": round(fill_prob, 4),
                "fillConfidence": self._determine_fill_confidence(
                    fill_prob, all_fill_probs
                ),
                "_expectedValue": round(cand["expected_value"], 6),
                # Rationale fields for "Why this item?" UX
                "reason": self._build_reason(cand),
                "isRecommended": True,
            }

            # Add optional volume24h field if available
            if cand.get("volume_24h") is not None:
                rec["volume24h"] = cand["volume_24h"]

            # Add multi-limit strategy metadata if applicable
            if cand.get("is_multi_limit", False):
                base_limit = cand.get("base_buy_limit", 0)
                rec["isMultiLimitStrategy"] = True
                rec["baseBuyLimit"] = base_limit
                rec["stagedBuys"] = self._format_staged_buys(
                    cand["item_name"],
                    base_limit,
                    opt["quantity"],
                    cand["buy_price"],
                    cand["sell_price"],
                )
            else:
                rec["isMultiLimitStrategy"] = False

            recommendations.append(rec)

        # Sort by expected profit descending
        recommendations.sort(key=lambda x: x["expectedProfit"], reverse=True)

        return recommendations

    def _format_staged_buys(
        self,
        item_name: str,
        base_limit: int,
        total_quantity: int,
        buy_price: int,
        sell_price: int,
    ) -> list[dict]:
        """Format staged buy instructions for multi-limit strategies.

        Args:
            item_name: Name of the item
            base_limit: Base GE buy limit
            total_quantity: Total quantity to buy
            buy_price: Buy price per unit
            sell_price: Sell price per unit

        Returns:
            List of staged buy dictionaries
        """
        stages = []

        # First buy: immediately
        first_qty = min(base_limit, total_quantity)
        stages.append(
            {
                "stage": 1,
                "quantity": first_qty,
                "price": buy_price,
                "timing": "now",
                "description": f"Buy {first_qty:,} @ {buy_price:,} gp now",
            }
        )

        # Second buy: after 4-hour reset (if applicable)
        remaining = total_quantity - first_qty
        if remaining > 0:
            stages.append(
                {
                    "stage": 2,
                    "quantity": remaining,
                    "price": buy_price,
                    "timing": "after_4h_reset",
                    "description": (
                        f"Buy {remaining:,} @ {buy_price:,} gp after 4hr reset"
                    ),
                }
            )

        # Sell instruction
        stages.append(
            {
                "stage": 3,
                "quantity": total_quantity,
                "price": sell_price,
                "timing": "passive_target",
                "description": (
                    f"Sell {total_quantity:,} @ {sell_price:,} gp (passive target)"
                ),
            }
        )

        return stages

    def _solve_portfolio_ilp(
        self,
        scored_options: list[dict],
        total_capital: int,
        num_slots: int,
    ) -> list[dict]:
        """Solve portfolio selection using Integer Linear Programming.

        Formulates the problem as a Multiple-Choice Knapsack:
        - Maximize total adjusted_profit
        - Subject to capital budget constraint
        - Subject to slot limit constraint
        - Each item can only be selected once (with one allocation level)

        Args:
            scored_options: List of options with candidate, quantity, capital_used,
                adjusted_profit
            total_capital: Total available capital
            num_slots: Maximum number of slots to fill

        Returns:
            List of selected options, or empty list if solver fails
        """
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
            import numpy as np
        except ImportError:
            logger.warning("scipy.optimize not available, ILP solver disabled")
            return []

        if not scored_options:
            return []

        n = len(scored_options)

        # Build index mapping: item_id -> list of option indices
        item_to_options: dict[int, list[int]] = {}
        for idx, opt in enumerate(scored_options):
            item_id = opt["candidate"]["item_id"]
            if item_id not in item_to_options:
                item_to_options[item_id] = []
            item_to_options[item_id].append(idx)

        # Objective: maximize adjusted_profit (negate for minimization)
        c = np.array([-opt["adjusted_profit"] for opt in scored_options])

        # Build constraint matrices
        A_ub = []
        b_ub = []

        # Constraint 1: Capital budget (sum of capital_used <= total_capital)
        capital_coeffs = np.array([opt["capital_used"] for opt in scored_options])
        A_ub.append(capital_coeffs)
        b_ub.append(total_capital)

        # Constraint 2: Slot limit (count of selected items <= num_slots)
        slot_coeffs = np.ones(n)
        A_ub.append(slot_coeffs)
        b_ub.append(num_slots)

        # Constraint 3: Mutual exclusivity per item (at most one allocation per item)
        for item_id, indices in item_to_options.items():
            row = np.zeros(n)
            row[indices] = 1
            A_ub.append(row)
            b_ub.append(1)

        # Stack all inequality constraints
        A_ub_matrix = np.vstack(A_ub)
        b_ub_array = np.array(b_ub)

        # All constraints are upper bounds (Ax <= b)
        constraints = LinearConstraint(A_ub_matrix, -np.inf, b_ub_array)

        # Variable bounds: 0 <= x <= 1
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        # All variables are binary (integer with bounds [0,1])
        integrality = np.ones(n, dtype=int)

        # Solve with time limit
        try:
            result = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                options={"time_limit": 0.5},  # 500ms limit for solver
            )
        except Exception as e:
            logger.warning(f"ILP solver raised exception: {e}")
            return []

        if not result.success or result.x is None:
            logger.debug(f"ILP solver did not find optimal solution: {result.message}")
            return []

        # Extract selected options
        selected = []
        for idx, val in enumerate(result.x):
            if val > 0.5:  # Binary variable is 1
                selected.append(scored_options[idx])

        return selected

    def _greedy_select(
        self,
        scored_options: list[dict],
        total_capital: int,
        num_slots: int,
        min_slots_target: int,
        params: dict,
        max_capital_per_trade: int,
    ) -> list[dict]:
        """Greedy fallback selection by profit-per-capital.

        Used when ILP solver fails or is unavailable. Sorts options by ROI
        and greedily selects options that fit within constraints.

        Args:
            scored_options: List of scored allocation options
            total_capital: Total available capital
            num_slots: Maximum number of slots to fill
            min_slots_target: Minimum slots to try to fill
            params: Risk parameters dict with target_pct, concentration_penalty
            max_capital_per_trade: Maximum capital allowed per trade

        Returns:
            List of selected options
        """
        # Sort by ROI (profit per capital)
        sorted_options = sorted(
            scored_options, key=lambda x: x["profit_per_capital"], reverse=True
        )

        selected = []
        used_items: set[int] = set()
        remaining_capital = total_capital

        for opt in sorted_options:
            if len(selected) >= num_slots:
                break

            item_id = opt["candidate"]["item_id"]
            if item_id in used_items:
                continue

            if opt["capital_used"] > remaining_capital:
                # Try to fit with remaining capital
                affordable_qty = remaining_capital // opt["candidate"]["buy_price"]
                if affordable_qty < 1:
                    continue
                # Recalculate with affordable quantity
                opt = self._recalc_option(
                    opt, affordable_qty, total_capital, params, max_capital_per_trade
                )

            selected.append(opt)
            used_items.add(item_id)
            remaining_capital -= opt["capital_used"]

        # If we have too few selections for low/medium risk, try to add more
        if len(selected) < min_slots_target and remaining_capital > 1000:
            for opt in sorted_options:
                if len(selected) >= num_slots:
                    break

                item_id = opt["candidate"]["item_id"]
                if item_id in used_items:
                    continue

                affordable_qty = remaining_capital // opt["candidate"]["buy_price"]
                if affordable_qty >= 1:
                    opt = self._recalc_option(
                        opt,
                        affordable_qty,
                        total_capital,
                        params,
                        max_capital_per_trade,
                    )
                    selected.append(opt)
                    used_items.add(item_id)
                    remaining_capital -= opt["capital_used"]

        return selected

    def _recalc_option(
        self,
        opt: dict,
        new_qty: int,
        total_capital: int,
        params: dict,
        max_capital_per_trade: int,
    ) -> dict:
        """Recalculate an option with a new quantity, respecting capital limits."""
        cand = opt["candidate"]
        capital_used = cand["buy_price"] * new_qty

        # Enforce max capital per trade limit
        if capital_used > max_capital_per_trade:
            new_qty = max_capital_per_trade // cand["buy_price"]
            capital_used = cand["buy_price"] * new_qty

        expected_profit = int(
            cand["profit_per_unit"] * new_qty * cand["fill_probability"]
        )

        concentration_ratio = capital_used / total_capital
        if concentration_ratio > params["target_pct"]:
            penalty = (concentration_ratio - params["target_pct"]) * params[
                "concentration_penalty"
            ]
            adjusted_profit = expected_profit * (1 - penalty)
        else:
            adjusted_profit = expected_profit

        return {
            "candidate": cand,
            "quantity": new_qty,
            "capital_used": capital_used,
            "expected_profit": expected_profit,
            "adjusted_profit": adjusted_profit,
            "profit_per_capital": (
                adjusted_profit / capital_used if capital_used > 0 else 0
            ),
        }

    def _adjust_confidence(
        self,
        db_confidence: str,
        pred_age_seconds: float,
    ) -> ConfidenceLevel:
        """Adjust confidence based on data freshness.

        Args:
            db_confidence: Confidence from database
            pred_age_seconds: Age of predictions

        Returns:
            Adjusted confidence level
        """
        # Map DB confidence
        confidence_map = {"high": 2, "medium": 1, "low": 0}
        base = confidence_map.get(db_confidence, 1)

        # Downgrade if data is stale (predictions refresh every 5 min)
        if pred_age_seconds > 360:  # > 6 min = missed a cycle
            base = max(0, base - 1)
        # Fresh data (< 6 min) keeps original confidence

        # Map back to string
        if base >= 2:
            return "high"
        elif base >= 1:
            return "medium"
        else:
            return "low"

    def _check_manipulation_signals(
        self,
        item_id: int,
        buy_price: int,
        spread_pct: float,
        volume_24h: Optional[int] = None,
        volume_1h: Optional[int] = None,
    ) -> tuple[bool, float, list[str]]:
        """Check for manipulation signals on an item.

        Evaluates multiple factors that indicate potential price manipulation:
        - High spread (volatility/illiquidity)
        - Volume concentration (recent pump activity)
        - Low volume for cheap items

        Args:
            item_id: OSRS item ID
            buy_price: Current buy price for the item
            spread_pct: Current spread percentage (high - low) / low
            volume_24h: 24-hour trading volume (optional, fetched if None)
            volume_1h: 1-hour trading volume (optional, fetched if None)

        Returns:
            Tuple of (is_suspicious, risk_score, reasons)
            - is_suspicious: True if item should be filtered out
            - risk_score: 0.0-1.0 manipulation risk score
            - reasons: List of human-readable reasons for the flags
        """
        reasons = []
        risk_score = 0.0

        # Fetch volume data if not provided
        if volume_24h is None:
            volume_24h = self.loader.get_item_volume_24h(item_id)
        if volume_1h is None:
            volume_1h = self.loader.get_item_volume_1h(item_id)

        # Check 1: High spread = volatility/manipulation
        if spread_pct > self.config.max_spread_pct:
            reasons.append(f"High spread ({spread_pct:.1%})")
            risk_score += 0.3

        # Check 2: Volume concentration = possible pump
        if volume_24h and volume_24h > 0 and volume_1h is not None:
            concentration = volume_1h / volume_24h
            if concentration > self.config.max_volume_concentration:
                reasons.append(f"Volume spike ({concentration:.0%} in 1h)")
                risk_score += 0.4

        # Check 3: Low volume for cheap items
        # Expensive items naturally have lower volume, but cheap items should be liquid
        # Only apply if we have volume data (None = unknown, not zero)
        if buy_price < 1000 and volume_24h is not None:  # Items under 1000gp
            if volume_24h < self.config.min_volume_for_low_value:
                reasons.append(
                    f"Low volume for cheap item ({volume_24h:,} < {self.config.min_volume_for_low_value:,})"
                )
                risk_score += 0.35

        # Item is suspicious if risk score exceeds threshold
        is_suspicious = risk_score > 0.5

        if is_suspicious:
            logger.debug(
                f"Manipulation signals detected for item {item_id}: "
                f"risk_score={risk_score:.2f}, reasons={reasons}"
            )

        return (is_suspicious, min(risk_score, 1.0), reasons)

    def _apply_price_stability_filter(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Filter out items with suspicious price/volume patterns (anti-manipulation).

        Uses value-tiered thresholds: expensive items can have larger deviations
        because they're harder to manipulate.

        Args:
            predictions: DataFrame with columns including 'current_high',
                        'price_vs_median_ratio', 'volume_24h'

        Returns:
            Filtered DataFrame with suspicious items removed
        """
        # Skip if stability fields not available (backwards compatibility)
        if 'price_vs_median_ratio' not in predictions.columns:
            logger.warning("price_vs_median_ratio not available, skipping stability filter")
            return predictions

        def passes_stability_check(row) -> bool:
            ratio = row.get('price_vs_median_ratio')
            volume = row.get('volume_24h', 0)
            item_value = row.get('current_high', 0)

            # Skip check if ratio is None/NaN
            if pd.isna(ratio) or ratio is None:
                return True

            # Value-tiered thresholds
            if item_value > 100_000_000:  # >100M (Torva, Tbow, etc.)
                max_deviation = 0.15
                min_volume = 100
            elif item_value > 10_000_000:  # >10M
                max_deviation = 0.12
                min_volume = 500
            elif item_value > 1_000_000:  # >1M
                max_deviation = 0.10
                min_volume = 2_000
            else:  # Cheap items (manipulation targets)
                max_deviation = 0.08
                min_volume = 5_000

            # If price elevated AND volume low → suspicious
            if ratio > (1 + max_deviation) and volume < min_volume:
                return False

            return True

        # Apply filter
        mask = predictions.apply(passes_stability_check, axis=1)
        rejected = predictions[~mask]

        # Log rejections for debugging
        for _, row in rejected.iterrows():
            ratio = row.get('price_vs_median_ratio')
            volume = row.get('volume_24h')
            value = row.get('current_high')
            ratio_str = f"{ratio:.1%}" if ratio is not None else "N/A"
            vol_str = str(volume) if volume is not None else "N/A"
            value_str = f"{value:,.0f}" if value is not None else "N/A"
            logger.info(
                f"Stability filter rejected {row.get('item_name', 'Unknown')}: "
                f"ratio={ratio_str}, vol={vol_str}, value={value_str}"
            )

        return predictions[mask].reset_index(drop=True)

    def _apply_trend_entry_filter(
        self, predictions: pd.DataFrame, style: str
    ) -> pd.DataFrame:
        """Filter out items in downtrends based on trading style (anti-adverse selection).

        Active traders need quick bounces - reject any downward momentum.
        Passive traders bet on mean reversion - only reject crashes.

        Args:
            predictions: DataFrame with columns including 'return_1h', 'return_4h',
                        'return_24h', 'hour_offset'
            style: Trading style ('active', 'hybrid', 'passive')

        Returns:
            Filtered DataFrame with falling items removed
        """
        # Skip if return fields not available (backwards compatibility)
        if 'return_4h' not in predictions.columns:
            logger.warning("return_4h not available, skipping trend entry filter")
            return predictions

        def passes_trend_check(row) -> bool:
            return_1h = row.get('return_1h')
            return_4h = row.get('return_4h')
            return_24h = row.get('return_24h')
            hour_offset = row.get('hour_offset', 4)

            # Skip check if returns are None/NaN
            if pd.isna(return_4h) or return_4h is None:
                return True

            # Determine effective style based on hour_offset
            if hour_offset <= 4:
                effective_style = 'active'
            elif hour_offset <= 12:
                effective_style = 'hybrid'
            else:
                effective_style = 'passive'

            # Override with user's style if more conservative
            if style == 'active':
                effective_style = 'active'
            elif style == 'hybrid' and effective_style == 'passive':
                effective_style = 'hybrid'

            # Apply style-specific thresholds
            if effective_style == 'active':
                # Strict: short-term momentum matters
                r1h = return_1h if not pd.isna(return_1h) else 0
                if return_4h < -0.02 and r1h < -0.01:
                    return False

            elif effective_style == 'hybrid':
                # Moderate: allow small dips, block sustained falls
                if return_4h < -0.04:
                    return False

            else:  # passive
                # Relaxed: only block major crashes
                r24h = return_24h if not pd.isna(return_24h) else 0
                r1h = return_1h if not pd.isna(return_1h) else 0

                if r24h <= -0.08:  # Down 8%+ over 24h
                    return False
                if return_4h < -0.06 and r1h < -0.02:  # Active freefall
                    return False

            return True

        # Apply filter
        mask = predictions.apply(passes_trend_check, axis=1)
        rejected = predictions[~mask]

        # Log rejections for debugging (with safe formatting)
        for _, row in rejected.iterrows():
            r1h = row.get('return_1h')
            r4h = row.get('return_4h')
            r24h = row.get('return_24h')
            r1h_str = "N/A" if r1h is None or pd.isna(r1h) else f"{r1h:.1%}"
            r4h_str = "N/A" if r4h is None or pd.isna(r4h) else f"{r4h:.1%}"
            r24h_str = "N/A" if r24h is None or pd.isna(r24h) else f"{r24h:.1%}"
            logger.info(
                "Trend filter rejected %s: 1h=%s, 4h=%s, 24h=%s",
                row.get('item_name', 'Unknown'),
                r1h_str,
                r4h_str,
                r24h_str
            )

        return predictions[mask].reset_index(drop=True)

    def _determine_volume_tier(self, spread_pct: float) -> str:
        """Determine volume tier from spread percentage.

        Tighter spreads generally indicate higher volume/liquidity.

        Tiers (with crowding limits):
        - Very High (< 1% spread): Unlimited concurrent users
        - High (< 2% spread): 50 concurrent users
        - Medium (< 5% spread): 20 concurrent users
        - Low (>= 5% spread): 10 concurrent users
        """
        if spread_pct < 0.01:  # < 1% spread
            return "Very High"
        elif spread_pct < 0.02:  # < 2% spread
            return "High"
        elif spread_pct < 0.05:  # < 5% spread
            return "Medium"
        else:
            return "Low"

    def _get_crowding_capacity(
        self, item_id: int, volume_1h: Optional[int] = None
    ) -> Optional[int]:
        """Get crowding limit based on actual 1-hour trade volume.

        Volume thresholds:
        - > 50,000: Unlimited (None)
        - > 10,000: 50 concurrent users
        - > 1,000: 20 concurrent users
        - <= 1,000: 10 concurrent users

        Args:
            item_id: OSRS item ID
            volume_1h: Pre-fetched 1h volume (if available)

        Returns:
            Crowding capacity limit or None for unlimited
        """
        # Use pre-fetched volume if provided, otherwise fetch individually
        if volume_1h is None:
            volume_1h = self.loader.get_item_volume_1h(item_id)

        if volume_1h is None:
            # If we can't get volume data, use conservative limit
            return 20

        if volume_1h > 50_000:
            return None  # Unlimited
        elif volume_1h > 10_000:
            return 50
        elif volume_1h > 1_000:
            return 20
        else:
            return 10

    def _determine_fill_confidence(
        self,
        fill_prob: float,
        all_probs: Optional[list[float]] = None,
    ) -> str:
        """Determine fill confidence using hybrid thresholds.

        Uses relative percentiles within batch, with absolute floors
        to prevent misleading labels on low-probability items.

        Args:
            fill_prob: This item's fill probability
            all_probs: All fill probabilities in current batch (for percentile calc)

        Returns:
            "Strong", "Good", or "Fair"
        """
        # Absolute floors - never call very low probs "Strong" or "Good"
        if fill_prob < 0.03:
            return "Fair"
        if fill_prob < 0.05:
            # Can be "Good" but never "Strong"
            if all_probs and len(all_probs) > 1:
                percentile = sum(1 for p in all_probs if p <= fill_prob) / len(
                    all_probs
                )
                if percentile >= 0.50:
                    return "Good"
            return "Fair"

        # Relative percentile logic for probs >= 0.05
        if all_probs and len(all_probs) > 1:
            percentile = sum(1 for p in all_probs if p <= fill_prob) / len(all_probs)
            if percentile >= 0.75:  # Top 25%
                return "Strong"
            elif percentile >= 0.50:  # Top 50%
                return "Good"
            else:
                return "Fair"

        # Fallback to absolute thresholds (single-item queries)
        if fill_prob >= 0.15:
            return "Strong"
        elif fill_prob >= 0.08:
            return "Good"
        else:
            return "Fair"

    def _build_reason(self, candidate: dict) -> str:
        """Build a human-readable reason string for why this item was recommended.

        Format: "{trend} trend, {volume} volume, {hour_offset}h window"

        Args:
            candidate: Candidate dict with trend, volume_tier, and hour_offset

        Returns:
            Reason string explaining the recommendation
        """
        trend = candidate.get("trend", "Stable")
        volume_tier = candidate.get("volume_tier", "Medium")
        hour_offset = candidate.get("hour_offset", 4)

        # Normalize volume tier to lowercase for readability
        volume_desc = volume_tier.lower() if volume_tier else "medium"

        return f"{trend} trend, {volume_desc} volume, {hour_offset}h window"

    def get_recommendation_by_id(self, rec_id: str) -> Optional[dict]:
        """Get recommendation by its ID.

        First checks in-memory cache, then regenerates from database
        if not found. This supports multi-worker deployments where the
        recommendation may have been generated by a different process.

        Args:
            rec_id: Recommendation ID

        Returns:
            Recommendation dict or None
        """
        # Fast path: check cache first
        cached = self.store.get_by_id(rec_id)
        if cached is not None:
            return cached

        # Cache miss: try to rehydrate from database (fixes issue #130)
        item_id = self.store.parse_rec_id(rec_id)
        if item_id is None:
            return None  # Invalid rec_id format

        # Regenerate recommendation from database with default context
        rec = self._generate_item_recommendation(
            item_id=item_id,
            capital=100_000_000,  # Default values for regeneration
            risk="medium",
            style="hybrid",
            slots=4,
        )

        # Only return if recommended
        if rec is None or not rec.get("isRecommended", True):
            return None

        # Ensure correct rec_id (preserves original even if hour bucket changed)
        rec["id"] = rec_id

        # Cache for subsequent requests on this worker
        self.store.by_id[rec_id] = (rec, datetime.now(timezone.utc))

        return rec

    def get_recommendation_by_item_id(
        self,
        item_id: int,
        capital: Optional[int] = None,
        risk: Optional[RiskLevel] = None,
        style: Optional[TradingStyle] = None,
        slots: Optional[int] = None,
        include_price_history: bool = False,
    ) -> Optional[dict]:
        """Get recommendation by item ID, with optional user context.

        If user context is provided, generates a fresh recommendation.
        Otherwise, returns cached recommendation from store.

        Args:
            item_id: OSRS item ID
            capital: User's available capital (optional)
            risk: User's risk tolerance (optional)
            style: User's trading style (optional)
            slots: User's available GE slots (optional)
            include_price_history: Whether to include 24h price history for charts

        Returns:
            Recommendation dict or None if item not recommended
        """
        # If user context provided, generate fresh recommendation
        if capital is not None and risk is not None and style is not None:
            rec = self._generate_item_recommendation(
                item_id, capital, risk, style, slots or 4
            )
        else:
            # Otherwise return from cache
            rec = self.store.get_by_item_id(item_id)

        # Add price history if requested and recommendation exists
        if rec is not None and include_price_history:
            price_history = self.loader.get_price_history(item_id, hours=24)
            if price_history:
                rec["priceHistory"] = price_history

        return rec

    def _generate_item_recommendation(
        self,
        item_id: int,
        capital: int,
        risk: RiskLevel,
        style: TradingStyle,
        slots: int,
    ) -> Optional[dict]:
        """Generate a recommendation for a specific item with user context.

        Args:
            item_id: OSRS item ID
            capital: User's available capital
            risk: User's risk tolerance
            style: User's trading style
            slots: User's available GE slots

        Returns:
            Recommendation dict with explanation, or dict with isRecommended=False
        """
        # Get filtering parameters based on style/risk
        min_ev = self.EV_THRESHOLDS.get(risk, 0.005)
        min_hour, max_hour = self.HOUR_RANGES.get(style, (1, 48))
        min_fill = self.FILL_PROB_MINIMUMS.get(risk, 0.05)

        # Fetch predictions for this specific item
        predictions_df = self.loader.get_predictions_for_item(item_id)

        if predictions_df.empty:
            return {
                "itemId": item_id,
                "isRecommended": False,
                "reason": "No predictions available for this item",
            }

        # Get the item name
        item_name = str(predictions_df.iloc[0]["item_name"])

        # Filter by user's style/risk preferences
        # Note: No upper bound on fill_probability - high confidence is good
        filtered = predictions_df[
            (predictions_df["hour_offset"] >= min_hour)
            & (predictions_df["hour_offset"] <= max_hour)
            & (predictions_df["fill_probability"] >= min_fill)
            & (predictions_df["expected_value"] >= min_ev)
        ]

        if filtered.empty:
            # Find the closest match to explain why
            best_any = predictions_df.loc[predictions_df["expected_value"].idxmax()]
            buy_price = (
                int(best_any["buy_price"]) if best_any["buy_price"] is not None else 0
            )
            sell_price = (
                int(best_any["sell_price"]) if best_any["sell_price"] is not None else 0
            )
            spread = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
            ev = (
                float(best_any["expected_value"])
                if best_any["expected_value"] is not None
                else 0
            )

            # Determine reason
            if ev < min_ev:
                reason = (
                    f"Expected value ({ev:.2%}) below minimum threshold ({min_ev:.2%})"
                )
            elif spread < 0.015:
                reason = f"Spread too low ({spread:.1%}) - below minimum threshold"
            else:
                fill_prob = (
                    float(best_any["fill_probability"])
                    if best_any["fill_probability"] is not None
                    else 0
                )
                if fill_prob < min_fill:
                    reason = (
                        f"Fill probability ({fill_prob:.1%}) too low for {risk} risk"
                    )
                else:
                    hour_offset = int(best_any["hour_offset"])
                    reason = (
                        f"Time window ({hour_offset}h) doesn't match "
                        f"{style} style ({min_hour}-{max_hour}h)"
                    )

            return {
                "itemId": item_id,
                "item": item_name,
                "isRecommended": False,
                "reason": reason,
                "currentBuyPrice": buy_price,
                "currentSellPrice": sell_price,
                "spread": round(spread, 4),
            }

        # Get the best prediction for this item matching user preferences
        best = filtered.loc[filtered["expected_value"].idxmax()]
        pred_age = self.loader.get_prediction_age_seconds()

        # Build candidate
        candidate = self._build_candidate(best, capital, pred_age)
        if not candidate:
            return {
                "itemId": item_id,
                "item": item_name,
                "isRecommended": False,
                "reason": "Unable to build viable trade with available capital",
            }

        # Format as recommendation (using full capital for single item)
        buy_price = candidate["buy_price"]
        max_quantity = min(candidate["max_quantity"], capital // buy_price)

        if max_quantity < 1:
            return {
                "itemId": item_id,
                "item": item_name,
                "isRecommended": False,
                "reason": f"Insufficient capital (need at least {buy_price:,} gp)",
                "currentBuyPrice": buy_price,
                "currentSellPrice": candidate["sell_price"],
            }

        capital_used = buy_price * max_quantity
        fill_prob = candidate["fill_probability"]
        expected_profit = int(candidate["profit_per_unit"] * max_quantity * fill_prob)

        # Build reason text
        volume_desc = candidate["volume_tier"]
        trend_desc = candidate["trend"]
        hour_desc = f"{candidate['hour_offset']}h window"
        reason = f"{trend_desc} trend, {volume_desc.lower()} volume, {hour_desc}"

        recommendation = {
            "id": self._generate_stable_id(item_id),
            "itemId": item_id,
            "item": item_name,
            "buyPrice": buy_price,
            "sellPrice": candidate["sell_price"],
            "quantity": max_quantity,
            "capitalRequired": capital_used,
            "expectedProfit": expected_profit,
            "confidence": candidate["confidence"],
            "volumeTier": candidate["volume_tier"],
            "trend": candidate["trend"],
            "expectedHours": candidate["hour_offset"],
            "reason": reason,
            "fillProbability": round(fill_prob, 4),
            "fillConfidence": self._determine_fill_confidence(fill_prob, None),
            "_expectedValue": round(candidate["expected_value"], 6),
        }

        # Add optional volume24h field if available
        if candidate.get("volume_24h") is not None:
            recommendation["volume24h"] = candidate["volume_24h"]

        # Store it
        self.store.store(recommendation)

        return recommendation

    def search_items_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Search for items by name.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching items with item_id and item_name
        """
        return self.loader.search_items_by_name(query, limit)

    def get_recommendation_by_item_name(
        self,
        item_name: str,
        capital: int,
        risk: RiskLevel,
        style: TradingStyle,
        slots: int,
    ) -> Optional[dict]:
        """Get recommendation by item name (with fuzzy matching).

        Args:
            item_name: Item name to search for
            capital: User's available capital
            risk: User's risk tolerance
            style: User's trading style
            slots: User's available GE slots

        Returns:
            Recommendation dict or error dict
        """
        # Search for matching items
        matches = self.search_items_by_name(item_name, limit=5)

        if not matches:
            return {
                "error": "Item not found",
                "query": item_name,
                "suggestions": [],
            }

        # Use first match (best match due to sorting)
        item_id = matches[0]["item_id"]

        # If we have multiple matches, include suggestions
        if len(matches) > 1:
            match_names = [m["item_name"] for m in matches]
            logger.info(f"Multiple matches for '{item_name}': {match_names}")

        # Generate recommendation for the matched item
        rec = self._generate_item_recommendation(item_id, capital, risk, style, slots)

        # If no exact match, include suggestions in error response
        if rec and not rec.get("isRecommended", True):
            if len(matches) > 1:
                rec["suggestions"] = [m["item_name"] for m in matches[1:]]

        return rec

    def get_prediction_for_item(self, item_id: int) -> Optional[dict]:
        """Get full prediction details for a specific item.

        Args:
            item_id: OSRS item ID

        Returns:
            Dict with all predictions for the item
        """
        df = self.loader.get_predictions_for_item(item_id)
        if df.empty:
            return None

        # Get the best configuration
        best = df.loc[df["expected_value"].idxmax()]

        return {
            "item_id": item_id,
            "item_name": best["item_name"],
            "best_config": {
                "hour_offset": int(best["hour_offset"]),
                "offset_pct": float(best["offset_pct"]),
            },
            "fill_probability": float(best["fill_probability"]),
            "expected_value": float(best["expected_value"]),
            "buy_price": int(best["buy_price"]),
            "sell_price": int(best["sell_price"]),
            "confidence": best["confidence"],
            "all_predictions": df.to_dict("records"),
        }

    def get_item_price_lookup(
        self,
        item_id: int,
        side: Literal["buy", "sell"] = "buy",
        window_hours: int = 24,
        offset_pct: Optional[float] = None,
        include_price_history: bool = False,
    ) -> Optional[dict]:
        """Get price lookup for any item.

        Unlike recommendations, this returns data for any item with predictions,
        even if it doesn't meet recommendation thresholds. This allows users
        to look up prices for items they want to trade regardless of EV.

        Args:
            item_id: OSRS item ID
            side: Trade side - "buy" or "sell"
            window_hours: Target time window in hours (1-48)
            offset_pct: Optional target offset percentage (0.01-0.03)
            include_price_history: Whether to include 24h price history for charts

        Returns:
            Dict with price info or None if item not found
        """
        # Fetch predictions for this item
        predictions_df = self.loader.get_predictions_for_item(item_id)

        if predictions_df.empty:
            return None

        # Get item metadata
        item_name = str(predictions_df.iloc[0]["item_name"])
        volume_24h = self.loader.get_item_volume_24h(item_id)
        trend = self.loader.get_item_trend(item_id)

        # Filter by window - find closest hour_offset to requested window
        # First, try exact match or closest match
        df = predictions_df.copy()

        # If offset_pct specified, filter to that offset
        if offset_pct is not None:
            df = df[df["offset_pct"] == offset_pct]
            if df.empty:
                # Try closest offset if exact not found
                available_offsets = predictions_df["offset_pct"].unique()
                closest_offset = min(
                    available_offsets, key=lambda x: abs(x - offset_pct)
                )
                df = predictions_df[predictions_df["offset_pct"] == closest_offset]

        # Find the prediction with hour_offset closest to requested window
        df["hour_diff"] = abs(df["hour_offset"] - window_hours)
        best_row = df.loc[df["hour_diff"].idxmin()]

        # Extract prediction data
        hour_offset = int(best_row["hour_offset"])
        actual_offset_pct = float(best_row["offset_pct"])
        fill_prob = float(best_row["fill_probability"])
        ev = float(best_row["expected_value"])

        # Get prices based on side
        if side == "buy":
            recommended_price = int(best_row["buy_price"])
            current_market_price = (
                int(best_row["current_high"])
                if best_row["current_high"]
                else recommended_price
            )
        else:  # sell
            recommended_price = int(best_row["sell_price"])
            current_market_price = (
                int(best_row["current_low"])
                if best_row["current_low"]
                else recommended_price
            )

        # Determine if this would meet recommendation thresholds
        # Using medium risk defaults as reference
        min_ev = self.EV_THRESHOLDS.get("medium", 0.005)
        min_fill = self.FILL_PROB_MINIMUMS.get("medium", 0.05)
        is_recommended = fill_prob >= min_fill and ev >= min_ev

        # Build warning message if below thresholds
        warning = None
        warnings = []
        if fill_prob < min_fill:
            warnings.append(f"Low fill probability ({fill_prob:.1%})")
        if ev < min_ev:
            warnings.append(f"Low expected value ({ev:.2%})")
        if warnings:
            warning = " - ".join(warnings) + ". Consider waiting for better conditions."

        # Calculate flip metrics (issue #129)
        buy_price = int(best_row["buy_price"])
        sell_price = int(best_row["sell_price"])
        tax_per_unit = self._calculate_tax_per_unit(sell_price)
        margin_gp = sell_price - buy_price - tax_per_unit
        margin_pct = margin_gp / buy_price if buy_price > 0 else 0

        # Get buy limit
        buy_limit = self.loader.get_item_buy_limit(item_id)

        result = {
            "itemId": item_id,
            "itemName": item_name,
            "side": side,
            "recommendedPrice": recommended_price,
            "currentMarketPrice": current_market_price,
            "offsetPercent": round(actual_offset_pct, 4),
            "fillProbability": round(fill_prob, 4),
            "expectedValue": round(ev, 6),
            "timeWindowHours": hour_offset,
            "volume24h": volume_24h,
            "trend": trend,
            "isRecommended": is_recommended,
            "warning": warning,
            # Flip metrics (issue #129)
            "marginGp": margin_gp,
            "marginPercent": round(margin_pct, 4),
            "buyLimit": buy_limit,
        }

        # Include price history if requested
        if include_price_history:
            price_history = self.loader.get_price_history(item_id, hours=24)
            if price_history:
                result["priceHistory"] = price_history

        return result

    def health_check(self) -> dict:
        """Check engine health.

        Returns:
            Health status dictionary
        """
        checks = []

        # Database/loader check
        loader_health = self.loader.health_check()
        checks.append(loader_health)

        # Model registry check
        model_stats = self.loader.get_model_registry_stats()
        model_check = {
            "status": "ok" if model_stats["active"] > 0 else "warning",
            "component": "model_registry",
            "message": f"{model_stats['active']} active, {model_stats['deprecated']} deprecated models",
            "stats": model_stats,
        }
        if model_stats["active"] == 0 and model_stats["total"] > 0:
            model_check["status"] = "warning"
            model_check["message"] = (
                "No active models - all predictions may be deprecated"
            )
        checks.append(model_check)

        # Overall status
        statuses = [c.get("status") for c in checks]
        if "error" in statuses:
            overall = "error"
        elif "warning" in statuses:
            overall = "warning"
        else:
            overall = "ok"

        # Get crowding stats
        crowding_stats = self.crowding_tracker.get_stats()

        return {
            "status": overall,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommendation_store_size": len(self.store.by_id),
            "crowding_stats": crowding_stats,
            "model_registry_stats": model_stats,
        }

    def get_freshness_metadata(self) -> dict:
        """Get freshness metadata for predictions.

        Returns:
            Dictionary with freshness metadata including:
            - inference_at: ISO 8601 timestamp of latest predictions
            - inference_age_seconds: Age of predictions in seconds
            - stale: Boolean indicating if predictions exceed threshold
            - stale_threshold_seconds: The threshold used for staleness check
        """
        age = self.loader.get_prediction_age_seconds()
        latest = self.loader.get_latest_timestamp()
        threshold = self.config.prediction_stale_seconds

        return {
            "inference_at": latest.isoformat() if latest else None,
            "inference_age_seconds": round(age, 1) if age != float("inf") else None,
            "stale": age > threshold if age != float("inf") else True,
            "stale_threshold_seconds": threshold,
        }

    def evaluate_active_order(
        self,
        item_id: int,
        order_type: Literal["buy", "sell"],
        user_price: int,
        quantity: int,
        time_elapsed_minutes: int,
        user_id: Optional[str] = None,
    ) -> dict:
        """Evaluate an active order and recommend action.

        Delegates to OrderAdvisor for the decision logic.

        Args:
            item_id: OSRS item ID
            order_type: 'buy' or 'sell'
            user_price: User's current order price
            quantity: Order quantity
            time_elapsed_minutes: Minutes since order was placed
            user_id: Optional hashed user ID

        Returns:
            Dict with action, confidence, recommendations, and reasoning
        """
        from .order_advisor import OrderAdvisor

        advisor = OrderAdvisor(loader=self.loader, engine=self)
        return advisor.evaluate_order(
            item_id=item_id,
            order_type=order_type,
            user_price=user_price,
            quantity=quantity,
            time_elapsed_minutes=time_elapsed_minutes,
            user_id=user_id,
        )

    def close(self):
        """Clean up resources."""
        self.loader.close()
        self.store.clear()
        self.crowding_tracker.clear()
