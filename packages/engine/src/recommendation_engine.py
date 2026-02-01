"""Recommendation engine that transforms raw predictions into actionable trades.

This engine reads pre-computed predictions from the Ampere server's database
and applies user constraints (capital, style, risk, slots) to generate
optimized trade recommendations for the Discord bot.
"""

import logging
import math
import random
from datetime import datetime, timezone
from typing import Literal, Optional, cast

import pandas as pd

from .config import Config
from .crowding import create_crowding_tracker
from .prediction_loader import PredictionLoader
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
    ):
        """Initialize recommendation engine.

        Args:
            db_connection_string: PostgreSQL connection string to Ampere server
            config: Optional configuration object
        """
        self.config = config or Config()
        self.loader = PredictionLoader(
            db_connection_string,
            pool_size=self.config.db_pool_size,
            preferred_model_id=self.config.preferred_model_id,
            config=self.config,
        )
        self.store = RecommendationStore(ttl_seconds=900)

        # Create crowding tracker (Redis if configured, otherwise in-memory)
        self.crowding_tracker = create_crowding_tracker(
            redis_url=self.config.redis_url or None,
            fallback_to_memory=self.config.redis_fallback_to_memory,
        )

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

    # ========== Public API ==========

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
        predictions_df = cast(pd.DataFrame, predictions_df[~predictions_df["item_id"].isin(excluded_items)])

        # Get prediction age for confidence adjustment
        pred_age = self.loader.get_prediction_age_seconds()

        # Batch fetch all per-item data upfront (eliminates N+1 queries)
        candidate_item_ids = predictions_df["item_id"].unique().tolist()
        buy_limits = self.loader.get_batch_buy_limits(candidate_item_ids)
        volumes_24h = self.loader.get_batch_volumes_24h(candidate_item_ids)
        volumes_1h = self.loader.get_batch_volumes_1h(candidate_item_ids)
        trends = self.loader.get_batch_trends(candidate_item_ids)

        # Apply liquidity filter (anti-manipulation: filter items where buy_limit >> volume)
        predictions_df = cast(pd.DataFrame, self._apply_liquidity_filter(predictions_df, buy_limits, volumes_24h))
        if predictions_df.empty:
            logger.info("All predictions filtered by liquidity check")
            return []

        # ========== Vectorized Candidate Building (Phase 4) ==========
        # Apply exclusions first (vectorized)
        if exclude_ids:
            predictions_df['_rec_id'] = predictions_df['item_id'].apply(self._generate_stable_id)
            predictions_df = predictions_df[~predictions_df['_rec_id'].isin(exclude_ids)]
            predictions_df = predictions_df.drop(columns=['_rec_id'])

            if predictions_df.empty:
                logger.info("All predictions filtered by exclusion list")
                return []

        # Vectorized candidate building pipeline
        df = cast(pd.DataFrame, self._filter_valid_candidates(cast(pd.DataFrame, predictions_df)))
        if df.empty:
            logger.info("All predictions filtered by validation checks")
            return []

        df = self._apply_price_buffer_vectorized(df)

        # Must filter instant-fill AFTER price buffer (checking final prices)
        df = self._filter_instant_fill_vectorized(df)
        if df.empty:
            logger.info("All predictions filtered by instant-fill check")
            return []

        df = self._calculate_quantities_vectorized(df, remaining_capital, style, buy_limits)
        if df.empty:
            logger.info("All predictions filtered by quantity checks")
            return []

        # Check manipulation signals and filter suspicious items
        suspicious_mask = self._check_manipulation_signals_vectorized(df, volumes_24h, volumes_1h)
        df = cast(pd.DataFrame, df[~suspicious_mask])
        if df.empty:
            logger.info("All predictions filtered by manipulation checks")
            return []

        # Enrich with metadata (tax, profit, confidence, etc.)
        df = cast(pd.DataFrame, self._enrich_metadata_vectorized(df, pred_age, volumes_24h, volumes_1h, trends))

        # Convert to list of dicts (candidate format expected by downstream code)
        candidates = df.to_dict('records')

        if not candidates:
            logger.info("No viable candidates after filtering")
            return []

        # Filter out crowded items
        candidates = self.crowding_tracker.filter_crowded_items(candidates)

        if not candidates:
            logger.info("No candidates available after crowding filter")
            return []

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
        predictions_df = cast(pd.DataFrame, predictions_df[~predictions_df["item_id"].isin(excluded_items)])

        # Get prediction age for confidence adjustment
        pred_age = self.loader.get_prediction_age_seconds()

        # Batch fetch all per-item data upfront (eliminates N+1 queries)
        candidate_item_ids = predictions_df["item_id"].unique().tolist()
        buy_limits = self.loader.get_batch_buy_limits(candidate_item_ids)
        volumes_24h = self.loader.get_batch_volumes_24h(candidate_item_ids)
        volumes_1h = self.loader.get_batch_volumes_1h(candidate_item_ids)
        trends = self.loader.get_batch_trends(candidate_item_ids)

        # Apply liquidity filter (anti-manipulation: filter items where buy_limit >> volume)
        predictions_df = cast(pd.DataFrame, self._apply_liquidity_filter(predictions_df, buy_limits, volumes_24h))
        if predictions_df.empty:
            logger.info("All predictions filtered by liquidity check for get_all")
            return []

        # ========== Vectorized Candidate Building (Phase 4) ==========
        # Vectorized candidate building pipeline
        df = self._filter_valid_candidates(predictions_df)
        if df.empty:
            logger.info("All predictions filtered by validation checks for get_all")
            return []

        df = self._apply_price_buffer_vectorized(df)

        # Must filter instant-fill AFTER price buffer (checking final prices)
        df = self._filter_instant_fill_vectorized(df)
        if df.empty:
            logger.info("All predictions filtered by instant-fill check for get_all")
            return []

        df = self._calculate_quantities_vectorized(df, remaining_capital, style, buy_limits)
        if df.empty:
            logger.info("All predictions filtered by quantity checks for get_all")
            return []

        # Check manipulation signals and filter suspicious items
        suspicious_mask = self._check_manipulation_signals_vectorized(df, volumes_24h, volumes_1h)
        df = cast(pd.DataFrame, df[~suspicious_mask])
        if df.empty:
            logger.info("All predictions filtered by manipulation checks for get_all")
            return []

        # Enrich with metadata (tax, profit, confidence, etc.)
        df = cast(pd.DataFrame, self._enrich_metadata_vectorized(df, pred_age, volumes_24h, volumes_1h, trends))

        # Convert to list of dicts (candidate format expected by downstream code)
        candidates = df.to_dict('records')

        if not candidates:
            logger.info("No viable candidates after filtering for get_all")
            return []

        # Filter out crowded items
        candidates = self.crowding_tracker.filter_crowded_items(candidates)

        if not candidates:
            logger.info("No candidates available after crowding filter for get_all")
            return []

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
                "whyChips": self.generate_why_chips(cand),
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
        """Build a candidate recommendation using vectorized pipeline.

        Thin wrapper for single-item edge cases that reuses the vectorized methods.
        Converts single row to DataFrame, processes through vectorized pipeline,
        and returns the result.

        Args:
            row: DataFrame row with prediction data
            max_capital: Maximum capital available for this recommendation
            pred_age_seconds: Age of prediction in seconds (for confidence adjustment)
            style: Trading style (passive, hybrid, active)
            buy_limits: Pre-fetched buy limits dict. If None, fetches for this item.
            volumes_24h: Pre-fetched 24h volumes dict. If None, fetches for this item.
            volumes_1h: Pre-fetched 1h volumes dict. If None, fetches for this item.
            trends: Pre-fetched trends dict. If None, fetches for this item.

        Returns:
            Candidate dict or None if filtered out
        """
        try:
            # Convert single row to DataFrame
            df = pd.DataFrame([row])

            # Get item_id for data fetching
            item_id = int(row["item_id"]) if row["item_id"] is not None else 0

            # Fetch missing data if not provided
            if buy_limits is None:
                buy_limits = {item_id: self.loader.get_item_buy_limit(item_id) or 0}
            if volumes_24h is None:
                volumes_24h = {item_id: self.loader.get_item_volume_24h(item_id) or 0}
            if volumes_1h is None:
                volumes_1h = {item_id: self.loader.get_item_volume_1h(item_id) or 0}
            if trends is None:
                trends = {item_id: self.loader.get_item_trend(item_id)}

            # Run through vectorized pipeline
            df = self._filter_valid_candidates(df)
            if df.empty:
                return None

            df = self._apply_price_buffer_vectorized(df)
            df = self._filter_instant_fill_vectorized(df)
            if df.empty:
                return None

            df = self._calculate_quantities_vectorized(df, max_capital, style, buy_limits)
            if df.empty:
                return None

            # Check manipulation signals
            suspicious_mask = self._check_manipulation_signals_vectorized(df, volumes_24h, volumes_1h)
            df = df[~suspicious_mask]
            if df.empty:
                return None

            # Enrich with metadata
            df = cast(pd.DataFrame, self._enrich_metadata_vectorized(cast(pd.DataFrame, df), pred_age_seconds, volumes_24h, volumes_1h, trends))

            # Convert to dict and return first (and only) result
            candidates = df.to_dict('records')
            return candidates[0] if candidates else None

        except (ValueError, TypeError, KeyError) as e:
            logger.debug(f"Error building candidate: {e}")
            return None

    # ========== Vectorized Candidate Building Methods ==========
    # These methods replace the iterrows() loop in get_recommendations()
    # and get_all_recommendations() for 30-50% performance improvement.

    # ========== Vectorized Candidate Pipeline ==========

    def _filter_valid_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all basic validation filters that would return None in _build_candidate.

        Filters applied (in order):
        1. Remove rows with None/NaN prices (buy_price, sell_price)
        2. Remove rows with price <= 0
        3. Remove rows with missing/NaN current_high

        NOTE: instant-fill check (buy_price >= current_high) is handled separately
        after price buffer is applied in _apply_price_buffer_vectorized().

        Args:
            df: DataFrame with prediction data

        Returns:
            Filtered DataFrame with only valid candidates
        """
        initial_count = len(df)

        # Filter 1: Remove None/NaN prices
        df = df.dropna(subset=['buy_price', 'sell_price'])

        # Filter 2: Remove non-positive prices
        df = cast(pd.DataFrame, df[(df['buy_price'] > 0) & (df['sell_price'] > 0)])

        # Filter 3: Remove missing/NaN current_high (fail closed)
        df = df.dropna(subset=['current_high'])

        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            logger.debug(
                f"Validation filters removed {filtered_count} invalid candidates "
                f"({initial_count} -> {len(df)})"
            )

        # Ensure prices are integers (OSRS prices are whole GP)
        # Some models (e.g. PatchTST) write fractional prices to the DB
        for col in ['buy_price', 'sell_price', 'current_high', 'current_low']:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def _apply_price_buffer_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply random price buffer to all candidates (vectorized version).

        The buffer moves prices toward market: buy UP, sell DOWN.
        This reduces price competition by randomizing the exact price points.

        Args:
            df: DataFrame with buy_price and sell_price columns

        Returns:
            DataFrame with buffered prices (modifies buy_price and sell_price in-place)
        """
        if not self.config.price_buffer_enabled or df.empty:
            return df

        # Calculate margins
        df = df.copy()  # Avoid modifying original
        df['_margin'] = df['sell_price'] - df['buy_price']

        # Only apply buffer where margin > 0 AND min_buffer >= 1gp
        min_buffer_threshold = df['_margin'] * (self.config.price_buffer_min_pct / 100.0)
        applicable_mask = (df['_margin'] > 0) & (min_buffer_threshold >= 1.0)

        if applicable_mask.sum() == 0:
            # No candidates meet criteria for buffering
            df = df.drop(columns=['_margin'])
            return df

        # Generate random buffer percentages (vectorized)
        import numpy as np  # type: ignore[import-untyped]
        buffer_pcts = np.random.uniform(
            self.config.price_buffer_min_pct,
            self.config.price_buffer_max_pct,
            size=len(df)
        )

        # Calculate buffer amounts in gp
        df['_buffer_gp'] = (df['_margin'] * (buffer_pcts / 100.0)).astype(int)

        # Apply buffer only where applicable
        df.loc[applicable_mask, 'buy_price'] = (
            df.loc[applicable_mask, 'buy_price'] + df.loc[applicable_mask, '_buffer_gp']
        ).astype(int)
        df.loc[applicable_mask, 'sell_price'] = (
            df.loc[applicable_mask, 'sell_price'] - df.loc[applicable_mask, '_buffer_gp']
        ).astype(int)

        # Ensure no crossover (buy >= sell) - revert to original if crossed
        crossover_mask = df['buy_price'] >= df['sell_price']
        if crossover_mask.sum() > 0:
            # Revert crossed prices to original (before buffer)
            df.loc[crossover_mask, 'buy_price'] = (
                df.loc[crossover_mask, 'buy_price'] - df.loc[crossover_mask, '_buffer_gp']
            ).astype(int)
            df.loc[crossover_mask, 'sell_price'] = (
                df.loc[crossover_mask, 'sell_price'] + df.loc[crossover_mask, '_buffer_gp']
            ).astype(int)

        # Clean up temporary columns
        df = df.drop(columns=['_margin', '_buffer_gp'])

        return df

    def _filter_instant_fill_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out instant-fill candidates (vectorized version).

        Instant-fill = buy_price >= current_high (order would fill immediately).
        This must run AFTER price buffer to check the final buy prices.

        Args:
            df: DataFrame with buy_price and current_high columns

        Returns:
            DataFrame with instant-fill candidates removed
        """
        if df.empty:
            return df

        # Block buy_price >= current_high (would instant-fill)
        instant_fill_mask = df['buy_price'] >= df['current_high']
        if instant_fill_mask.sum() > 0:
            logger.debug(
                f"Blocking {instant_fill_mask.sum()} instant-fill candidates "
                f"(buy_price >= current_high)"
            )
            df = cast(pd.DataFrame, df[~instant_fill_mask])

        return df

    def _calculate_quantities_vectorized(
        self,
        df: pd.DataFrame,
        max_capital: int,
        style: TradingStyle,
        buy_limits: dict[int, int],
    ) -> pd.DataFrame:
        """Calculate max quantities for all candidates (vectorized version).

        Applies the same logic as _build_candidate for quantity calculation:
        1. Max quantity by capital = max_capital // buy_price
        2. Map buy limits from pre-fetched dict
        3. Calculate effective limits based on style and hour_offset
        4. Take minimum of capital-based and limit-based quantities
        5. Filter out candidates with max_quantity < 1

        Args:
            df: DataFrame with buy_price, hour_offset, and item_id columns
            max_capital: Maximum capital available per recommendation
            style: Trading style (passive allows multi-limit strategies)
            buy_limits: Pre-fetched dict mapping item_id -> base_buy_limit

        Returns:
            DataFrame with max_quantity, is_multi_limit, and base_buy_limit columns
        """
        if df.empty:
            return df

        df = df.copy()

        # Step 1: Map buy limits
        df['base_buy_limit'] = df['item_id'].map(buy_limits)

        # Step 2: Calculate effective limits based on style and hour_offset
        # Vectorize the get_effective_buy_limit() logic
        import numpy as np  # type: ignore[import-untyped]

        # Apply conservative fallback for missing or zero buy limits
        # Original logic: if base_buy_limit and base_buy_limit > 0
        # This means both None and 0 trigger the fallback to 1000
        has_valid_limit = df['base_buy_limit'].notna() & (df['base_buy_limit'] > 0)

        if style == 'passive':
            # For passive: multiplier based on 4-hour resets, capped at 4x
            df['_limit_resets'] = df['hour_offset'] // 4
            df['_multiplier'] = np.minimum(df['_limit_resets'] + 1, 4)

            # effective_limit = base_limit * multiplier (where base_limit is valid)
            # For invalid limits (None or 0), use fallback of 1000
            df['effective_limit'] = 1000  # Default fallback
            df.loc[has_valid_limit, 'effective_limit'] = (
                df.loc[has_valid_limit, 'base_buy_limit'] * df.loc[has_valid_limit, '_multiplier']
            ).astype(int)
            df['is_multi_limit'] = False
            df.loc[has_valid_limit, 'is_multi_limit'] = df.loc[has_valid_limit, '_multiplier'] > 1

            # Clean up temp columns
            df = df.drop(columns=['_limit_resets', '_multiplier'])
        else:
            # For hybrid/active: effective_limit = base_limit (no multiplier)
            # For invalid limits (None or 0), use fallback of 1000
            df['effective_limit'] = 1000  # Default fallback
            df.loc[has_valid_limit, 'effective_limit'] = df.loc[has_valid_limit, 'base_buy_limit'].astype(int)
            df['is_multi_limit'] = False

        # Step 4: Calculate max quantity by capital
        df['max_qty_by_capital'] = (max_capital // df['buy_price']).astype(int)

        # Step 5: Take minimum of capital-based and limit-based
        df['max_quantity'] = df[['max_qty_by_capital', 'effective_limit']].min(axis=1)  # type: ignore[call-overload]

        # Step 6: Filter out zero/negative quantity candidates
        zero_qty_count = (df['max_quantity'] < 1).sum()
        if zero_qty_count > 0:
            logger.debug(f"Filtering out {zero_qty_count} candidates with max_quantity < 1")
        df = cast(pd.DataFrame, df[df['max_quantity'] >= 1])

        # Clean up temp columns
        df = df.drop(columns=['max_qty_by_capital', 'effective_limit'])

        return df

    def _check_manipulation_signals_vectorized(
        self,
        df: pd.DataFrame,
        volumes_24h: dict[int, int],
        volumes_1h: dict[int, int],
    ) -> pd.Series:
        """Check for manipulation signals on all candidates (vectorized version).

        Returns a boolean mask where True = suspicious (should be filtered out).

        Manipulation checks:
        1. High spread (> max_spread_pct)
        2. Volume concentration (1h volume / 24h volume > threshold)
        3. Low volume for cheap items (< 1000gp with low 24h volume)

        Args:
            df: DataFrame with item_id, buy_price, current_high, current_low
            volumes_24h: Pre-fetched 24h volumes (item_id -> volume)
            volumes_1h: Pre-fetched 1h volumes (item_id -> volume)

        Returns:
            Boolean Series where True = suspicious item to filter out
        """
        if df.empty:
            return pd.Series([], dtype=bool)

        # Enrich with volume data
        df = df.copy()
        df['volume_24h'] = df['item_id'].map(volumes_24h)
        df['volume_1h'] = df['item_id'].map(volumes_1h).fillna(0)

        # Calculate spread_pct (with fallback if current_low is missing/NaN)
        df['current_low'] = df['current_low'].fillna(df['buy_price'])
        # Replace zero with buy_price to avoid division by zero
        df['current_low'] = df['current_low'].where(df['current_low'] != 0, df['buy_price'])
        df['_spread_pct'] = (df['current_high'] - df['current_low']) / df['current_low']

        # Initialize risk score column
        df['_risk_score'] = 0.0

        # Check 1: High spread
        high_spread_mask = df['_spread_pct'] > self.config.max_spread_pct
        df.loc[high_spread_mask, '_risk_score'] += 0.3

        # Check 2: Volume concentration (only where volume_24h > 0)
        has_volume_data = (df['volume_24h'].notna()) & (df['volume_24h'] > 0)
        df['_concentration'] = 0.0
        df.loc[has_volume_data, '_concentration'] = (
            df.loc[has_volume_data, 'volume_1h'] / df.loc[has_volume_data, 'volume_24h']
        )
        volume_spike_mask = df['_concentration'] > self.config.max_volume_concentration
        df.loc[volume_spike_mask, '_risk_score'] += 0.4

        # Check 3: Low volume for cheap items (< 1000gp)
        cheap_items = df['buy_price'] < 1000
        has_volume = df['volume_24h'].notna()
        low_volume = df['volume_24h'] < self.config.min_volume_for_low_value
        low_volume_cheap_mask = cheap_items & has_volume & low_volume
        df.loc[low_volume_cheap_mask, '_risk_score'] += 0.35

        # Final decision: suspicious if risk_score > 0.5
        suspicious_mask = df['_risk_score'] > 0.5

        if suspicious_mask.sum() > 0:
            logger.debug(
                f"Manipulation check flagged {suspicious_mask.sum()} suspicious items "
                f"(risk score > 0.5)"
            )

        return suspicious_mask

    def _enrich_metadata_vectorized(
        self,
        df: pd.DataFrame,
        pred_age_seconds: float,
        volumes_24h: dict[int, int],
        volumes_1h: dict[int, int],
        trends: dict[int, str],
    ) -> pd.DataFrame:
        """Enrich candidates with metadata (vectorized version).

        Adds the following columns:
        - tax_per_unit: GE tax per unit (2% with floor and cap)
        - profit_per_unit: Profit after tax
        - confidence: Adjusted confidence based on prediction freshness
        - volume_tier: Volume tier based on spread
        - crowding_capacity: Concurrent user limit based on volume
        - trend: Market trend (pre-fetched)
        - volume_24h, volume_1h: Trading volumes (pre-fetched)
        - item_name: Item name with fallback

        Args:
            df: DataFrame with candidates
            pred_age_seconds: Age of predictions in seconds (for confidence adjustment)
            volumes_24h: Pre-fetched 24h volumes
            volumes_1h: Pre-fetched 1h volumes
            trends: Pre-fetched trends

        Returns:
            DataFrame with enriched metadata columns
        """
        if df.empty:
            return df

        df = df.copy()
        import numpy as np  # type: ignore[import-untyped]

        # 1. Tax calculation (vectorized tax_calculator.calculate_tax logic)
        # Tax = 2% of sell_price, rounded down, with floor of 50gp and cap of 5M
        df['tax_per_unit'] = (df['sell_price'] * 0.02).astype(int)
        df['tax_per_unit'] = np.minimum(df['tax_per_unit'], 5_000_000)
        df.loc[df['sell_price'] < 50, 'tax_per_unit'] = 0

        # 2. Profit per unit
        df['profit_per_unit'] = df['sell_price'] - df['buy_price'] - df['tax_per_unit']

        # 3. Confidence adjustment (vectorized _adjust_confidence logic)
        # Map DB confidence to numeric
        df['confidence'] = df['confidence'].fillna('medium')
        confidence_map = {"high": 2, "medium": 1, "low": 0}
        df['_confidence_num'] = df['confidence'].map(confidence_map).fillna(1).astype(int)

        # Downgrade if data is stale (> 6 minutes)
        if pred_age_seconds > 360:
            df['_confidence_num'] = np.maximum(0, df['_confidence_num'] - 1)

        # Map back to string
        df['confidence'] = 'medium'  # Default
        df.loc[df['_confidence_num'] >= 2, 'confidence'] = 'high'
        df.loc[df['_confidence_num'] == 1, 'confidence'] = 'medium'
        df.loc[df['_confidence_num'] <= 0, 'confidence'] = 'low'
        df = df.drop(columns=['_confidence_num'])

        # 4. Spread percentage (already calculated in manipulation check, recalculate if needed)
        if '_spread_pct' not in df.columns:
            df['current_low'] = df['current_low'].fillna(df['buy_price'])
            # Replace zero with buy_price to avoid division by zero
            df['current_low'] = df['current_low'].where(df['current_low'] != 0, df['buy_price'])
            df['_spread_pct'] = (df['current_high'] - df['current_low']) / df['current_low']

        # 5. Volume tier (vectorized _determine_volume_tier logic)
        df['volume_tier'] = 'Low'  # Default
        df.loc[df['_spread_pct'] < 0.05, 'volume_tier'] = 'Medium'
        df.loc[df['_spread_pct'] < 0.02, 'volume_tier'] = 'High'
        df.loc[df['_spread_pct'] < 0.01, 'volume_tier'] = 'Very High'

        # 6. Volumes (if not already mapped)
        if 'volume_24h' not in df.columns:
            df['volume_24h'] = df['item_id'].map(volumes_24h)
        if 'volume_1h' not in df.columns:
            df['volume_1h'] = df['item_id'].map(volumes_1h).fillna(0)

        # Convert NaN to None for volume_24h to match original behavior
        # (NaN values become None in dict, which are then excluded from final recommendations)
        import numpy as np  # type: ignore[import-untyped]
        df['volume_24h'] = df['volume_24h'].replace({np.nan: None})

        # 7. Crowding capacity (vectorized _get_crowding_capacity logic)
        # None = unlimited, otherwise integer limit
        df['crowding_capacity'] = 10  # Default
        df.loc[df['volume_1h'] > 1_000, 'crowding_capacity'] = 20
        df.loc[df['volume_1h'] > 10_000, 'crowding_capacity'] = 50
        df.loc[df['volume_1h'] > 50_000, 'crowding_capacity'] = None
        # Handle missing volume data conservatively
        df.loc[df['volume_1h'].isna(), 'crowding_capacity'] = 20

        # 8. Trend (map from pre-fetched dict)
        df['trend'] = df['item_id'].map(trends).fillna('Stable')

        # 9. Item name (with fallback)
        if 'item_name' not in df.columns:
            df['item_name'] = df['item_id'].apply(lambda x: f"Item {x}")
        else:
            df['item_name'] = df['item_name'].fillna(
                df['item_id'].apply(lambda x: f"Item {x}")
            )

        return df

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

    # ========== Portfolio Optimization ==========

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

        min_slots_target = int(min(params["min_slots"], num_slots, len(candidates)))
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
                "whyChips": self.generate_why_chips(cand),
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
            from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
            import numpy as np  # type: ignore[import-untyped]
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

        selected: list[dict] = []
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

    def _check_manipulation_signals(
        self,
        item_id: int,
        buy_price: int,
        spread_pct: float,
        volume_24h: Optional[int] = None,
        volume_1h: Optional[int] = None,
    ) -> tuple[bool, float, list[str]]:
        """Check for manipulation signals on a single item.

        Thin wrapper for backwards compatibility with tests.
        Replicates the exact logic from the original method.

        Args:
            item_id: OSRS item ID
            buy_price: Current buy price
            spread_pct: Current spread percentage
            volume_24h: 24-hour trading volume
            volume_1h: 1-hour trading volume

        Returns:
            Tuple of (is_suspicious, risk_score, reasons)
        """
        reasons = []
        risk_score = 0.0

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
        if buy_price < 1000 and volume_24h is not None:
            if volume_24h < self.config.min_volume_for_low_value:
                reasons.append(
                    f"Low volume for cheap item ({volume_24h:,} < {self.config.min_volume_for_low_value:,})"
                )
                risk_score += 0.35

        # Item is suspicious if risk score exceeds threshold
        is_suspicious = risk_score > 0.5

        return (is_suspicious, min(risk_score, 1.0), reasons)

    def _apply_liquidity_filter(
        self,
        predictions: pd.DataFrame,
        buy_limits: dict[int, int],
        volumes_24h: dict[int, int],
    ) -> pd.DataFrame:
        """Filter out items where buy_limit is too large relative to daily volume.

        Items where buy_limit / volume_24h > max_ratio are filtered out because:
        1. Filling your buy limit would dominate the market (move prices)
        2. Low relative liquidity makes these items easier to manipulate
        3. Fill times would be unpredictable

        Args:
            predictions: DataFrame with 'item_id' column
            buy_limits: Dict mapping item_id -> buy_limit
            volumes_24h: Dict mapping item_id -> 24h volume

        Returns:
            Filtered DataFrame with illiquid items removed
        """
        max_ratio = self.config.max_buy_limit_volume_ratio

        def passes_liquidity_check(row) -> bool:
            item_id = int(row["item_id"])
            buy_limit = buy_limits.get(item_id)
            volume = volumes_24h.get(item_id)

            # Skip check if data missing
            if not buy_limit or not volume:
                return True

            # Skip if volume is 0 (would divide by zero)
            if volume <= 0:
                return False

            ratio = buy_limit / volume

            if ratio > max_ratio:
                return False

            return True

        mask = predictions.apply(passes_liquidity_check, axis=1)
        rejected = predictions[~mask]

        # Log rejections for debugging (batch summary)
        if not rejected.empty:
            item_names = rejected['item_name'].fillna('Unknown').tolist()
            logger.info(
                f"Liquidity filter rejected {len(rejected)} items: "
                f"{', '.join(item_names[:5])}"
                + (" ..." if len(rejected) > 5 else "")
            )

        return cast(pd.DataFrame, predictions[mask].reset_index(drop=True))

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

    # ========== Utility Methods ==========

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

    def generate_why_chips(self, candidate: dict) -> list[dict]:
        """Generate why chips from candidate features.

        Why chips are small visual indicators that explain to users why a
        recommendation is good (e.g., "High confidence", "Fast fill").

        Args:
            candidate: Candidate dict with prediction and feature data

        Returns:
            List of chip dicts with icon, label, and type (max 4 chips)
        """
        chips = []

        # Confidence
        conf = candidate.get("confidence", "low")
        if conf == "high":
            chips.append({"icon": "🎯", "label": "High confidence", "type": "positive"})
        elif conf == "medium":
            chips.append({"icon": "🎯", "label": "Med confidence", "type": "neutral"})

        # Volume (use volume_24h if available, otherwise fall back to volume_tier)
        vol_24h = candidate.get("volume_24h", 0)
        if vol_24h and vol_24h > 100000:
            chips.append({"icon": "🔥", "label": "High volume", "type": "positive"})
        elif vol_24h and vol_24h > 10000:
            chips.append({"icon": "📊", "label": "Good volume", "type": "neutral"})
        else:
            # Fall back to volume tier
            volume_tier = candidate.get("volume_tier", "")
            if volume_tier in ("Very High", "High"):
                chips.append({"icon": "🔥", "label": "High volume", "type": "positive"})
            elif volume_tier == "Medium":
                chips.append({"icon": "📊", "label": "Good volume", "type": "neutral"})

        # Fill probability
        fill_prob = candidate.get("fill_probability", 0)
        if fill_prob >= 0.9:
            chips.append({"icon": "⚡", "label": "Fast fill", "type": "positive"})

        # Trend
        trend = candidate.get("trend", "Stable")
        if trend == "Rising":
            chips.append({"icon": "📈", "label": "Trending up", "type": "positive"})
        elif trend == "Falling":
            chips.append({"icon": "📉", "label": "Trending down", "type": "negative"})

        # Time (hour_offset)
        hours = candidate.get("hour_offset", 4)
        if hours <= 2:
            chips.append({"icon": "⏱", "label": "Quick flip", "type": "positive"})
        elif hours >= 6:
            chips.append({"icon": "🕐", "label": "Longer hold", "type": "neutral"})

        # Spread (use _spread_pct if available)
        spread_pct = candidate.get("_spread_pct", 0)
        if spread_pct and spread_pct > 0.02:  # > 2% spread
            spread_display = int(spread_pct * 100)
            chips.append(
                {"icon": "📈", "label": f"Spread +{spread_display}%", "type": "positive"}
            )

        return chips[:4]  # Max 4 chips

    def get_all_opportunities(self) -> list[dict]:
        """Get all valid trading opportunities for browsing.

        Unlike get_recommendations(), this method:
        - Uses default thresholds (not user-specific)
        - Returns all valid opportunities (no crowding filter)
        - Includes fields for opportunity browsing UI

        Returns:
            List of opportunity dicts with item details, prices, profits, etc.
        """
        # Use generous default thresholds for browsing
        min_fill_prob = 0.1  # Lower threshold since model outputs range 0-0.3
        min_ev = 0.003  # Low EV threshold for broad results
        max_hour = 48  # Include all time horizons
        candidate_limit = 500  # Get a large pool of candidates

        # Fetch predictions from database
        predictions_df = self.loader.get_best_prediction_per_item(
            min_fill_prob=min_fill_prob,
            min_ev=min_ev,
            min_hour_offset=1,
            max_hour_offset=max_hour,
            min_offset_pct=0.0125,
            max_offset_pct=0.0250,
            limit=candidate_limit,
            min_volume_24h=None,  # Volume filter applied post-fetch via _apply_liquidity_filter
        )

        if predictions_df.empty:
            logger.warning("No predictions found for opportunities browsing")
            return []

        # Batch fetch all per-item data upfront (eliminates N+1 queries)
        candidate_item_ids = predictions_df["item_id"].unique().tolist()
        buy_limits = self.loader.get_batch_buy_limits(candidate_item_ids)
        volumes_24h = self.loader.get_batch_volumes_24h(candidate_item_ids)
        trends = self.loader.get_batch_trends(candidate_item_ids)

        # Apply liquidity filter (anti-manipulation: filter items where buy_limit >> volume)
        predictions_df = self._apply_liquidity_filter(
            predictions_df, buy_limits, volumes_24h
        )
        if predictions_df.empty:
            logger.info("All predictions filtered by liquidity check for opportunities")
            return []

        # Use a large default capital for quantity calculations
        default_capital = 1_000_000_000  # 1B gp

        # Build opportunity list (vectorized)
        df = predictions_df.copy()

        # Type conversions (vectorized)
        df['item_id'] = df['item_id'].astype(int)
        df['buy_price'] = df['buy_price'].astype(int)
        df['sell_price'] = df['sell_price'].astype(int)
        df['fill_probability'] = df['fill_probability'].astype(float)
        df['expected_value'] = df['expected_value'].astype(float)
        df['hour_offset'] = df['hour_offset'].astype(int)

        # Handle optional columns with defaults
        if 'confidence' not in df.columns:
            df['confidence'] = 'medium'
        else:
            df['confidence'] = df['confidence'].fillna('medium')

        if 'item_name' not in df.columns:
            df['item_name'] = df['item_id'].apply(lambda x: f"Item {x}")
        else:
            df['item_name'] = df['item_name'].fillna(df['item_id'].apply(lambda x: f"Item {x}"))

        # Batch lookups (vectorized)
        df['buy_limit'] = df['item_id'].map(buy_limits).fillna(10000).astype(int)
        df['volume_24h'] = df['item_id'].map(volumes_24h)
        df['trend'] = df['item_id'].map(trends).fillna('Stable')

        # Quantity calculation (vectorized)
        df['max_qty_by_capital'] = (default_capital // df['buy_price']).fillna(0).astype(int)
        df['max_quantity'] = df[['buy_limit', 'max_qty_by_capital']].min(axis=1)  # type: ignore[call-overload]

        # Filter out zero quantity (vectorized)
        df = df[df['max_quantity'] >= 1]

        # Profit calculations (vectorized)
        df['tax_per_unit'] = df['sell_price'].apply(calculate_tax)
        df['profit_per_unit'] = df['sell_price'] - df['buy_price'] - df['tax_per_unit']

        # Filter negative profit (vectorized)
        df = df[df['profit_per_unit'] > 0]

        # Expected profit and capital (vectorized)
        df['capital_required'] = df['buy_price'] * df['max_quantity']
        df['expected_profit'] = (df['profit_per_unit'] * df['max_quantity'] * df['fill_probability']).astype(int)

        # Icon URLs (vectorized)
        df['icon_url'] = df['item_id'].apply(
            lambda item_id: f"https://secure.runescape.com/m=itemdb_oldschool/obj_big.gif?id={item_id}"
        )

        # Volume tier (vectorized)
        df['volume_tier'] = df['volume_24h'].apply(lambda v: self._volume_to_tier(v) if v else None)

        # Spread percentage (vectorized)
        df['_spread_pct'] = (df['sell_price'] - df['buy_price']) / df['buy_price']

        # Round numerical fields (vectorized)
        df['fill_probability_rounded'] = df['fill_probability'].round(4)
        df['expected_value_rounded'] = df['expected_value'].round(6)

        # Build opportunity dicts (convert to list of dicts)
        df['category'] = None  # Not currently in items table
        opportunities = df[[
            'item_id', 'item_name', 'icon_url', 'buy_price', 'sell_price',
            'max_quantity', 'capital_required', 'expected_profit', 'hour_offset',
            'confidence', 'fill_probability_rounded', 'expected_value_rounded',
            'volume_24h', 'trend', 'category', 'volume_tier', '_spread_pct'
        ]].rename(columns={
            'fill_probability_rounded': 'fill_probability',
            'expected_value_rounded': 'expected_value',
            'max_quantity': 'quantity',
            'hour_offset': 'expected_hours'
        }).copy()

        # Add hour_offset back for generate_why_chips
        opportunities['hour_offset'] = df['hour_offset'].values

        # Sort by expected profit descending (vectorized)
        opportunities = opportunities.sort_values('expected_profit', ascending=False)

        # Convert to list of dicts
        opportunities = opportunities.to_dict('records')

        logger.info(f"Generated {len(opportunities)} opportunities for browsing")
        return opportunities

    def _volume_to_tier(self, volume_24h: Optional[int]) -> Optional[str]:
        """Convert 24h volume to tier string for why chips."""
        if volume_24h is None:
            return None
        if volume_24h > 100000:
            return "Very High"
        if volume_24h > 50000:
            return "High"
        if volume_24h > 10000:
            return "Medium"
        return "Low"

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
                int(best_any["buy_price"].item()) if best_any["buy_price"] is not None else 0
            )
            sell_price = (
                int(best_any["sell_price"].item()) if best_any["sell_price"] is not None else 0
            )
            spread = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
            ev = (
                float(best_any["expected_value"].item())
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
                    float(best_any["fill_probability"].item())
                    if best_any["fill_probability"] is not None
                    else 0
                )
                if fill_prob < min_fill:
                    reason = (
                        f"Fill probability ({fill_prob:.1%}) too low for {risk} risk"
                    )
                else:
                    hour_offset = int(best_any["hour_offset"].item())
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
            "whyChips": self.generate_why_chips(candidate),
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
                "hour_offset": int(best["hour_offset"].item()),
                "offset_pct": float(best["offset_pct"].item()),
            },
            "fill_probability": float(best["fill_probability"].item()),
            "expected_value": float(best["expected_value"].item()),
            "buy_price": int(best["buy_price"].item()),
            "sell_price": int(best["sell_price"].item()),
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
        hour_offset = int(best_row["hour_offset"].item())
        actual_offset_pct = float(best_row["offset_pct"].item())
        fill_prob = float(best_row["fill_probability"].item())
        ev = float(best_row["expected_value"].item())

        # Get prices based on side
        if side == "buy":
            recommended_price = int(best_row["buy_price"].item())
            try:
                current_market_price = int(best_row["current_high"].item())
            except (ValueError, TypeError, AttributeError):
                current_market_price = recommended_price
        else:  # sell
            recommended_price = int(best_row["sell_price"].item())
            try:
                current_market_price = int(best_row["current_low"].item())
            except (ValueError, TypeError, AttributeError):
                current_market_price = recommended_price

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
        buy_price = int(best_row["buy_price"].item())
        sell_price = int(best_row["sell_price"].item())
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
