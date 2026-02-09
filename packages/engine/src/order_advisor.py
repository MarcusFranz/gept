"""Active order advisor for evaluating existing GE orders.

Provides recommendations for users with unfilled buy or sell orders:
- adjust_price: Modify the order price to increase fill probability
- wait: Keep the current order and wait longer
- abort_retry: Cancel and try a different item
- liquidate: Execute immediately at market price (accept loss)
"""

import logging
import math
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from .prediction_loader import PredictionLoader
    from .recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)


class OrderAdvisor:
    """Evaluates active orders and provides recommendations."""

    # Decision thresholds
    HIGH_FILL_PROB_THRESHOLD = 0.7
    LOW_FILL_PROB_THRESHOLD = 0.1
    UNFAVORABLE_MOVE_THRESHOLD = 0.02
    TARGET_FILL_PROB = 0.6
    SELL_TARGET_FILL_PROB = 0.8
    # When the market is moving against the user (especially on sells in a downturn),
    # we should bias toward faster fills to avoid "chasing" the price down.
    SHARP_DOWNTURN_TARGET_FILL_PROB = 0.9
    DEFAULT_HOUR_WINDOW = 4
    # Fallback heuristic when predictions are missing:
    # fill_prob ~= base * exp(-k * offset_from_market)
    _HEURISTIC_BASE_FILL_PROB = 0.95
    _HEURISTIC_OFFSET_DECAY = 40.0

    def __init__(
        self,
        loader: "PredictionLoader",
        engine: Optional["RecommendationEngine"] = None,
        use_beta_model: bool = False,
    ):
        """Initialize the OrderAdvisor.

        Args:
            loader: PredictionLoader for accessing price and prediction data
            engine: RecommendationEngine for finding alternatives (optional)
            use_beta_model: Whether to use the engine's beta model for any follow-on
                recommendation calls (e.g., alternative item suggestions).
        """
        self.loader = loader
        self.engine = engine
        self.use_beta_model = use_beta_model

    def evaluate_order(
        self,
        item_id: int,
        order_type: Literal["buy", "sell"],
        user_price: int,
        quantity: int,
        time_elapsed_minutes: int,
        user_id: Optional[str] = None,
    ) -> dict:
        """Evaluate an active order and recommend action.

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
        # Get current market prices
        price_data = self.loader.get_latest_price(item_id)
        if price_data is None:
            return self._error_response("Unable to fetch current market prices")

        current_high = price_data["high"]
        current_low = price_data["low"]

        # Get predictions for this item
        predictions_df = self.loader.get_predictions_for_item(item_id)
        has_predictions = not predictions_df.empty

        # Get item name from predictions if present; fall back to Wiki mapping.
        item_name: Optional[str] = None
        if has_predictions and "item_name" in predictions_df.columns and len(predictions_df) > 0:
            try:
                item_name = str(predictions_df.iloc[0].get("item_name"))
            except Exception:
                item_name = None
        if not item_name:
            item_name = self.loader.get_item_name(item_id) or f"Item {item_id}"

        # Calculate fill probability at user's price
        fill_prob = self._calculate_fill_probability_at_price(
            predictions_df, user_price, order_type, current_high, current_low
        )

        # Get trend for context
        trend = self.loader.get_item_trend(item_id)

        # Calculate price movement analysis
        price_move_pct, moved_favorably = self._analyze_price_movement(
            user_price, order_type, current_high, current_low
        )

        # Estimate predicted window from predictions
        predicted_window_hours = self._get_typical_hour_window(predictions_df)

        # Build all recommendation options
        recommendations = self._build_recommendations(
            item_id=item_id,
            item_name=item_name,
            order_type=order_type,
            user_price=user_price,
            quantity=quantity,
            current_high=current_high,
            current_low=current_low,
            predictions_df=predictions_df,
            trend=trend,
            price_move_pct=price_move_pct,
            price_moved_favorably=moved_favorably,
            time_elapsed_minutes=time_elapsed_minutes,
            predicted_window_hours=predicted_window_hours,
        )

        # Determine recommended action
        has_alternatives = (
            recommendations.get("abort_retry") is not None
            and len(recommendations["abort_retry"]["alternative_items"]) > 0
        )

        action = self._determine_action(
            fill_prob=fill_prob,
            price_move_pct=price_move_pct,
            price_moved_favorably=moved_favorably,
            time_elapsed=time_elapsed_minutes,
            predicted_window_hours=predicted_window_hours,
            has_alternatives=has_alternatives,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            fill_prob=fill_prob,
            predictions_df=predictions_df,
            time_elapsed=time_elapsed_minutes,
            predicted_window_hours=predicted_window_hours,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            action=action,
            fill_prob=fill_prob,
            price_move_pct=price_move_pct,
            moved_favorably=moved_favorably,
            trend=trend,
            order_type=order_type,
            time_elapsed_minutes=time_elapsed_minutes,
        )
        if not has_predictions:
            reasoning = (
                "No prediction data available for this item; using live price ticks only. "
                + reasoning
            )

        # If we chose an action that requires a recommendation payload but couldn't
        # generate one, fall back to "wait" to avoid surprising clients.
        if action == "adjust_price" and not recommendations.get("adjust_price"):
            action = "wait"
        if action == "abort_retry" and not recommendations.get("abort_retry"):
            action = "wait"
        if action == "liquidate" and not recommendations.get("liquidate"):
            action = "wait"

        return {
            "action": action,
            "confidence": round(confidence, 2),
            "current_fill_probability": round(fill_prob, 2),
            "recommendations": recommendations,
            "reasoning": reasoning,
        }

    def _calculate_fill_probability_at_price(
        self,
        predictions_df,
        user_price: int,
        order_type: str,
        current_high: int,
        current_low: int,
    ) -> float:
        """Interpolate fill probability at user's specific price.

        Args:
            predictions_df: DataFrame with predictions for the item
            user_price: User's order price
            order_type: 'buy' or 'sell'
            current_high: Current "high" tick (often corresponds to buy-side pressure)
            current_low: Current "low" tick (often corresponds to sell-side pressure)

        Returns:
            Estimated fill probability (0-1)
        """
        # Calculate user's effective offset from market
        if order_type == "buy":
            # For buy orders: offset = how much below the current low tick
            # (model buy offsets are trained around low-side ticks).
            if current_low <= 0:
                return 0.0
            user_offset = (current_low - user_price) / current_low
        else:
            # For sell orders: offset = how much above the current high tick.
            if current_high <= 0:
                return 0.0
            user_offset = (user_price - current_high) / current_high

        # If user is at or better than market, very high fill prob
        if user_offset <= 0:
            return 0.95

        # If predictions are missing, fall back to a simple heuristic based only
        # on the user's distance from the current instant price.
        try:
            is_empty = predictions_df is None or bool(getattr(predictions_df, "empty"))
        except Exception:
            is_empty = True
        if is_empty:
            return self._heuristic_fill_probability(user_offset)

        # Use default hour window for interpolation
        window_preds = predictions_df[
            predictions_df["hour_offset"] == self.DEFAULT_HOUR_WINDOW
        ]
        if window_preds.empty:
            # Fall back to any predictions
            window_preds = predictions_df

        if window_preds.empty:
            return 0.5  # No data, return moderate estimate

        # Sort by offset percentage
        sorted_preds = window_preds.sort_values("offset_pct")
        offsets = sorted_preds["offset_pct"].values
        probs = sorted_preds["fill_probability"].values

        # Interpolate
        if user_offset <= offsets[0]:
            # Closer to market than any trained offset - higher probability
            return min(0.95, float(probs[0]) * 1.2)

        if user_offset >= offsets[-1]:
            # Further from market than any trained offset - lower probability
            return max(0.01, float(probs[-1]) * 0.5)

        # Linear interpolation
        return float(np.interp(user_offset, offsets, probs))

    def _heuristic_fill_probability(self, user_offset: float) -> float:
        """Heuristic fill probability when we don't have model predictions."""
        try:
            offset = max(0.0, float(user_offset))
            prob = self._HEURISTIC_BASE_FILL_PROB * math.exp(
                -self._HEURISTIC_OFFSET_DECAY * offset
            )
            return max(0.01, min(self._HEURISTIC_BASE_FILL_PROB, float(prob)))
        except Exception:
            return 0.5

    def _analyze_price_movement(
        self,
        user_price: int,
        order_type: str,
        current_high: int,
        current_low: int,
    ) -> tuple[float, bool]:
        """Analyze how price has moved relative to user's order.

        Returns:
            Tuple of (move percentage, whether move is favorable)
        """
        if order_type == "buy":
            # For buy: favorable if the current low tick is at/below the user's bid.
            # If the user's bid remains below the low tick, fills are less likely.
            reference = current_low
            if reference <= 0:
                return 0.0, True
            move_pct = (reference - user_price) / reference
            # Positive = user bid is below market (unfavorable for buy)
            favorable = move_pct <= 0
        else:
            # For sell: favorable if the current high tick is at/above the user's ask.
            reference = current_high
            if reference <= 0:
                return 0.0, True
            move_pct = (user_price - reference) / reference
            # Positive = user ask is above market (unfavorable for sell)
            favorable = move_pct <= 0

        return abs(move_pct), favorable

    def _get_typical_hour_window(self, predictions_df) -> int:
        """Get typical hour window from predictions."""
        if predictions_df.empty:
            return self.DEFAULT_HOUR_WINDOW

        # Get the most common hour offset with reasonable fill prob
        # Top ~20% of probability range (cap is 0.50) for hour window estimation
        reasonable = predictions_df[predictions_df["fill_probability"] >= 0.4]
        if reasonable.empty:
            reasonable = predictions_df

        # Return median hour offset
        return int(reasonable["hour_offset"].median())

    def _build_recommendations(
        self,
        item_id: int,
        item_name: str,
        order_type: str,
        user_price: int,
        quantity: int,
        current_high: int,
        current_low: int,
        predictions_df,
        trend: str,
        price_move_pct: float,
        price_moved_favorably: bool,
        time_elapsed_minutes: int,
        predicted_window_hours: int,
    ) -> dict:
        """Build all recommendation options."""
        recommendations = {}

        # Adjust price recommendation
        adjust = self._suggest_adjusted_price(
            predictions_df=predictions_df,
            order_type=order_type,
            user_price=user_price,
            quantity=quantity,
            current_high=current_high,
            current_low=current_low,
            trend=trend,
            price_move_pct=price_move_pct,
            price_moved_favorably=price_moved_favorably,
            time_elapsed_minutes=time_elapsed_minutes,
            predicted_window_hours=predicted_window_hours,
        )
        if adjust:
            recommendations["adjust_price"] = adjust

        # Wait recommendation
        wait = self._estimate_wait_time(
            predictions_df=predictions_df,
            user_price=user_price,
            order_type=order_type,
            current_high=current_high,
            current_low=current_low,
            time_elapsed_minutes=time_elapsed_minutes,
            predicted_window_hours=predicted_window_hours,
        )
        if wait:
            recommendations["wait"] = wait

        # Abort/retry recommendation (find alternatives)
        abort = self._find_alternatives(
            item_id=item_id,
            capital=user_price * quantity,
        )
        if abort:
            recommendations["abort_retry"] = abort

        # Liquidate recommendation
        liquidate = self._calculate_liquidation(
            order_type=order_type,
            user_price=user_price,
            quantity=quantity,
            current_high=current_high,
            current_low=current_low,
        )
        if liquidate:
            recommendations["liquidate"] = liquidate

        return recommendations

    def _suggest_adjusted_price(
        self,
        predictions_df,
        order_type: str,
        user_price: int,
        quantity: int,
        current_high: int,
        current_low: int,
        trend: str,
        price_move_pct: float,
        price_moved_favorably: bool,
        time_elapsed_minutes: int,
        predicted_window_hours: int,
    ) -> Optional[dict]:
        """Calculate suggested price adjustment to achieve target fill probability."""
        # Dynamic target fill probability: in a downturn (or when price moved
        # against the user), bias toward being more aggressive to avoid repeated
        # revisions that still don't fill.
        target_fill_prob = float(
            self.SELL_TARGET_FILL_PROB if order_type == "sell" else self.TARGET_FILL_PROB
        )
        trend_l = str(trend or "").strip().lower()
        moved_against = (not price_moved_favorably) and price_move_pct > self.UNFAVORABLE_MOVE_THRESHOLD
        if order_type == "sell" and (trend_l.startswith("down") or moved_against):
            if price_move_pct >= 0.05:
                target_fill_prob = max(
                    target_fill_prob, float(self.SHARP_DOWNTURN_TARGET_FILL_PROB)
                )
        if order_type == "buy" and (trend_l.startswith("up") or moved_against):
            target_fill_prob = max(target_fill_prob, float(self.SELL_TARGET_FILL_PROB))

        predicted_window_minutes = max(1, int(predicted_window_hours) * 60)
        time_ratio = time_elapsed_minutes / predicted_window_minutes
        if moved_against and time_ratio >= 1.0:
            target_fill_prob = max(target_fill_prob, float(self.SHARP_DOWNTURN_TARGET_FILL_PROB))

        # Compute user's current offset from the market reference price.
        # Positive offset means the order is less aggressive than the reference
        # (buy below low tick, sell above high tick). Negative means more aggressive.
        if order_type == "buy":
            if current_low <= 0:
                return None
            user_offset = (current_low - user_price) / current_low
        else:
            if current_high <= 0:
                return None
            user_offset = (user_price - current_high) / current_high

        # If predictions are missing, use a heuristic offset that corresponds to
        # the TARGET_FILL_PROB under the exponential decay model.
        try:
            is_empty = predictions_df is None or bool(getattr(predictions_df, "empty"))
        except Exception:
            is_empty = True
        if is_empty:
            target_prob = float(target_fill_prob)
            try:
                target_offset = -math.log(
                    target_prob / self._HEURISTIC_BASE_FILL_PROB
                ) / self._HEURISTIC_OFFSET_DECAY
            except Exception:
                target_offset = 0.0

            # If the user is already at the target price (within rounding), don't suggest a change.
            if abs(user_offset - target_offset) < 1e-9:
                return None

            if order_type == "buy":
                suggested = int(current_low * (1 - target_offset))
            else:
                suggested = int(current_high * (1 + target_offset))
            if suggested <= 0 or suggested == user_price:
                return None

            # If the market is moving against the user, cap at "at market" to
            # avoid suggesting a price that is still above (sell) / below (buy)
            # the current reference tick.
            if order_type == "sell" and (trend_l.startswith("down") or moved_against):
                suggested = min(suggested, int(current_high))
            if order_type == "buy" and (trend_l.startswith("up") or moved_against):
                suggested = max(suggested, int(current_low))
            if suggested <= 0 or suggested == user_price:
                return None

            cost_diff = abs(suggested - user_price) * quantity
            new_prob = self._calculate_fill_probability_at_price(
                predictions_df, suggested, order_type, current_high, current_low
            )
            return {
                "suggested_price": suggested,
                "new_fill_probability": round(float(new_prob), 2),
                "cost_difference": cost_diff,
            }

        # Find offset that achieves target fill prob (vectorized)
        # Filter to rows meeting target fill probability
        viable = predictions_df[predictions_df["fill_probability"] >= target_fill_prob]

        if not viable.empty:
            # Get row with minimum offset_pct among viable candidates
            min_idx = viable["offset_pct"].idxmin()
            target_offset = viable.loc[min_idx, "offset_pct"]
            target_prob = viable.loc[min_idx, "fill_probability"]
        else:
            # Use best available (highest fill probability)
            best_idx = predictions_df["fill_probability"].idxmax()
            target_offset = predictions_df.loc[best_idx, "offset_pct"]
            target_prob = predictions_df.loc[best_idx, "fill_probability"]

        try:
            target_offset_f = float(target_offset)
        except Exception:
            target_offset_f = 0.0

        # Calculate new price based on offset
        if order_type == "buy":
            suggested = int(current_low * (1 - target_offset_f))
        else:
            suggested = int(current_high * (1 + target_offset_f))
        if suggested <= 0 or suggested == user_price:
            return None

        # In a downturn (sell) or upturn (buy), cap at market tick to be more
        # proactive about fills. This prevents "chasing" down with repeated
        # adjustments that are still not aggressive enough.
        if order_type == "sell" and (trend_l.startswith("down") or moved_against):
            suggested = min(suggested, int(current_high))
        if order_type == "buy" and (trend_l.startswith("up") or moved_against):
            suggested = max(suggested, int(current_low))
        if suggested <= 0 or suggested == user_price:
            return None

        # Calculate cost difference
        cost_diff = abs(suggested - user_price) * quantity

        return {
            "suggested_price": suggested,
            "new_fill_probability": round(
                float(
                    self._calculate_fill_probability_at_price(
                        predictions_df, suggested, order_type, current_high, current_low
                    )
                ),
                2,
            ),
            "cost_difference": cost_diff,
        }

    def _estimate_wait_time(
        self,
        predictions_df,
        user_price: int,
        order_type: str,
        current_high: int,
        current_low: int,
        time_elapsed_minutes: int,
        predicted_window_hours: int,
    ) -> Optional[dict]:
        """Estimate remaining wait time until fill."""
        # Get fill probability at user's price
        fill_prob = self._calculate_fill_probability_at_price(
            predictions_df, user_price, order_type, current_high, current_low
        )

        # Estimate total time based on fill probability and hour window
        # Higher fill prob = lower time estimate
        if fill_prob >= 0.7:
            estimated_total = predicted_window_hours * 60 * 0.5  # Half the window
        elif fill_prob >= 0.4:
            estimated_total = predicted_window_hours * 60 * 0.8
        else:
            estimated_total = predicted_window_hours * 60 * 1.2  # Longer than window

        remaining = max(0, int(estimated_total - time_elapsed_minutes))

        return {"estimated_fill_time_minutes": remaining}

    def _find_alternatives(
        self,
        item_id: int,
        capital: int,
    ) -> Optional[dict]:
        """Find alternative items if abort_retry is recommended."""
        if self.engine is None:
            return None

        try:
            # Get recommendations excluding current item
            alternatives = self.engine.get_recommendations(
                style="hybrid",
                capital=capital,
                risk="medium",
                slots=3,
                exclude_item_ids={item_id},
                use_beta_model=self.use_beta_model,
            )

            if not alternatives:
                return None

            alt_list = [
                {
                    "item_id": alt.get("itemId", alt.get("item_id")),
                    "item_name": alt.get("item", alt.get("item_name")),
                    "expected_profit": alt.get(
                        "expectedProfit", alt.get("expected_profit", 0)
                    ),
                    "fill_probability": alt.get(
                        "fillProbability", alt.get("fill_probability", 0)
                    ),
                    "expected_hours": alt.get(
                        "expectedHours", alt.get("expected_hours", 4)
                    ),
                }
                for alt in alternatives[:3]
            ]

            return {"alternative_items": alt_list}

        except Exception as e:
            logger.debug(f"Could not find alternatives: {e}")
            return None

    def _calculate_liquidation(
        self,
        order_type: str,
        user_price: int,
        quantity: int,
        current_high: int,
        current_low: int,
    ) -> Optional[dict]:
        """Calculate instant liquidation details."""
        if order_type == "buy":
            # For buy order: liquidate means cancel and don't buy.
            # Show the current instant-buy price for context.
            instant_price = current_high
            # If they've placed a buy but haven't filled, no loss yet
            loss_amount = 0
        else:
            # For sell order: liquidate means sell at current low (instant-sell)
            instant_price = current_low
            loss_amount = max(0, (user_price - current_low) * quantity)

        return {
            "instant_price": instant_price,
            "loss_amount": loss_amount,
        }

    def _determine_action(
        self,
        fill_prob: float,
        price_move_pct: float,
        price_moved_favorably: bool,
        time_elapsed: int,
        predicted_window_hours: int,
        has_alternatives: bool,
    ) -> str:
        """Apply decision logic to determine recommended action."""
        predicted_window_minutes = predicted_window_hours * 60
        time_ratio = time_elapsed / max(1, predicted_window_minutes)

        # Case 1: High probability, wait it out
        if fill_prob > self.HIGH_FILL_PROB_THRESHOLD and time_ratio < 0.5:
            return "wait"

        # Case 2: Very low probability after significant time
        if fill_prob < self.LOW_FILL_PROB_THRESHOLD and time_ratio > 0.5:
            return "abort_retry" if has_alternatives else "liquidate"

        # Case 3: Price moved unfavorably, consider adjusting
        if (
            not price_moved_favorably
            and price_move_pct > self.UNFAVORABLE_MOVE_THRESHOLD
        ):
            return "adjust_price"

        # Case 4: Moderate probability within the expected window, wait.
        # If we're past the predicted window, bias toward adjusting.
        if fill_prob > 0.4 and time_ratio < 1.0:
            return "wait"

        # Case 5: Low-moderate probability, adjust to improve
        return "adjust_price"

    def _calculate_confidence(
        self,
        fill_prob: float,
        predictions_df,
        time_elapsed: int,
        predicted_window_hours: int,
    ) -> float:
        """Calculate overall confidence in recommendation."""
        # Base confidence from fill probability certainty
        # High or low fill prob = more certain
        fill_certainty = abs(fill_prob - 0.5) * 2  # 0-1 scale

        # Data freshness factor
        if not predictions_df.empty and "prediction_time" in predictions_df.columns:
            # Predictions are usually fresh, assume good
            freshness = 0.9
        else:
            freshness = 0.7

        # Time factor - more confident early in window
        time_ratio = time_elapsed / max(1, predicted_window_hours * 60)
        time_confidence = max(0.5, 1 - time_ratio * 0.5)

        # Weighted average
        confidence = 0.4 * fill_certainty + 0.3 * freshness + 0.3 * time_confidence

        return max(0.3, min(0.95, confidence))

    def _generate_reasoning(
        self,
        action: str,
        fill_prob: float,
        price_move_pct: float,
        moved_favorably: bool,
        trend: str,
        order_type: str,
        time_elapsed_minutes: int,
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        parts = []

        # Fill probability context
        if fill_prob >= 0.7:
            parts.append(f"Fill probability is strong at {fill_prob:.0%}")
        elif fill_prob >= 0.4:
            parts.append(f"Fill probability is moderate at {fill_prob:.0%}")
        else:
            parts.append(f"Fill probability is low at {fill_prob:.0%}")

        # Price movement
        if price_move_pct > 0.01:
            direction = "favorably" if moved_favorably else "unfavorably"
            parts.append(f"Price has moved {price_move_pct:.1%} {direction}")

        # Trend
        if trend != "Stable":
            parts.append(f"Market trend is {trend.lower()}")

        # Time context
        hours = time_elapsed_minutes / 60
        if hours >= 1:
            parts.append(f"Order has been active for {hours:.1f} hours")
        else:
            parts.append(f"Order has been active for {time_elapsed_minutes} minutes")

        # Action-specific reasoning
        if action == "wait":
            parts.append("Waiting is recommended as conditions are favorable")
        elif action == "adjust_price":
            parts.append("Adjusting price would improve fill probability")
        elif action == "abort_retry":
            parts.append("Better opportunities may be available with different items")
        elif action == "liquidate":
            parts.append("Consider cutting losses and moving on")

        return ". ".join(parts) + "."

    def _error_response(self, message: str) -> dict:
        """Generate an error response."""
        return {
            "action": "wait",
            "confidence": 0.3,
            "current_fill_probability": 0.5,
            "recommendations": {},
            "reasoning": f"Unable to evaluate order: {message}. Recommend waiting.",
        }
