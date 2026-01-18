"""Output formatting for API responses."""

import logging
from datetime import datetime, timezone
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# Type definitions
ConfidenceLevel = Literal["high", "medium", "low", "stale", "unreliable"]
ActionType = Literal["trade", "hold", "avoid"]


class OutputFormatter:
    """Formats predictions for API response.

    Note: For detailed tax calculations including floor, cap, and rounding,
    use the calculate_tax() function from tax_calculator module.
    This class uses the simplified TAX_RATE for EV calculations.
    """

    # GE tax rate (only applies to items with sell price >= 50gp)
    TAX_RATE = 0.02
    TAX_THRESHOLD = 50  # Items below this price are tax-exempt

    def __init__(
        self,
        ev_threshold: float = 0.001,
        confidence_high_auc: float = 0.75,
        confidence_medium_auc: float = 0.60,
        data_stale_seconds: int = 600,
    ):
        """Initialize formatter.

        Args:
            ev_threshold: Minimum EV threshold for recommendations
            confidence_high_auc: AUC threshold for high confidence
            confidence_medium_auc: AUC threshold for medium confidence
            data_stale_seconds: Seconds before data is considered stale
        """
        self.ev_threshold = ev_threshold
        self.confidence_high_auc = confidence_high_auc
        self.confidence_medium_auc = confidence_medium_auc
        self.data_stale_seconds = data_stale_seconds

    def calculate_tax(self, sell_price: int, quantity: int) -> int:
        """Calculate GE tax for a transaction.

        GE tax is 2% of the sell price, but only applies to items
        with sell price >= 50gp. Items below 50gp are tax-exempt.

        Args:
            sell_price: Price per item when selling
            quantity: Number of items being sold

        Returns:
            Total tax amount in gp
        """
        if sell_price < self.TAX_THRESHOLD:
            return 0
        return int(sell_price * quantity * self.TAX_RATE)

    def calculate_expected_value(
        self, fill_probability: float, offset_pct: float
    ) -> float:
        """Calculate expected value of a flip.

        EV = P(fill) x net_profit_if_fill
        Net profit = gross_margin - tax
        Gross margin = buy_offset + sell_offset = 2 x offset

        Note: This uses simplified 2% tax for EV estimation.
        For actual tax calculations with floor/cap/rounding,
        use calculate_ge_tax() method.

        Args:
            fill_probability: Probability that both buy and sell fill
            offset_pct: Offset percentage from current price

        Returns:
            Expected value as a fraction
        """
        gross_margin = offset_pct * 2
        net_margin = gross_margin - self.TAX_RATE
        return fill_probability * net_margin

    def calculate_suggested_prices(
        self, current_low: int, current_high: int, offset_pct: float
    ) -> tuple[int, int]:
        """Calculate suggested buy and sell prices.

        Buy below current low, sell above current high.

        Applies 50gp boundary optimization: at exact multiples of 50gp, the 2% GE tax
        rounds down, creating an exploitable asymmetry. Selling at (50×N - 1) instead
        of (50×N) yields the seller identical net proceeds but costs buyers 1gp less,
        which improves fill speed.

        Example: Selling at 100gp yields 98gp after tax (100 - 2gp tax).
                 Selling at 99gp also yields 98gp after tax (99 - 1gp tax).
                 Buyers save 1gp and orders fill faster at 99gp.

        Args:
            current_low: Current instant-sell price
            current_high: Current instant-buy price
            offset_pct: Offset percentage

        Returns:
            Tuple of (suggested_buy, suggested_sell)
        """
        suggested_buy = int(current_low * (1 - offset_pct))
        suggested_sell = int(current_high * (1 + offset_pct))

        # Ensure prices are positive and in correct order
        suggested_buy = max(1, suggested_buy)
        suggested_sell = max(suggested_buy + 1, suggested_sell)

        # 50gp boundary optimization: round down multiples of 50
        # Seller nets identical proceeds but buyer pays 1gp less (faster fills)
        # Guard: only apply if it doesn't violate buy < sell invariant
        if (
            suggested_sell % 50 == 0
            and suggested_sell > 1
            and (suggested_sell - 1) > suggested_buy
        ):
            suggested_sell -= 1

        return suggested_buy, suggested_sell

    def assign_confidence(
        self, auc: Optional[float], data_age_seconds: float, tier: int
    ) -> ConfidenceLevel:
        """Assign confidence rating based on model quality and data freshness.

        Args:
            auc: Model AUC score
            data_age_seconds: Age of the most recent data in seconds
            tier: Item tier (1, 2, or 3)

        Returns:
            Confidence level
        """
        # Check for stale data first
        if data_age_seconds > self.data_stale_seconds:
            return "stale"

        # Check for unreliable model
        if auc is None or auc < 0.52:
            return "unreliable"

        # High confidence: AUC > 0.75, data < 2 min, tier 1
        if auc > self.confidence_high_auc and data_age_seconds < 120 and tier == 1:
            return "high"

        # Medium confidence: AUC > 0.60, data < 5 min, tier 1-2
        if (
            auc > self.confidence_medium_auc
            and data_age_seconds < 300
            and tier in [1, 2]
        ):
            return "medium"

        # Low confidence: AUC > 0.52, data < 10 min
        if auc > 0.52 and data_age_seconds < 600:
            return "low"

        return "unreliable"

    def select_recommendation(self, predictions: list[dict]) -> Optional[dict]:
        """Choose the best configuration based on EV.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Best prediction or None if none meet threshold
        """
        if not predictions:
            return None

        # Filter to positive EV predictions
        positive_ev = [p for p in predictions if p.get("expected_value", 0) > 0]
        if not positive_ev:
            return None

        # Sort by EV descending
        sorted_preds = sorted(
            positive_ev,
            key=lambda p: p.get("expected_value", 0),
            reverse=True,
        )

        return sorted_preds[0]

    def determine_action(
        self, best_prediction: Optional[dict], confidence: ConfidenceLevel
    ) -> ActionType:
        """Determine recommended action.

        Args:
            best_prediction: Best prediction dictionary
            confidence: Confidence level

        Returns:
            Recommended action
        """
        if confidence in ["stale", "unreliable"]:
            return "avoid"

        if best_prediction is None:
            return "hold"

        ev = best_prediction.get("expected_value", 0)
        if ev < self.ev_threshold:
            return "hold"

        return "trade"

    def format_prediction(
        self,
        item_id: int,
        item_name: str,
        tier: int,
        current_prices: dict,
        predictions: list[dict],
        model_metadata: dict,
        data_age_seconds: float,
    ) -> dict:
        """Format a single item's predictions.

        Args:
            item_id: OSRS item ID
            item_name: Item display name
            tier: Item tier
            current_prices: Dict with high, low, high_volume, low_volume
            predictions: List of raw predictions from model runner
            model_metadata: Model metadata dictionary
            data_age_seconds: Age of price data in seconds

        Returns:
            Formatted prediction dictionary
        """
        current_high = current_prices.get("high", 0)
        current_low = current_prices.get("low", 0)
        current_mid = (
            (current_high + current_low) / 2 if current_high and current_low else 0
        )
        spread_pct = (
            (current_high - current_low) / current_mid if current_mid > 0 else 0
        )

        # Calculate volume (last hour approximation)
        high_vol = current_prices.get("high_volume", 0) or 0
        low_vol = current_prices.get("low_volume", 0) or 0
        volume_1h = (high_vol + low_vol) * 60  # Extrapolate from 1-min to 1-hour

        # Format each prediction
        formatted_predictions = []
        avg_auc = model_metadata.get("avg_auc", 0)

        for pred in predictions:
            hour_window = pred["hour_window"]
            offset_pct = pred["offset_pct"]
            fill_prob = pred["fill_probability"]

            ev = self.calculate_expected_value(fill_prob, offset_pct)
            buy_price, sell_price = self.calculate_suggested_prices(
                current_low, current_high, offset_pct
            )

            # Get specific model AUC if available
            model_auc = pred.get("auc", avg_auc)
            confidence = self.assign_confidence(model_auc, data_age_seconds, tier)

            formatted_predictions.append(
                {
                    "hour_window": hour_window,
                    "offset_pct": offset_pct,
                    "fill_probability": round(fill_prob, 4),
                    "expected_value": round(ev, 6),
                    "confidence": confidence,
                    "suggested_buy": buy_price,
                    "suggested_sell": sell_price,
                }
            )

        # Select best recommendation
        best = self.select_recommendation(formatted_predictions)
        overall_confidence = self.assign_confidence(avg_auc, data_age_seconds, tier)
        action = self.determine_action(best, overall_confidence)

        recommendation = None
        if best:
            recommendation = {
                "action": action,
                "confidence": overall_confidence,
                "best_config": {
                    "hour_window": best["hour_window"],
                    "offset_pct": best["offset_pct"],
                },
                "fill_probability": best["fill_probability"],
                "expected_value": best["expected_value"],
                "suggested_buy": best["suggested_buy"],
                "suggested_sell": best["suggested_sell"],
                "reasoning": self._generate_reasoning(best, action, overall_confidence),
            }
        else:
            recommendation = {
                "action": action,
                "confidence": overall_confidence,
                "best_config": None,
                "fill_probability": 0,
                "expected_value": 0,
                "suggested_buy": None,
                "suggested_sell": None,
                "reasoning": "No profitable configurations found",
            }

        return {
            "item_id": item_id,
            "item_name": item_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tier": tier,
            "current_market": {
                "high": current_high,
                "low": current_low,
                "spread_pct": round(spread_pct, 4),
                "volume_1h": int(volume_1h),
            },
            "predictions": formatted_predictions,
            "recommendation": recommendation,
            "model_metadata": {
                "model_version": model_metadata.get("model_version", "v1.0.0"),
                "avg_auc": model_metadata.get("avg_auc"),
                "last_trained": model_metadata.get("last_trained"),
                "data_freshness_seconds": round(data_age_seconds, 1),
            },
        }

    def _generate_reasoning(
        self, prediction: dict, action: ActionType, confidence: ConfidenceLevel
    ) -> str:
        """Generate human-readable reasoning for recommendation.

        Args:
            prediction: Best prediction
            action: Recommended action
            confidence: Confidence level

        Returns:
            Reasoning string
        """
        ev = prediction.get("expected_value", 0)
        fill_prob = prediction.get("fill_probability", 0)
        hour = prediction.get("hour_window", 0)

        if action == "avoid":
            return "Data is stale or model reliability is low"

        if action == "hold":
            return "Expected value below threshold for active trading"

        parts = []
        parts.append(f"Highest EV configuration ({ev * 100:.2f}%)")

        if confidence == "high":
            parts.append("high model confidence")
        elif confidence == "medium":
            parts.append("moderate model confidence")
        else:
            parts.append("lower model confidence")

        parts.append(f"{fill_prob * 100:.1f}% fill probability within {hour}h")

        return "; ".join(parts)

    def format_for_discord_bot(
        self, prediction: dict, capital: int, buy_limit: Optional[int] = None
    ) -> Optional[dict]:
        """Format prediction for Discord bot consumption.

        Converts internal format to the Recommendation interface expected
        by the Discord bot.

        Args:
            prediction: Internal prediction format
            capital: User's available capital
            buy_limit: Optional GE buy limit for the item

        Returns:
            Dictionary matching Discord bot's Recommendation interface
        """
        rec = prediction.get("recommendation", {})
        market = prediction.get("current_market", {})

        if rec.get("action") != "trade" or rec.get("suggested_buy") is None:
            return None

        buy_price = rec["suggested_buy"]
        sell_price = rec["suggested_sell"]
        fill_prob = rec["fill_probability"]

        # Calculate quantity based on capital
        max_quantity = capital // buy_price if buy_price > 0 else 0

        # Apply buy limit if provided
        if buy_limit and buy_limit > 0:
            max_quantity = min(max_quantity, buy_limit)
        else:
            # Conservative fallback cap if no buy limit known
            max_quantity = min(max_quantity, 1000)

        quantity = max_quantity

        if quantity < 1:
            return None

        capital_required = buy_price * quantity

        # Calculate profit accounting for GE tax
        revenue = sell_price * quantity
        cost = buy_price * quantity
        tax = self.calculate_tax(sell_price, quantity)
        profit_if_fill = revenue - cost - tax
        expected_profit = int(profit_if_fill * fill_prob)

        # Map confidence
        confidence_map = {
            "high": "high",
            "medium": "medium",
            "low": "low",
            "stale": "low",
            "unreliable": "low",
        }

        # Determine volume tier from volume_1h
        volume = market.get("volume_1h", 0)
        if volume > 100000:
            volume_tier = "High"
        elif volume > 10000:
            volume_tier = "Medium"
        else:
            volume_tier = "Low"

        # Determine trend from model metadata or predictions
        # Use 1h return to determine trend direction
        trend = self._determine_trend(prediction)

        # Generate placeholder ID - replaced with stable ID by RecommendationStore
        item_id = prediction["item_id"]
        placeholder_id = f"rec_{item_id}_placeholder"

        return {
            "id": placeholder_id,
            "itemId": item_id,
            "item": prediction["item_name"],
            "buyPrice": buy_price,
            "sellPrice": sell_price,
            "quantity": quantity,
            "capitalRequired": capital_required,
            "expectedProfit": expected_profit,
            "confidence": confidence_map.get(rec.get("confidence", "low"), "low"),
            "volumeTier": volume_tier,
            "trend": trend,
        }

    def _determine_trend(self, prediction: dict) -> str:
        """Determine price trend from prediction data.

        Args:
            prediction: Full prediction dict

        Returns:
            Trend string: "Rising", "Stable", or "Falling"
        """
        # Default to looking at the spread and market conditions
        market = prediction.get("current_market", {})
        spread_pct = market.get("spread_pct", 0)

        # Without historical return data, infer from spread
        # Tighter spreads often indicate stable markets
        # Wider spreads can indicate volatility/movement
        if spread_pct > 0.05:  # >5% spread suggests volatility
            # Could be either direction - mark as potentially moving
            return "Stable"  # Conservative default
        elif spread_pct < 0.01:  # <1% very tight spread
            return "Stable"
        else:
            return "Stable"  # Default to stable without historical data
