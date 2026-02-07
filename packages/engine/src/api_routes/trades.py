"""Active trade polling endpoints."""


from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..api_dependencies import get_engine
from ..api_models import NewItemRecommendation, TradeUpdate, TradeUpdatesResponse
from ..api_security import limiter, verify_api_key
from ..api_utils import is_valid_sha256
from ..config import config
from ..logging_config import get_logger
from ..recommendation_engine import RecommendationEngine

logger = get_logger(__name__)

router = APIRouter()


@router.get("/api/v1/trades/updates", response_model=TradeUpdatesResponse)
@limiter.limit(config.rate_limit_trade_updates)
async def get_trade_updates(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    tradeIds: str = Query(..., description="Comma-separated trade/recommendation IDs to check"),
    user_id: Optional[str] = Query(default=None, description="Hashed user ID for crowding awareness"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Poll for update recommendations on active trades."""
    if user_id and not is_valid_sha256(user_id):
        raise HTTPException(status_code=400, detail="user_id must be SHA256 hash (64 hex characters)")

    trade_id_list = [tid.strip() for tid in tradeIds.split(",") if tid.strip()]
    if not trade_id_list:
        raise HTTPException(status_code=400, detail="No valid trade IDs provided")

    if len(trade_id_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 trade IDs per request")

    updates: list[TradeUpdate] = []
    update_counter = 0

    for trade_id in trade_id_list:
        try:
            rec = engine.get_recommendation_by_id(trade_id)
            if rec is None:
                continue

            result = engine.evaluate_active_order(
                item_id=rec["itemId"],
                order_type="sell",
                user_price=rec["sellPrice"],
                quantity=rec.get("quantity", 1),
                time_elapsed_minutes=30,
                user_id=user_id,
            )

            action = result["action"]
            confidence = result["confidence"]

            if action == "wait":
                continue

            if confidence >= 0.85:
                urgency = "high"
            elif confidence >= 0.65:
                urgency = "medium"
            else:
                urgency = "low"

            update_counter += 1
            update_id = f"update_{update_counter:03d}"

            if action == "adjust_price" and result["recommendations"].get("adjust_price"):
                adjust = result["recommendations"]["adjust_price"]
                updates.append(
                    TradeUpdate(
                        id=update_id,
                        tradeId=trade_id,
                        type="ADJUST_PRICE",
                        reason=result["reasoning"],
                        confidence=confidence,
                        urgency=urgency,
                        newSellPrice=adjust["suggested_price"],
                        originalSellPrice=rec["sellPrice"],
                        profitDelta=adjust.get("cost_difference", 0),
                    )
                )
            elif action == "abort_retry" and result["recommendations"].get("abort_retry"):
                abort = result["recommendations"]["abort_retry"]
                alternatives = abort.get("alternative_items", [])
                if alternatives:
                    alt = alternatives[0]
                    updates.append(
                        TradeUpdate(
                            id=update_id,
                            tradeId=trade_id,
                            type="SWITCH_ITEM",
                            reason=result["reasoning"],
                            confidence=confidence,
                            urgency=urgency,
                            newItem=NewItemRecommendation(
                                itemId=alt["item_id"],
                                item=alt["item_name"],
                                buyPrice=alt.get("buy_price", rec.get("buyPrice", 0)),
                                sellPrice=alt.get("sell_price", rec.get("sellPrice", 0)),
                                quantity=1,
                                expectedProfit=alt["expected_profit"],
                            ),
                        )
                    )
            elif action == "liquidate" and result["recommendations"].get("liquidate"):
                liq = result["recommendations"]["liquidate"]
                updates.append(
                    TradeUpdate(
                        id=update_id,
                        tradeId=trade_id,
                        type="SELL_NOW",
                        reason=result["reasoning"],
                        confidence=confidence,
                        urgency=urgency,
                        adjustedSellPrice=liq["instant_price"],
                    )
                )
        except Exception as e:
            logger.warning("Error evaluating trade", trade_id=trade_id, error=str(e))
            continue

    if any(u.urgency == "high" for u in updates):
        next_check_in = 15
    elif updates:
        next_check_in = 30
    else:
        next_check_in = 60

    logger.info(
        "Trade updates computed",
        checked=len(trade_id_list),
        updates=len(updates),
    )

    return TradeUpdatesResponse(updates=updates, nextCheckIn=next_check_in)

