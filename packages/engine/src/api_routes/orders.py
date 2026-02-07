"""Active order evaluation endpoints."""


from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..api_dependencies import get_engine
from ..api_models import (
    AbortRetryRecommendation,
    AdjustPriceRecommendation,
    LiquidateRecommendation,
    OrderRecommendations,
    OrderUpdateRequest,
    OrderUpdateResponse,
    WaitRecommendation,
)
from ..api_security import limiter, verify_api_key
from ..api_utils import is_valid_sha256
from ..config import config
from ..logging_config import get_logger
from ..recommendation_engine import RecommendationEngine

logger = get_logger(__name__)

router = APIRouter()


@router.post("/api/v1/recommendations/update", response_model=OrderUpdateResponse)
@limiter.limit(config.rate_limit_outcomes)
async def evaluate_active_order(
    request: Request,
    body: OrderUpdateRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Evaluate an active/pending order and recommend action."""
    if body.user_id and not is_valid_sha256(body.user_id):
        raise HTTPException(status_code=400, detail="user_id must be SHA256 hash (64 hex characters)")

    try:
        result = engine.evaluate_active_order(
            item_id=body.item_id,
            order_type=body.order_type,
            user_price=body.user_price,
            quantity=body.quantity,
            time_elapsed_minutes=body.time_elapsed_minutes,
            user_id=body.user_id,
        )

        recommendations = OrderRecommendations(
            adjust_price=(
                AdjustPriceRecommendation(**result["recommendations"]["adjust_price"])
                if result["recommendations"].get("adjust_price")
                else None
            ),
            wait=(
                WaitRecommendation(**result["recommendations"]["wait"])
                if result["recommendations"].get("wait")
                else None
            ),
            abort_retry=(
                AbortRetryRecommendation(**result["recommendations"]["abort_retry"])
                if result["recommendations"].get("abort_retry")
                else None
            ),
            liquidate=(
                LiquidateRecommendation(**result["recommendations"]["liquidate"])
                if result["recommendations"].get("liquidate")
                else None
            ),
        )

        logger.info(
            "Order evaluation completed",
            item_id=body.item_id,
            order_type=body.order_type,
            action=result["action"],
            confidence=result["confidence"],
        )

        return OrderUpdateResponse(
            action=result["action"],
            confidence=result["confidence"],
            current_fill_probability=result["current_fill_probability"],
            recommendations=recommendations,
            reasoning=result["reasoning"],
        )
    except Exception as e:
        logger.error("Error evaluating order", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to evaluate order: {str(e)}")

