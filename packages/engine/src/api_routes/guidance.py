"""Web UI guidance (check-in) endpoints."""


from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..api_dependencies import get_engine
from ..api_models import GuidanceAction, GuidanceRequest, GuidanceResponse
from ..api_security import limiter, verify_api_key
from ..config import config
from ..logging_config import get_logger
from ..recommendation_engine import RecommendationEngine

logger = get_logger(__name__)

router = APIRouter()


@router.post("/api/v1/guidance", response_model=GuidanceResponse)
@limiter.limit(config.rate_limit_outcomes)
async def get_guidance(
    request: Request,
    body: GuidanceRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
) -> GuidanceResponse:
    """Get prescriptive guidance for an active trade."""
    try:
        result = engine.evaluate_active_order(
            item_id=body.item_id,
            order_type=body.order_type,
            user_price=body.user_price,
            quantity=body.quantity,
            time_elapsed_minutes=body.time_elapsed_minutes,
        )

        action_map = {
            "wait": GuidanceAction.hold,
            "adjust_price": GuidanceAction.relist,
            "abort_retry": GuidanceAction.exit,
            "liquidate": GuidanceAction.sell_now,
        }
        action = action_map.get(result["action"], GuidanceAction.hold)

        params = None
        if action == GuidanceAction.relist and result["recommendations"].get("adjust_price"):
            adj = result["recommendations"]["adjust_price"]
            params = {
                "newPrice": adj["suggested_price"],
                "priceDelta": abs(adj["suggested_price"] - body.user_price),
                "expectedSpeedup": "faster fill",
            }

        wait_rec = result["recommendations"].get("wait", {})
        expected_hours = wait_rec.get("estimated_fill_time_minutes", 60) / 60
        if body.reported_progress < 30:
            next_check = max(30, int(expected_hours * 15))
        else:
            next_check = max(60, int(expected_hours * 25))

        logger.info(
            "Guidance computed",
            item_id=body.item_id,
            order_type=body.order_type,
            action=action.value,
            confidence=result["confidence"],
        )

        return GuidanceResponse(
            action=action,
            reason=result["reasoning"],
            confidence=result["confidence"],
            params=params,
            next_check_in_minutes=next_check,
        )
    except Exception as e:
        logger.error("Error getting guidance", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get guidance: {str(e)}")
