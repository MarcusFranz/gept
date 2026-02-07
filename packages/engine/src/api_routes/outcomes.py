"""Trade outcome recording endpoints (ML feedback loop)."""


from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.engine import Engine

from ..api_dependencies import get_outcome_db_engine
from ..api_models import TradeOutcomeRequest, TradeOutcomeResponse
from ..api_security import limiter, verify_api_key
from ..api_utils import is_valid_sha256
from ..config import config
from ..logging_config import get_logger
from ..schema import trade_outcomes

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/api/v1/recommendations/{rec_id}/outcome",
    response_model=TradeOutcomeResponse,
)
@limiter.limit(config.rate_limit_outcomes)
async def report_trade_outcome(
    request: Request,
    rec_id: str,
    body: TradeOutcomeRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
    outcome_db_engine: Engine = Depends(get_outcome_db_engine),
):
    """Record a trade outcome for the ML feedback loop."""
    if rec_id != body.recId:
        raise HTTPException(status_code=400, detail="rec_id in URL does not match recId in request body")

    if not is_valid_sha256(body.userId):
        raise HTTPException(status_code=400, detail="userId must be SHA256 hash (64 hex characters)")

    try:
        reported_at = datetime.fromisoformat(body.reportedAt.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid reportedAt timestamp: {e}")

    t = trade_outcomes
    stmt = t.insert().values(
        user_id_hash=body.userId,
        rec_id=body.recId,
        item_id=body.itemId,
        item_name=body.itemName,
        buy_price=body.buyPrice,
        sell_price=body.sellPrice,
        quantity=body.quantity,
        actual_profit=body.actualProfit,
        reported_at=reported_at,
    )

    try:
        with outcome_db_engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        logger.info(
            "Recorded trade outcome",
            rec_id=rec_id,
            item_name=body.itemName,
            profit_gp=body.actualProfit,
        )
        return TradeOutcomeResponse(success=True, message="Outcome recorded")
    except Exception as e:
        logger.error("Error recording trade outcome", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to record trade outcome: {str(e)}")
