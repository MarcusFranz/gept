"""Recommendation feedback endpoints."""


import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import and_, func, select
from sqlalchemy.engine import Engine

from ..api_dependencies import get_feedback_db_engine
from ..api_models import (
    FeedbackAnalyticsResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackTypeSummary,
)
from ..api_security import limiter, verify_api_key
from ..api_utils import is_valid_sha256
from ..config import config
from ..logging_config import get_logger
from ..schema import recommendation_feedback

logger = get_logger(__name__)

router = APIRouter()


@router.post("/api/v1/feedback", response_model=FeedbackResponse)
@limiter.limit(config.rate_limit_outcomes)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Submit structured feedback on a recommendation."""
    if not is_valid_sha256(body.userId):
        raise HTTPException(status_code=400, detail="userId must be SHA256 hash (64 hex characters)")

    try:
        submitted_at = datetime.fromisoformat(body.submittedAt.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid submittedAt timestamp: {e}")

    # recId format validation (rec_{item_id}_{YYYYMMDDHH})
    if body.recId and not re.match(r"^rec_\d+_\d{10}$", body.recId):
        raise HTTPException(
            status_code=400,
            detail="Invalid recId format. Expected: rec_{item_id}_{YYYYMMDDHH}",
        )

    outcome_db_engine = get_feedback_db_engine(request)

    f = recommendation_feedback
    stmt = (
        f.insert()
        .values(
            user_id_hash=body.userId,
            rec_id=body.recId,
            item_id=body.itemId,
            item_name=body.itemName,
            offset_pct=body.offsetPct,
            feedback_type=body.feedbackType,
            side=body.side,
            notes=body.notes,
            recommended_price=body.recommendedPrice,
            actual_price=body.actualPrice,
            submitted_at=submitted_at,
        )
        .returning(f.c.id)
    )

    try:
        with outcome_db_engine.connect() as conn:
            result = conn.execute(stmt)
            row = result.fetchone()
            feedback_id = row[0] if row else None
            conn.commit()

        logger.info(
            "Recorded feedback",
            item_name=body.itemName,
            feedback_type=body.feedbackType,
            user_hash=body.userId,
        )

        return FeedbackResponse(success=True, message="Feedback recorded", feedbackId=feedback_id)
    except Exception as e:
        logger.error("Error recording feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get("/api/v1/feedback/analytics", response_model=FeedbackAnalyticsResponse)
@limiter.limit(config.rate_limit_outcomes)
async def get_feedback_analytics(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    period: Literal["week", "month", "all"] = Query(
        default="week",
        description="Analytics period: week (7 days), month (30 days), or all time",
    ),
    item_id: Optional[int] = Query(default=None, description="Filter by specific item ID"),
    outcome_db_engine: Engine = Depends(get_feedback_db_engine),
):
    """Get aggregated feedback analytics."""
    now = datetime.now(timezone.utc)
    if period == "week":
        start_date = now - timedelta(days=7)
    elif period == "month":
        start_date = now - timedelta(days=30)
    else:
        start_date = None

    try:
        with outcome_db_engine.connect() as conn:
            f = recommendation_feedback

            type_conditions = []
            if start_date:
                type_conditions.append(f.c.submitted_at >= start_date)
            if item_id is not None:
                type_conditions.append(f.c.item_id == item_id)

            type_query = select(f.c.feedback_type, func.count().label("count")).group_by(f.c.feedback_type).order_by(func.count().desc())
            if type_conditions:
                type_query = type_query.where(and_(*type_conditions))

            type_rows = conn.execute(type_query).fetchall()

            total_feedback = sum(row[1] for row in type_rows)
            by_type = [
                FeedbackTypeSummary(
                    feedbackType=row[0],
                    count=row[1],
                    percentage=(round(row[1] / total_feedback * 100, 1) if total_feedback > 0 else 0),
                )
                for row in type_rows
            ]

            top_items_query = (
                select(f.c.item_id, f.c.item_name, func.count().label("count"))
                .group_by(f.c.item_id, f.c.item_name)
                .order_by(func.count().desc())
                .limit(10)
            )
            if start_date:
                top_items_query = top_items_query.where(f.c.submitted_at >= start_date)

            top_items = [
                {"itemId": row[0], "itemName": row[1], "count": row[2]}
                for row in conn.execute(top_items_query).fetchall()
            ]

        return FeedbackAnalyticsResponse(
            period=period,
            startDate=start_date.strftime("%Y-%m-%d") if start_date else "all-time",
            endDate=now.strftime("%Y-%m-%d"),
            totalFeedback=total_feedback,
            byType=by_type,
            topItems=top_items,
        )
    except Exception as e:
        logger.error("Error fetching feedback analytics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")
