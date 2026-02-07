"""User analytics endpoints."""


from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import and_, case, func, select
from sqlalchemy.engine import Engine

from ..api_dependencies import get_outcome_db_engine
from ..api_models import FlipInfo, PeriodComparison, UserStatsResponse
from ..api_security import limiter, verify_api_key
from ..api_utils import is_valid_sha256
from ..config import config
from ..logging_config import get_logger
from ..schema import trade_outcomes

logger = get_logger(__name__)

router = APIRouter()


@router.get("/api/v1/users/{hashed_user_id}/stats", response_model=UserStatsResponse)
@limiter.limit(config.rate_limit_outcomes)
async def get_user_stats(
    request: Request,
    hashed_user_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
    period: Literal["week", "month", "all"] = Query(
        default="week",
        description="Stats period: week (7 days), month (30 days), or all time",
    ),
    outcome_db_engine: Engine = Depends(get_outcome_db_engine),
):
    """Get aggregated trade statistics for a user."""
    if not is_valid_sha256(hashed_user_id):
        raise HTTPException(
            status_code=400,
            detail="hashed_user_id must be SHA256 hash (64 hex characters)",
        )

    now = datetime.now(timezone.utc)
    if period == "week":
        start_date = now - timedelta(days=7)
        prev_start = start_date - timedelta(days=7)
        prev_end = start_date
    elif period == "month":
        start_date = now - timedelta(days=30)
        prev_start = start_date - timedelta(days=30)
        prev_end = start_date
    else:
        start_date = None
        prev_start = None
        prev_end = None

    t = trade_outcomes
    try:
        with outcome_db_engine.connect() as conn:
            def _stats_query(uid: str, date_start=None, date_end=None):
                conditions = [t.c.user_id_hash == uid]
                if date_start:
                    conditions.append(t.c.reported_at >= date_start)
                if date_end:
                    conditions.append(t.c.reported_at < date_end)
                return select(
                    func.count().label("total_trades"),
                    func.coalesce(func.sum(t.c.actual_profit), 0).label("total_profit"),
                    func.coalesce(func.sum(case((t.c.actual_profit > 0, 1), else_=0)), 0).label("winning_trades"),
                ).where(and_(*conditions))

            result = conn.execute(_stats_query(hashed_user_id, start_date, now if start_date else None))
            row = result.fetchone()
            total_trades = row[0] if row else 0
            total_profit = row[1] if row else 0
            winning_trades = row[2] if row else 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            def _flip_query(uid: str, order_dir: str, date_start=None, date_end=None):
                conditions = [t.c.user_id_hash == uid]
                if date_start:
                    conditions.append(t.c.reported_at >= date_start)
                if date_end:
                    conditions.append(t.c.reported_at < date_end)
                order_col = t.c.actual_profit.desc() if order_dir == "best" else t.c.actual_profit.asc()
                return select(t.c.item_name, t.c.actual_profit).where(and_(*conditions)).order_by(order_col).limit(1)

            best_flip = None
            if total_trades > 0:
                best_row = conn.execute(_flip_query(hashed_user_id, "best", start_date, now if start_date else None)).fetchone()
                if best_row:
                    best_flip = FlipInfo(itemName=best_row[0], profit=best_row[1])

            worst_flip = None
            if total_trades > 0:
                worst_row = conn.execute(_flip_query(hashed_user_id, "worst", start_date, now if start_date else None)).fetchone()
                if worst_row:
                    worst_flip = FlipInfo(itemName=worst_row[0], profit=worst_row[1])

            comparison = None
            if prev_start and prev_end:
                prev_result = conn.execute(_stats_query(hashed_user_id, prev_start, prev_end))
                prev_row = prev_result.fetchone()
                prev_total_trades = prev_row[0] if prev_row else 0
                prev_total_profit = prev_row[1] if prev_row else 0
                prev_winning_trades = prev_row[2] if prev_row else 0
                prev_win_rate = prev_winning_trades / prev_total_trades if prev_total_trades > 0 else 0.0

                comparison = PeriodComparison(
                    profitDelta=int(total_profit) - int(prev_total_profit),
                    winRateDelta=round(win_rate - prev_win_rate, 4),
                )

        start_date_str = start_date.strftime("%Y-%m-%d") if start_date else "all-time"
        end_date_str = now.strftime("%Y-%m-%d")

        logger.info(
            "User stats computed",
            user_hash=hashed_user_id,
            period=period,
            trades=total_trades,
            profit_gp=int(total_profit),
        )

        return UserStatsResponse(
            period=period,
            startDate=start_date_str,
            endDate=end_date_str,
            totalProfit=int(total_profit),
            totalTrades=int(total_trades),
            winRate=round(win_rate, 4),
            bestFlip=best_flip,
            worstFlip=worst_flip,
            comparisonToPreviousPeriod=comparison,
        )
    except Exception as e:
        logger.error("Error fetching user stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch user stats: {str(e)}")
