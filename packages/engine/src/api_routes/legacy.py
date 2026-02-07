"""Web-frontend compatibility routes.

These endpoints exist for the gept-gg web frontend and map to the /api/v1 surface.
"""


from typing import Literal, Optional, Union

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.engine import Engine

from ..api_dependencies import get_engine, get_outcome_db_engine
from ..api_models import (
    AllRecommendationsResponse,
    HealthResponse,
    ItemSearchResult,
    RecommendationResponse,
    RecommendationsWithMetadata,
    TradeOutcomeRequest,
    TradeOutcomeResponse,
    TradeOutcomeWebRequest,
)
from ..api_security import limiter, verify_api_key
from ..config import config
from ..recommendation_engine import RecommendationEngine
from .health import health_check
from .items import search_items
from .outcomes import report_trade_outcome
from .recommendations import get_recommendation_by_item_id, get_recommendations_get

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
@limiter.limit(config.rate_limit_health)
async def health_check_alias(request: Request):
    """Health check endpoint alias for web frontend."""
    return await health_check(request)


@router.get(
    "/recommendations",
    response_model=Union[
        list[RecommendationResponse],
        RecommendationsWithMetadata,
        AllRecommendationsResponse,
    ],
)
@limiter.limit(config.rate_limit_recommendations)
async def recommendations_web(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    user_hash: Optional[str] = Query(default=None, description="User hash for crowding tracking"),
    capital: int = Query(ge=1000, description="Available capital in gp"),
    style: Literal["passive", "hybrid", "active"] = Query(default="hybrid", description="Trading style"),
    risk: Literal["low", "medium", "high"] = Query(default="medium", description="Risk tolerance"),
    count: int = Query(default=4, ge=1, le=20, description="Number of recommendations"),
    exclude_items: Optional[str] = Query(default=None, description="Comma-separated item IDs to exclude"),
    max_offset_pct: Optional[float] = Query(
        default=None, ge=0.0125, le=0.0250, description="Maximum margin percentage"
    ),
    min_offset_pct: Optional[float] = Query(
        default=None, ge=0.0125, le=0.0250, description="Minimum margin percentage"
    ),
    include_metadata: bool = Query(default=False, description="Include freshness metadata"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get recommendations for web frontend."""
    return await get_recommendations_get(
        request=request,
        _api_key=_api_key,
        style=style,
        capital=capital,
        risk=risk,
        slots=count,
        user_tier="free",
        exclude=None,
        user_id=user_hash,
        exclude_item_ids=exclude_items,
        offset_pct=None,
        min_offset_pct=min_offset_pct,
        max_offset_pct=max_offset_pct,
        max_hour_offset=None,
        min_ev=None,
        include_metadata=include_metadata,
        return_all=False,
        use_beta_model=False,
        engine=engine,
    )


@router.get("/item/{item_id}")
@limiter.limit(config.rate_limit_items)
async def get_item_web(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    capital: Optional[int] = Query(default=None, ge=1000, description="Available capital in gp"),
    risk: Optional[Literal["low", "medium", "high"]] = Query(default=None, description="Risk tolerance"),
    style: Optional[Literal["passive", "hybrid", "active"]] = Query(default=None, description="Trading style"),
    include_price_history: bool = Query(default=False, description="Include price history for charts"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Web-friendly endpoint alias for item recommendations by ID."""
    return await get_recommendation_by_item_id(
        request=request,
        item_id=item_id,
        _api_key=_api_key,
        capital=capital,
        risk=risk,
        style=style,
        slots=None,
        include_price_history=include_price_history,
        engine=engine,
    )


@router.get("/search-items", response_model=list[ItemSearchResult])
@limiter.limit(config.rate_limit_search)
async def search_items_web(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=25, description="Max results"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Web-friendly endpoint alias for /api/v1/items/search."""
    return await search_items(
        request=request,
        _api_key=_api_key,
        q=q,
        limit=limit,
        no_cache=False,
        engine=engine,
    )


@router.post("/trade-outcome", response_model=TradeOutcomeResponse)
@limiter.limit(config.rate_limit_outcomes)
async def report_trade_outcome_web(
    request: Request,
    body: TradeOutcomeWebRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
    outcome_db_engine: Engine = Depends(get_outcome_db_engine),
):
    """Record a trade outcome from web frontend."""
    outcome_body = TradeOutcomeRequest(
        userId=body.userId,
        itemId=body.itemId,
        itemName=body.itemName,
        recId=body.rec_id,
        buyPrice=body.buyPrice,
        sellPrice=body.sellPrice,
        quantity=body.quantity,
        actualProfit=body.actualProfit,
        reportedAt=body.reportedAt,
    )
    return await report_trade_outcome(
        request=request,
        rec_id=body.rec_id,
        body=outcome_body,
        _api_key=_api_key,
        outcome_db_engine=outcome_db_engine,
    )

