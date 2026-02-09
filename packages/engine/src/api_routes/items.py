"""Item lookup endpoints (search, price history, price recommendations)."""


import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request

from ..api_dependencies import get_engine
from ..api_models import (
    ExtendedPriceHistoryPoint,
    ItemPriceLookupResponse,
    ItemSearchResult,
    PriceHistoryResponse,
)
from ..api_security import limiter, verify_api_key
from ..cache import get_cached, set_cached
from ..config import config
from ..logging_config import get_logger
from ..recommendation_engine import RecommendationEngine

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/api/v1/items/{item_id}/price",
    response_model=ItemPriceLookupResponse,
)
@limiter.limit(config.rate_limit_items)
async def get_item_price_lookup(
    request: Request,
    item_id: int = Path(..., gt=0, description="OSRS item ID"),
    _api_key: Optional[str] = Depends(verify_api_key),
    side: Literal["buy", "sell"] = Query(default="buy", description="Trade side - buy or sell"),
    window: int = Query(default=24, ge=1, le=24, description="Target time window in hours"),
    offset: Optional[float] = Query(default=None, ge=0.01, le=0.03, description="Target offset percentage"),
    include_price_history: bool = Query(default=False, description="Include 24h price history for charts"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get price recommendation for any item, regardless of recommendation status."""
    try:
        result = engine.get_item_price_lookup(
            item_id=item_id,
            side=side,
            window_hours=window,
            offset_pct=offset,
            include_price_history=include_price_history,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Item {item_id} not found or has no predictions",
            )

        return ItemPriceLookupResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting price lookup for item", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/items/search", response_model=list[ItemSearchResult])
@limiter.limit(config.rate_limit_search)
async def search_items(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=25, description="Maximum number of results"),
    no_cache: bool = Query(default=False, description="Bypass cache"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Search for items by name with fuzzy matching."""
    start_time = time.monotonic()
    cache_key = f"search:{q.lower().strip()}:{limit}"

    if not no_cache:
        cached = await get_cached(cache_key)
        if cached is not None:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "search_served",
                query=q,
                results_count=len(cached),
                cache_hit=True,
                duration_ms=round(duration_ms, 2),
            )
            return cached

    try:
        matches = engine.search_items_by_name(q, limit)

        results = [
            ItemSearchResult(
                itemId=match["item_id"],
                name=match["item_name"],
                category=None,
            )
            for match in matches
        ]

        await set_cached(cache_key, [r.model_dump() for r in results], config.cache_ttl_search)

        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "search_served",
            query=q,
            results_count=len(results),
            cache_hit=False,
            duration_ms=round(duration_ms, 2),
        )

        return results
    except Exception as e:
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "search_error",
            query=q,
            error=str(e),
            duration_ms=round(duration_ms, 2),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/items/{item_id}/price-history", response_model=PriceHistoryResponse)
@limiter.limit(config.rate_limit_items)
async def get_item_price_history(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    hours: int = Query(default=24, ge=1, le=168, description="Hours of history (max 7 days)"),
    no_cache: bool = Query(default=False, description="Bypass cache"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get price history for an item with trend analysis."""
    cache_key = f"price_history:{item_id}:{hours}"

    if not no_cache:
        cached = await get_cached(cache_key)
        if cached is not None:
            return PriceHistoryResponse(**cached)

    try:
        history = engine.loader.get_extended_price_history(item_id, hours=hours)

        if not history:
            item_name = engine.loader.get_item_name(item_id)
            if item_name is None:
                raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
            return PriceHistoryResponse(
                itemId=item_id,
                itemName=item_name,
                history=[],
                trend="Stable",
                change24h=0.0,
            )

        item_name = engine.loader.get_item_name(item_id) or f"Item {item_id}"
        trend = engine.loader.get_item_trend(item_id)

        change_24h = 0.0
        if len(history) >= 2:
            first_point = history[0]
            last_point = history[-1]
            first_avg = (first_point["high"] + first_point["low"]) / 2
            last_avg = (last_point["high"] + last_point["low"]) / 2
            if first_avg > 0:
                change_24h = round(((last_avg - first_avg) / first_avg) * 100, 2)

        history_points = [
            ExtendedPriceHistoryPoint(
                timestamp=h["timestamp"],
                high=h["high"],
                low=h["low"],
                avgHigh=h["avgHigh"],
                avgLow=h["avgLow"],
            )
            for h in history
        ]

        response = PriceHistoryResponse(
            itemId=item_id,
            itemName=item_name,
            history=history_points,
            trend=trend,
            change24h=change_24h,
        )

        await set_cached(cache_key, response.model_dump(), config.cache_ttl_prices)

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting price history for item", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
