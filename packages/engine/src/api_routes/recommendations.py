"""Recommendation and opportunity browsing endpoints."""


import math
import time
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..api_dependencies import get_engine
from ..api_models import (
    AllRecommendationsResponse,
    FreshnessMetadata,
    ItemPriceRefreshResponse,
    OpportunitiesListResponse,
    OpportunityFilters,
    OpportunityResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationsWithMetadata,
    WhyChip,
)
from ..api_security import limiter, verify_api_key
from ..api_utils import normalize_user_id
from ..config import config
from ..logging_config import get_logger
from ..recommendation_engine import RecommendationEngine

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/api/v1/recommendations",
    response_model=Union[
        list[RecommendationResponse],
        RecommendationsWithMetadata,
        AllRecommendationsResponse,
    ],
)
@limiter.limit(config.rate_limit_recommendations)
async def get_recommendations(
    request: Request,
    body: RecommendationRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get recommendations formatted for Discord bot."""
    # Validate slots based on user tier (only applicable when not using return_all)
    if not body.return_all:
        max_slots = 20 if body.user_tier == "unlimited" else 8
        if body.slots > max_slots:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_slots} slots allowed for {body.user_tier} tier",
            )

    start_time = time.monotonic()
    try:
        active_trades = [
            {"itemId": t.itemId, "quantity": t.quantity, "buyPrice": t.buyPrice}
            for t in body.activeTrades
        ]

        normalized_user_id = normalize_user_id(body.userId)

        if body.return_all:
            recommendations = engine.get_all_recommendations(
                style=body.style,
                capital=body.capital,
                risk=body.risk,
                active_trades=active_trades,
                user_id=normalized_user_id,
                offset_pct=body.offset_pct,
                min_offset_pct=body.min_offset_pct,
                max_offset_pct=body.max_offset_pct,
            )

            duration_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                "all_recommendations_served",
                user_hash=normalized_user_id,
                style=body.style,
                capital=body.capital,
                risk=body.risk,
                recommendations_count=len(recommendations),
                duration_ms=round(duration_ms, 2),
            )

            generated_at = datetime.now(timezone.utc)
            valid_until = generated_at + timedelta(minutes=5)

            return AllRecommendationsResponse(
                recommendations=recommendations,
                generated_at=generated_at.isoformat(),
                valid_until=valid_until.isoformat(),
                total_count=len(recommendations),
            )

        recommendations = engine.get_recommendations(
            style=body.style,
            capital=body.capital,
            risk=body.risk,
            slots=body.slots,
            active_trades=active_trades,
            user_id=normalized_user_id,
            offset_pct=body.offset_pct,
            min_offset_pct=body.min_offset_pct,
            max_offset_pct=body.max_offset_pct,
        )

        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "recommendations_served",
            user_hash=normalized_user_id,
            style=body.style,
            capital=body.capital,
            risk=body.risk,
            slots=body.slots,
            recommendations_count=len(recommendations),
            duration_ms=round(duration_ms, 2),
        )

        if body.include_metadata:
            metadata = engine.get_freshness_metadata()
            return RecommendationsWithMetadata(
                recommendations=recommendations,
                metadata=FreshnessMetadata(**metadata),
            )

        return recommendations
    except Exception as e:
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "recommendations_error",
            error=str(e),
            style=body.style,
            capital=body.capital,
            duration_ms=round(duration_ms, 2),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/recommendations",
    response_model=Union[
        list[RecommendationResponse],
        RecommendationsWithMetadata,
        AllRecommendationsResponse,
    ],
)
@limiter.limit(config.rate_limit_recommendations)
async def get_recommendations_get(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    style: Literal["passive", "hybrid", "active"] = "hybrid",
    capital: int = Query(ge=1000, description="Available capital in gp"),
    risk: Literal["low", "medium", "high"] = "medium",
    slots: int = Query(ge=1, le=20, default=4, description="Available GE slots"),
    user_tier: Literal["free", "premium", "unlimited"] = Query(
        default="free",
        description="User tier (free/premium/unlimited). Unlimited allows up to 20 slots.",
    ),
    exclude: Optional[str] = Query(
        default=None, description="Comma-separated recommendation IDs to exclude"
    ),
    user_id: Optional[str] = Query(
        default=None,
        description="User identifier for crowding tracking (SHA256 hash recommended)",
    ),
    exclude_item_ids: Optional[str] = Query(
        default=None,
        description="Comma-separated item IDs to exclude (e.g., '536,5295,4151')",
    ),
    offset_pct: Optional[float] = Query(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Specific offset percentage to filter by (0.0125 to 0.0250)",
    ),
    min_offset_pct: Optional[float] = Query(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Minimum offset percentage threshold (0.0125 to 0.0250)",
    ),
    max_offset_pct: Optional[float] = Query(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Maximum offset percentage threshold (0.0125 to 0.0250)",
    ),
    max_hour_offset: Optional[int] = Query(
        default=None,
        ge=1,
        le=48,
        description="Maximum time horizon in hours (1-48, overrides style default)",
    ),
    min_ev: Optional[float] = Query(
        default=None,
        ge=0.001,
        le=0.1,
        description="Minimum expected value threshold (0.001-0.1)",
    ),
    include_metadata: bool = Query(default=False, description="Include freshness metadata in response"),
    return_all: bool = Query(
        default=False,
        description="If true, return all viable flips sorted by score instead of slot-limited portfolio",
    ),
    use_beta_model: bool = Query(default=False, description="Use beta model predictions if available"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get recommendations via GET request."""
    if not return_all:
        max_slots_limit = 20 if user_tier == "unlimited" else 8
        if slots > max_slots_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_slots_limit} slots allowed for {user_tier} tier",
            )

    try:
        exclude_ids: set[str] = set()
        if exclude:
            exclude_ids = {rid.strip() for rid in exclude.split(",") if rid.strip()}

        excluded_items: set[int] = set()
        if exclude_item_ids:
            excluded_items = {
                int(id.strip())
                for id in exclude_item_ids.split(",")
                if id.strip().isdigit()
            }

        normalized_user_id = normalize_user_id(user_id)

        if return_all:
            recommendations = engine.get_all_recommendations(
                style=style,
                capital=capital,
                risk=risk,
                exclude_item_ids=excluded_items,
                user_id=normalized_user_id,
                offset_pct=offset_pct,
                min_offset_pct=min_offset_pct,
                max_offset_pct=max_offset_pct,
                max_hour_offset=max_hour_offset,
                min_ev=min_ev,
                use_beta_model=use_beta_model,
            )

            generated_at = datetime.now(timezone.utc)
            valid_until = generated_at + timedelta(minutes=5)

            return AllRecommendationsResponse(
                recommendations=recommendations,
                generated_at=generated_at.isoformat(),
                valid_until=valid_until.isoformat(),
                total_count=len(recommendations),
            )

        recommendations = engine.get_recommendations(
            style=style,
            capital=capital,
            risk=risk,
            slots=slots,
            exclude_ids=exclude_ids,
            exclude_item_ids=excluded_items,
            user_id=normalized_user_id,
            offset_pct=offset_pct,
            min_offset_pct=min_offset_pct,
            max_offset_pct=max_offset_pct,
            max_hour_offset=max_hour_offset,
            min_ev=min_ev,
            use_beta_model=use_beta_model,
        )

        if include_metadata:
            metadata = engine.get_freshness_metadata()
            return RecommendationsWithMetadata(
                recommendations=recommendations,
                metadata=FreshnessMetadata(**metadata),
            )
        return recommendations
    except Exception as e:
        logger.error("Error getting recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/opportunities", response_model=OpportunitiesListResponse)
@limiter.limit(config.rate_limit_recommendations)
async def browse_opportunities(
    request: Request,
    filters: OpportunityFilters,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
) -> OpportunitiesListResponse:
    """Browse all available trading opportunities with filters."""
    start_time = time.monotonic()
    try:
        min_hour_offset = 1
        max_hour_offset = 48
        if filters.min_hours is not None:
            min_hour_offset = max(1, min(48, math.ceil(filters.min_hours)))
        if filters.max_hours is not None:
            max_hour_offset = max(1, min(48, math.floor(filters.max_hours)))
        if min_hour_offset > max_hour_offset:
            return OpportunitiesListResponse(items=[], total=0, has_more=False)

        all_opportunities = engine.get_all_opportunities(
            use_beta_model=filters.use_beta_model,
            min_hour_offset=min_hour_offset,
            max_hour_offset=max_hour_offset,
        )

        filtered: list[dict] = []
        for opp in all_opportunities:
            if filters.min_profit is not None and opp["expected_profit"] < filters.min_profit:
                continue
            if filters.max_profit is not None and opp["expected_profit"] > filters.max_profit:
                continue

            if filters.min_hours is not None and opp["expected_hours"] < filters.min_hours:
                continue
            if filters.max_hours is not None and opp["expected_hours"] > filters.max_hours:
                continue

            if filters.confidence and opp["confidence"] not in filters.confidence:
                continue

            if filters.max_capital is not None and opp["capital_required"] > filters.max_capital:
                continue

            if filters.categories and opp.get("category") not in filters.categories:
                continue

            filtered.append(opp)

        # Sort by conservative EV (bag-hold adjusted) when available; fall back to filled-profit.
        filtered.sort(
            key=lambda x: x.get("expected_profit_ev", x["expected_profit"]),
            reverse=True,
        )

        total = len(filtered)
        paginated = filtered[filters.offset : filters.offset + filters.limit]

        active_model_id: Optional[str] = None
        if filters.use_beta_model and config.beta_model_id:
            active_model_id = config.beta_model_id
        elif config.preferred_model_id:
            active_model_id = config.preferred_model_id

        items: list[OpportunityResponse] = []
        for opp in paginated:
            chips = engine.generate_why_chips(opp)
            raw_vol = opp.get("volume_24h")
            safe_vol = None
            if raw_vol is not None:
                if isinstance(raw_vol, float) and not math.isfinite(raw_vol):
                    safe_vol = None
                else:
                    safe_vol = int(raw_vol)

            items.append(
                OpportunityResponse(
                    id=f"opp-{opp['item_id']}",
                    item_id=opp["item_id"],
                    item_name=opp["item_name"],
                    icon_url=opp.get("icon_url"),
                    buy_price=opp["buy_price"],
                    sell_price=opp["sell_price"],
                    offset_pct=opp.get("offset_pct"),
                    quantity=opp["quantity"],
                    capital_required=opp["capital_required"],
                    expected_profit=opp["expected_profit"],
                    expected_profit_ev=opp.get("expected_profit_ev"),
                    expected_hours=opp["expected_hours"],
                    confidence=opp["confidence"],
                    fill_probability=opp["fill_probability"],
                    volume_24h=safe_vol,
                    trend=opp.get("trend"),
                    why_chips=[WhyChip(**c) for c in chips],
                    category=opp.get("category"),
                    model_id=active_model_id,
                )
            )

        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "opportunities_served",
            total_count=total,
            returned_count=len(items),
            offset=filters.offset,
            limit=filters.limit,
            duration_ms=round(duration_ms, 2),
        )

        return OpportunitiesListResponse(
            items=items,
            total=total,
            has_more=filters.offset + len(items) < total,
        )
    except Exception as e:
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "opportunities_error",
            error=str(e),
            duration_ms=round(duration_ms, 2),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/recommendations/item/{item_id}")
@limiter.limit(config.rate_limit_items)
async def get_recommendation_by_item_id(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    capital: Optional[int] = Query(default=None, ge=1000, description="Available capital in gp"),
    risk: Optional[Literal["low", "medium", "high"]] = Query(default=None, description="Risk tolerance"),
    style: Optional[Literal["passive", "hybrid", "active"]] = Query(default=None, description="Trading style"),
    slots: Optional[int] = Query(default=None, ge=1, le=8, description="Available GE slots"),
    include_price_history: bool = Query(default=False, description="Include 24h price history for charts"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get recommendation for a specific item by ID."""
    try:
        rec = engine.get_recommendation_by_item_id(
            item_id=item_id,
            capital=capital,
            risk=risk,
            style=style,
            slots=slots,
            include_price_history=include_price_history,
        )

        if rec is None:
            raise HTTPException(status_code=404, detail="No active recommendation for this item")

        return rec
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting recommendation for item", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/recommendations/item/{item_id}/refresh",
    response_model=ItemPriceRefreshResponse,
)
@limiter.limit(config.rate_limit_items)
async def refresh_item_prices(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Refresh buy/sell prices for an item without generating a new recommendation."""
    try:
        item_name = engine.loader.get_item_name(item_id)
        if item_name is None:
            raise HTTPException(status_code=404, detail="Item not found")

        price_data = engine.loader.get_latest_price(item_id)
        if price_data is None:
            raise HTTPException(status_code=404, detail="No price data available for this item")

        timestamp = price_data["timestamp"]
        if hasattr(timestamp, "isoformat"):
            updated_at = timestamp.isoformat().replace("+00:00", "") + "Z"
        else:
            updated_at = str(timestamp) + "Z"

        return ItemPriceRefreshResponse(
            itemId=item_id,
            itemName=item_name,
            buyPrice=int(price_data["high"]),
            sellPrice=int(price_data["low"]),
            updatedAt=updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error refreshing prices for item", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/recommendations/item")
@limiter.limit(config.rate_limit_items)
async def get_recommendation_by_item_name(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    name: str = Query(description="Item name to search for"),
    capital: int = Query(ge=1000, description="Available capital in gp"),
    risk: Literal["low", "medium", "high"] = Query(default="medium", description="Risk tolerance"),
    style: Literal["passive", "hybrid", "active"] = Query(default="hybrid", description="Trading style"),
    slots: int = Query(default=4, ge=1, le=8, description="Available GE slots"),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get recommendation for a specific item by name (with fuzzy matching)."""
    try:
        rec = engine.get_recommendation_by_item_name(
            item_name=name,
            capital=capital,
            risk=risk,
            style=style,
            slots=slots,
        )

        if rec is None:
            raise HTTPException(status_code=404, detail="Item not found")

        if "error" in rec:
            raise HTTPException(
                status_code=404,
                detail=rec["error"],
                headers={"X-Suggestions": str(rec.get("suggestions", []))},
            )

        return rec
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting recommendation by name", name=name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/v1/recommendations/{rec_id}",
    response_model=Optional[RecommendationResponse],
)
@limiter.limit(config.rate_limit_items)
async def get_recommendation_by_id(
    request: Request,
    rec_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get a specific recommendation by its ID."""
    rec = engine.get_recommendation_by_id(rec_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recommendation not found or expired")
    return rec
