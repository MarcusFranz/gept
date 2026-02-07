"""Prediction inspection endpoints."""


from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..api_dependencies import get_engine
from ..api_models import ItemPredictionResponse
from ..api_security import limiter, verify_api_key
from ..config import config
from ..recommendation_engine import RecommendationEngine

router = APIRouter()


@router.get("/api/v1/predictions/{item_id}", response_model=ItemPredictionResponse)
@limiter.limit(config.rate_limit_items)
async def get_item_predictions(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    engine: RecommendationEngine = Depends(get_engine),
):
    """Get full prediction details for a specific item."""
    prediction = engine.get_prediction_for_item(item_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="No predictions found for this item")
    return prediction

