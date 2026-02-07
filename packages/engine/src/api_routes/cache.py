"""Cache administration endpoints."""


from typing import Optional

from fastapi import APIRouter, Depends, Request

from ..api_models import CacheClearResponse
from ..api_security import limiter, verify_api_key
from ..cache import clear_all as cache_clear_all
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/api/v1/cache/clear", response_model=CacheClearResponse)
@limiter.limit("5/minute")
async def clear_cache(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Clear all cached recommendation data."""
    try:
        keys_cleared = await cache_clear_all()
        return CacheClearResponse(
            success=True,
            message=f"Cleared {keys_cleared} cache keys",
            keys_cleared=keys_cleared,
        )
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e))
        return CacheClearResponse(
            success=False,
            message=f"Failed to clear cache: {str(e)}",
            keys_cleared=0,
        )
