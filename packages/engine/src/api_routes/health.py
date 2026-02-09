"""Health and probe endpoints."""


from typing import Optional

import time

from fastapi import APIRouter, Depends, Request

from ..api_dependencies import get_state
from ..api_models import HealthResponse, LivenessResponse, ReadinessResponse, StartupResponse
from ..api_security import limiter, verify_api_key
from ..cache import get_cache_stats
import src.config as config_module
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _get_uptime_seconds(startup_time: float | None) -> int:
    if startup_time is None:
        return 0
    return int(time.time() - startup_time)


@router.get("/api/v1/health", response_model=HealthResponse)
@limiter.limit(config_module.config.rate_limit_health)
async def health_check(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Check system health status with full details."""
    state = get_state(request)

    cache_stats = await get_cache_stats()
    uptime = _get_uptime_seconds(state.startup_time)

    if state.engine is None:
        return HealthResponse(
            status="error",
            checks=[{"status": "error", "component": "engine", "error": "Not initialized"}],
            timestamp="",
            recommendation_store_size=0,
            crowding_stats={},
            cache_stats=cache_stats,
            uptime_seconds=uptime,
        )

    try:
        health = state.engine.health_check()
        health["cache_stats"] = cache_stats
        health["uptime_seconds"] = uptime
        return HealthResponse(**health)
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="error",
            checks=[{"status": "error", "component": "health_check", "error": str(e)}],
            timestamp="",
            recommendation_store_size=0,
            crowding_stats={},
            cache_stats=cache_stats,
            uptime_seconds=uptime,
        )


@router.get("/healthz", response_model=LivenessResponse)
async def health_liveness():
    """Lightweight liveness probe for load balancers."""
    return LivenessResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
@limiter.limit(config_module.config.rate_limit_health)
async def health_readiness(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Detailed readiness probe with dependency checks."""
    state = get_state(request)
    checks: dict[str, dict] = {}
    overall_status = "ok"

    # Database and predictions check
    if state.engine is None:
        checks["database"] = {"status": "error", "message": "Engine not initialized"}
        overall_status = "error"
    else:
        try:
            start_time = time.monotonic()
            health = state.engine.health_check()
            latency_ms = int((time.monotonic() - start_time) * 1000)

            checks["database"] = {"status": "ok", "latency_ms": latency_ms}

            # Check prediction freshness
            if health.get("checks"):
                db_check = health["checks"][0]
                pred_age = db_check.get("prediction_age_seconds", 0)
                if pred_age > config_module.config.prediction_stale_seconds:
                    checks["predictions"] = {
                        "status": "warning",
                        "message": f"Stale ({int(pred_age)}s old)",
                        "age_seconds": int(pred_age),
                    }
                    if overall_status == "ok":
                        overall_status = "degraded"
                else:
                    checks["predictions"] = {"status": "ok", "age_seconds": int(pred_age)}
            else:
                checks["predictions"] = {"status": "ok"}

        except Exception as e:
            checks["database"] = {"status": "error", "message": str(e)}
            overall_status = "error"

    # Redis cache check
    cache_stats = await get_cache_stats()
    if cache_stats.get("available"):
        checks["redis"] = {"status": "ok", "connected": True}
    elif config_module.config.redis_url:
        checks["redis"] = {"status": "warning", "message": "Unavailable", "connected": False}
        if overall_status == "ok":
            overall_status = "degraded"
    else:
        checks["redis"] = {
            "status": "ok",
            "message": "Not configured (in-memory)",
            "connected": False,
        }

    return ReadinessResponse(
        status=overall_status,
        checks=checks,
        uptime_seconds=_get_uptime_seconds(state.startup_time),
    )


@router.get("/startup", response_model=StartupResponse)
async def health_startup(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Startup probe for container orchestration."""
    from fastapi.responses import JSONResponse

    state = get_state(request)
    if not state.is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Application is starting up"},
        )

    return StartupResponse(status="ok", message="Application ready")
