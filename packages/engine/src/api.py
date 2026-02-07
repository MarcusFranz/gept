"""FastAPI server for the recommendation engine.

This module is the composition root: it wires together middleware, rate limiting,
lifespan startup/shutdown, and the route modules.
"""


from os import environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from .api_lifespan import lifespan
from .api_security import limiter
from .api_security import get_rate_limit_key, verify_api_key  # noqa: F401
from .api_state import RuntimeState
from .app_metadata import APP_VERSION
from .config import config
from .logging_config import RequestIDMiddleware, configure_logging
from .recommendation_engine import RecommendationEngine  # noqa: F401

# Configure structured logging before importing route modules.
configure_logging()

# Backwards-compatible module-level state.
#
# Some tests and integrations patch/read `src.api.outcome_db_engine` directly.
# When unset, handlers fall back to `app.state.runtime.*`.
_OUTCOME_DB_ENGINE_UNSET = object()
outcome_db_engine = _OUTCOME_DB_ENGINE_UNSET

app = FastAPI(
    title="GePT Recommendation Engine",
    description="OSRS Grand Exchange flipping recommendation API",
    version=APP_VERSION,
    lifespan=lifespan,
    # Default to secure-by-default: do not expose docs/OpenAPI unless explicitly enabled.
    # Enable temporarily (e.g. in local dev) with ENGINE_DOCS_ENABLED=true.
    docs_url="/docs" if environ.get("ENGINE_DOCS_ENABLED", "").lower() == "true" else None,
    redoc_url="/redoc" if environ.get("ENGINE_DOCS_ENABLED", "").lower() == "true" else None,
    openapi_url="/openapi.json" if environ.get("ENGINE_DOCS_ENABLED", "").lower() == "true" else None,
)

# Ensure runtime exists even when TestClient doesn't run the lifespan context.
app.state.runtime = RuntimeState()

# Rate limiting (SlowAPI)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - configured via CORS_ORIGINS and CORS_ORIGIN_REGEX env vars
# Default is localhost only for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_origin_regex=config.cors_origin_regex if config.cors_origin_regex else None,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request ID middleware for request tracking and logging correlation.
app.add_middleware(RequestIDMiddleware)

# Routers
from .api_routes.cache import router as cache_router  # noqa: E402
from .api_routes.feedback import router as feedback_router  # noqa: E402
from .api_routes.guidance import router as guidance_router  # noqa: E402
from .api_routes.health import router as health_router  # noqa: E402
from .api_routes.items import router as items_router  # noqa: E402
from .api_routes.legacy import router as legacy_router  # noqa: E402
from .api_routes.orders import router as orders_router  # noqa: E402
from .api_routes.outcomes import router as outcomes_router  # noqa: E402
from .api_routes.predictions import router as predictions_router  # noqa: E402
from .api_routes.recommendations import router as recommendations_router  # noqa: E402
from .api_routes.root import router as root_router  # noqa: E402
from .api_routes.trades import router as trades_router  # noqa: E402
from .api_routes.users import router as users_router  # noqa: E402
from .api_routes.webhooks import router as webhooks_router  # noqa: E402

app.include_router(health_router)
app.include_router(cache_router)
app.include_router(recommendations_router)
app.include_router(items_router)
app.include_router(predictions_router)
app.include_router(outcomes_router)
app.include_router(users_router)
app.include_router(feedback_router)
app.include_router(orders_router)
app.include_router(guidance_router)
app.include_router(trades_router)
app.include_router(legacy_router)
app.include_router(webhooks_router)
app.include_router(root_router)


def create_app() -> FastAPI:
    """Create and return the FastAPI app instance."""
    return app
