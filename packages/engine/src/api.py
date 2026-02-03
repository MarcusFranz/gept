"""FastAPI server for the recommendation engine."""

import asyncio
import hashlib
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Literal, Optional, Union

import httpx
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import and_, case, create_engine, func, select, text
from sqlalchemy.engine import Engine

from .cache import (
    clear_all as cache_clear_all,
    close_cache,
    get_cache_stats,
    get_cached,
    init_cache,
    set_cached,
)
from .config import config
from .logging_config import (
    RequestIDMiddleware,
    configure_logging,
    get_logger,
)
from .alert_dispatcher import AlertDispatcher
from .recommendation_engine import RecommendationEngine
from .trade_events import TradeEvent, TradeEventHandler, TradeEventType, TradePayload
from .trade_price_monitor import TradePriceMonitor
from .schema import recommendation_feedback, trade_outcomes
from .webhook import WebhookSignatureError, verify_webhook_signature

# Configure structured logging before creating logger
configure_logging()
logger = get_logger(__name__)

# Global engine instance
engine: Optional[RecommendationEngine] = None

# Separate connection pool for outcome database (gept_bot)
outcome_db_engine: Optional[Engine] = None

# Background cleanup task handle
_cleanup_task: Optional[asyncio.Task] = None

# Trade price monitor background task
_monitor_task: Optional[asyncio.Task] = None

# Alert dispatcher for price monitoring
alert_dispatcher: Optional[AlertDispatcher] = None

# Startup time for uptime calculation
_startup_time: Optional[float] = None

# Ready state flag (False until fully initialized)
_is_ready: bool = False

# Trade event handler for webhook integration
trade_event_handler: Optional[TradeEventHandler] = None

# Application version
APP_VERSION = "2.0.0"


def _is_valid_sha256(value: str) -> bool:
    """Check if a string is a valid SHA256 hash (64 hex characters).

    Args:
        value: String to validate

    Returns:
        True if value is exactly 64 lowercase hex characters
    """
    if len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


def _normalize_user_id(user_id: Optional[str]) -> Optional[str]:
    """Normalize user_id to ensure it's a valid SHA256 hash.

    If the provided user_id is already a valid SHA256 hash (64 hex chars),
    returns it as-is. Otherwise, hashes the value to create a valid SHA256.
    This ensures raw Discord IDs are never stored in the crowding tracker.

    Args:
        user_id: User identifier (may be raw Discord ID or pre-hashed)

    Returns:
        SHA256 hash of the user_id, or None if user_id is None/empty
    """
    if not user_id:
        return None

    user_id = user_id.strip()
    if not user_id:
        return None

    # If already a valid SHA256 hash, return as-is (lowercase)
    if _is_valid_sha256(user_id):
        return user_id.lower()

    # Otherwise, hash it server-side to protect privacy
    logger.warning(
        f"Received non-hashed user_id (length={len(user_id)}), hashing server-side"
    )
    return hashlib.sha256(user_id.encode()).hexdigest()


# Cleanup interval in seconds (run every 5 minutes)
CROWDING_CLEANUP_INTERVAL_SECONDS = 300


async def _crowding_cleanup_loop() -> None:
    """Background task that periodically cleans up the crowding tracker.

    Runs cleanup_all() on the crowding tracker every CROWDING_CLEANUP_INTERVAL_SECONDS
    to remove expired deliveries and enforce memory limits.
    """
    while True:
        try:
            await asyncio.sleep(CROWDING_CLEANUP_INTERVAL_SECONDS)
            if engine is not None:
                engine.crowding_tracker.cleanup_all()
                logger.debug("Crowding tracker cleanup completed")
        except asyncio.CancelledError:
            logger.info("Crowding cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in crowding cleanup task: {e}")
            # Continue running despite errors


async def _resync_active_trades() -> None:
    """Trigger a resync of active trades from the web app."""
    if not config.trade_webhooks_enabled:
        logger.info(
            "Active trade resync skipped: TRADE_WEBHOOKS_ENABLED=false",
        )
        return

    if not config.web_app_resync_url:
        logger.info("Active trade resync skipped: WEB_APP_RESYNC_URL not configured")
        return

    if not config.webhook_secret:
        logger.warning("Active trade resync skipped: WEBHOOK_SECRET not configured")
        return

    headers = {"Authorization": f"Bearer {config.webhook_secret}"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(config.web_app_resync_url, headers=headers)

        if 200 <= response.status_code < 300:
            dispatched = None
            try:
                payload = response.json()
                dispatched = payload.get("data", {}).get("dispatched")
            except Exception:
                dispatched = None
            logger.info(
                "Active trade resync completed",
                status_code=response.status_code,
                dispatched=dispatched,
            )
            return

        logger.warning(
            "Active trade resync failed",
            status_code=response.status_code,
        )
    except Exception as e:
        logger.warning("Active trade resync error", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global engine, outcome_db_engine, _cleanup_task, _startup_time, _is_ready, trade_event_handler

    alert_dispatcher = None
    _monitor_task = None

    # Startup
    _startup_time = time.time()
    logger.info("Initializing recommendation engine...")

    errors = config.validate()
    if errors:
        logger.error(f"Configuration errors: {errors}")
        raise RuntimeError(f"Configuration errors: {errors}")

    engine = RecommendationEngine(
        db_connection_string=config.db_connection_string,
        config=config,
    )

    # Test connection
    health = engine.health_check()
    if health["status"] == "error":
        logger.error(f"Engine health check failed: {health}")
        raise RuntimeError("Failed to connect to predictions database")

    pred_age = health["checks"][0].get("prediction_age_seconds", "unknown")
    logger.info(f"Engine initialized, predictions age: {pred_age}s")

    # Initialize trade event handler for webhook integration
    trade_event_handler = TradeEventHandler(
        crowding_tracker=engine.crowding_tracker,
        recommendation_engine=engine,
    )
    logger.info("Trade event handler initialized")

    if config.trade_webhooks_enabled:
        # Resync active trades after engine restart
        await _resync_active_trades()
    else:
        logger.warning(
            "Trade webhooks disabled",
            reason="TRADE_WEBHOOKS_ENABLED=false",
        )

    if config.price_drop_monitor_enabled:
        # Initialize alert dispatcher for price monitoring
        alert_dispatcher = AlertDispatcher()
        logger.info("Alert dispatcher initialized")

        # Start trade price monitor background task
        trade_monitor = TradePriceMonitor(
            trade_event_handler=trade_event_handler,
            prediction_loader=engine.loader,
            alert_dispatcher=alert_dispatcher,
            config=config,
        )
        _monitor_task = asyncio.create_task(trade_monitor.run())
        logger.info("Trade price monitor started")
    else:
        logger.warning(
            "Price drop monitor disabled",
            reason="PRICE_DROP_MONITOR_ENABLED=false",
        )

    # Initialize outcome database connection (optional - for ML feedback loop)
    if config.outcome_db_connection_string:
        try:
            outcome_db_engine = create_engine(
                config.outcome_db_connection_string,
                pool_size=config.outcome_db_pool_size,
                pool_pre_ping=True,
            )
            # Test connection
            with outcome_db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Outcome database connection established")
        except Exception as e:
            logger.warning(
                f"Outcome database connection failed: {e}. "
                "Trade outcome recording will be disabled."
            )
            outcome_db_engine = None
    else:
        logger.info(
            "Outcome database not configured (OUTCOME_DB_CONNECTION_STRING). "
            "Trade outcome recording disabled."
        )

    # Start background cleanup task for crowding tracker
    _cleanup_task = asyncio.create_task(_crowding_cleanup_loop())
    logger.info(
        f"Crowding cleanup task started (interval={CROWDING_CLEANUP_INTERVAL_SECONDS}s)"
    )

    # Initialize Redis cache (optional - for response caching)
    cache_available = await init_cache()
    if cache_available:
        logger.info("Redis response cache initialized")
    else:
        logger.info("Redis response cache not available, using direct database queries")

    # Mark as ready for startup probes
    _is_ready = True
    logger.info("Application startup complete, ready to serve requests")

    yield

    # Mark as not ready during shutdown
    _is_ready = False

    # Shutdown
    # Cancel the trade price monitor task
    if _monitor_task is not None:
        _monitor_task.cancel()
        try:
            await _monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("Trade price monitor stopped")

    if alert_dispatcher is not None:
        await alert_dispatcher.close()
        logger.info("Alert dispatcher closed")

    # Cancel the cleanup task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Crowding cleanup task stopped")

    # Close cache connection
    await close_cache()

    if outcome_db_engine:
        outcome_db_engine.dispose()
        logger.info("Outcome database connection closed")
    if engine:
        engine.close()
    logger.info("Recommendation engine shut down")


app = FastAPI(
    title="GePT Recommendation Engine",
    description="OSRS Grand Exchange flipping recommendation API",
    version="2.0.0",
    lifespan=lifespan,
)

# Rate limiting with user-based key function
# For web frontend traffic (Vercel), we rate limit by user hash or forwarded IP
# This prevents a single Vercel edge IP from being rate limited for all users


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on user identifier or IP.

    Priority:
    1. User hash from X-User-Hash header (authenticated web users)
    2. User hash from user_id query param
    3. X-Forwarded-For header (proxied requests)
    4. Client IP (direct requests)
    """
    # Try user hash from header (web frontend with auth)
    user_hash = request.headers.get("X-User-Hash")
    if user_hash and len(user_hash) == 64:
        return f"user:{user_hash}"

    # Try user_id from query params
    user_id = request.query_params.get("user_id")
    if user_id and len(user_id) == 64:
        return f"user:{user_id}"

    # Only trust X-Forwarded-For from known proxies (localhost, Docker network)
    # Prevents rate limit bypass via spoofed headers from direct connections
    client_host = request.client.host if request.client else None
    trusted_proxies = {"127.0.0.1", "::1", "172.17.0.1"}
    if client_host in trusted_proxies:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
            return f"ip:{client_ip}"

    # Use direct connection IP (not spoofable)
    if client_host:
        return f"ip:{client_host}"

    return "ip:unknown"


limiter = Limiter(key_func=get_rate_limit_key)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - configured via CORS_ORIGINS and CORS_ORIGIN_REGEX env vars
# Default is localhost only for security (server-to-server API)
# For production:
#   CORS_ORIGINS=https://gept.gg,https://www.gept.gg
#   CORS_ORIGIN_REGEX=https://gept-gg-.*\.vercel\.app
#
# This allows:
#   - Exact matches from CORS_ORIGINS list
#   - Pattern matches for Vercel preview deployments via CORS_ORIGIN_REGEX
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_origin_regex=config.cors_origin_regex if config.cors_origin_regex else None,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request ID middleware for request tracking and logging correlation
app.add_middleware(RequestIDMiddleware)

# API Key Authentication
# Security scheme for OpenAPI documentation
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    x_api_key: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """Verify API key for protected endpoints.

    INTERNAL_API_KEY is required at startup (enforced by config.validate()).
    The bypass below is unreachable in production but kept for test ergonomics.

    Returns:
        The API key if valid, None if auth is not configured (unreachable in prod)

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    # Unreachable in production — config.validate() blocks startup without key.
    # Kept for test fixtures that don't go through the lifespan startup path.
    if not config.internal_api_key:
        return None

    # API key is required
    if not x_api_key:
        logger.warning("API request rejected: missing X-API-Key header")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key
    if x_api_key != config.internal_api_key:
        logger.warning("API request rejected: invalid API key")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


class PriceHistoryPoint(BaseModel):
    """A single point in price history."""

    timestamp: str
    price: int


# Extended price history models (Issue #152)
class ExtendedPriceHistoryPoint(BaseModel):
    """Extended price history point with high/low data."""

    timestamp: str
    high: int
    low: int
    avgHigh: int
    avgLow: int


class PriceHistoryResponse(BaseModel):
    """Response for dedicated price history endpoint."""

    itemId: int
    itemName: str
    history: list[ExtendedPriceHistoryPoint]
    trend: Literal["Rising", "Stable", "Falling"]
    change24h: float  # Percentage change over 24h


class WhyChip(BaseModel):
    """Feature explanation chip for recommendations."""

    icon: str = Field(description="Emoji icon")
    label: str = Field(description="Short label")
    type: Literal["positive", "neutral", "negative"]


# Pydantic models for request/response
class RecommendationResponse(BaseModel):
    """Discord bot recommendation response."""

    id: str
    itemId: int
    item: str
    buyPrice: int
    sellPrice: int
    quantity: int
    capitalRequired: int
    expectedProfit: int
    confidence: Literal["high", "medium", "low"]
    fillProbability: float  # Model's fill probability (0-1)
    fillConfidence: Literal["Strong", "Good", "Fair"]  # Human-readable confidence
    trend: Literal["Rising", "Stable", "Falling"]
    expectedHours: int  # Expected hours until trade completes
    volume24h: Optional[int] = None  # Optional: actual 24h trade volume
    priceHistory: Optional[list[PriceHistoryPoint]] = (
        None  # Optional: for expanded view
    )
    # Model attribution fields (for tracking which model generated the prediction)
    modelId: Optional[int] = None  # ID of the model that generated this prediction
    modelStatus: Optional[str] = None  # Status of the model (ACTIVE, DEPRECATED, etc.)
    # Rationale fields (for "Why this item?" UX)
    reason: Optional[str] = None  # Human-readable explanation for the recommendation
    isRecommended: Optional[bool] = (
        None  # Whether item meets criteria (always True for portfolio recs)
    )
    whyChips: Optional[list[WhyChip]] = None  # Feature explanations for UI


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok", "warning", "error", "degraded"]
    checks: list[dict]
    timestamp: str
    recommendation_store_size: int
    crowding_stats: dict
    cache_stats: Optional[dict] = None  # Redis cache statistics (if available)
    version: str = APP_VERSION
    uptime_seconds: Optional[int] = None


class LivenessResponse(BaseModel):
    """Lightweight liveness check response."""

    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    """Detailed readiness check response with dependency status."""

    status: Literal["ok", "degraded", "error"]
    checks: dict[str, dict]
    version: str = APP_VERSION
    uptime_seconds: int


class StartupResponse(BaseModel):
    """Startup probe response."""

    status: Literal["ok", "starting"]
    message: str


class FreshnessMetadata(BaseModel):
    """Metadata about prediction freshness."""

    inference_at: Optional[str] = None  # ISO 8601 timestamp of latest predictions
    inference_age_seconds: Optional[float] = None  # Age of predictions in seconds
    stale: bool = False  # Whether predictions exceed staleness threshold
    stale_threshold_seconds: int = 300  # The threshold used for staleness check


class RecommendationsWithMetadata(BaseModel):
    """Recommendations response with freshness metadata."""

    recommendations: list[RecommendationResponse]
    metadata: FreshnessMetadata


class AllRecommendationsResponse(BaseModel):
    """Response for return_all mode with all viable flips sorted by score."""

    recommendations: list[RecommendationResponse]
    generated_at: str  # ISO 8601 timestamp
    valid_until: str  # ISO 8601 timestamp (generated_at + 5 min TTL)
    total_count: int


class ActiveTrade(BaseModel):
    """An active trade the user is tracking."""

    itemId: int
    quantity: int
    buyPrice: int


class RecommendationRequest(BaseModel):
    """Request for Discord bot recommendations."""

    style: Literal["passive", "hybrid", "active"] = "hybrid"
    capital: int = Field(ge=1000, description="Total capital in gp")
    risk: Literal["low", "medium", "high"] = "medium"
    slots: int = Field(ge=1, le=20, default=4, description="Total GE slots")
    user_tier: Literal["free", "premium", "unlimited"] = Field(
        default="free",
        description="User tier (free/premium/unlimited). Unlimited allows up to 20 slots.",
    )
    activeTrades: list[ActiveTrade] = Field(
        default=[], description="Currently tracked trades"
    )
    userId: Optional[str] = Field(
        default=None,
        description="User identifier for crowding tracking (SHA256 hash recommended)",
    )
    offset_pct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Specific offset percentage to filter by (0.0125 to 0.0250)",
    )
    min_offset_pct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Minimum offset percentage threshold (0.0125 to 0.0250)",
    )
    max_offset_pct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Maximum offset percentage threshold (0.0125 to 0.0250)",
    )
    include_metadata: bool = Field(
        default=False,
        description="Include freshness metadata in response",
    )
    return_all: bool = Field(
        default=False,
        description="If true, return all viable flips sorted by score instead of slot-limited portfolio",
    )


class ItemPredictionResponse(BaseModel):
    """Full prediction details for an item."""

    item_id: int
    item_name: str
    best_config: dict
    fill_probability: float
    expected_value: float
    buy_price: int
    sell_price: int
    confidence: str
    all_predictions: list[dict]
    # Model attribution fields
    model_id: Optional[int] = None  # ID of the model for this item
    model_status: Optional[str] = None  # Status (ACTIVE, DEPRECATED, etc.)
    mean_auc: Optional[float] = None  # Model quality metric


class ItemSearchResult(BaseModel):
    """Item search result for autocomplete."""

    itemId: int
    name: str
    category: Optional[str] = None


class ItemPriceRefreshResponse(BaseModel):
    """Response for refreshing stale item prices."""

    itemId: int
    itemName: str
    buyPrice: int
    sellPrice: int
    updatedAt: str  # ISO 8601 format


class ItemPriceLookupResponse(BaseModel):
    """Response for single-item price lookup.

    Returns price recommendations for any supported item, regardless of
    whether it appears in the top recommendations list.
    """

    itemId: int
    itemName: str
    side: str  # "buy" or "sell"
    recommendedPrice: int
    currentMarketPrice: int
    offsetPercent: float
    fillProbability: float
    expectedValue: float
    timeWindowHours: int
    volume24h: Optional[int] = None
    trend: Optional[str] = None
    isRecommended: bool  # Whether this item meets recommendation thresholds
    warning: Optional[str] = None  # Warning if low confidence/EV
    # Flip metrics (issue #129)
    marginGp: Optional[int] = None  # Profit per unit in gp (sell - buy - tax)
    marginPercent: Optional[float] = None  # Margin as percentage of buy price
    buyLimit: Optional[int] = None  # GE buy limit for this item
    priceHistory: Optional[list[dict]] = None  # 24h price history for charts


class TradeOutcomeRequest(BaseModel):
    """Request to report a trade outcome."""

    userId: str = Field(description="Hashed user ID (SHA256 of Discord ID)")
    itemId: int = Field(description="OSRS item ID")
    itemName: str = Field(description="Item name")
    recId: str = Field(description="Recommendation ID that led to this trade")
    buyPrice: int = Field(ge=1, description="Buy price in gp")
    sellPrice: int = Field(ge=1, description="Sell price in gp")
    quantity: int = Field(ge=1, description="Quantity traded")
    actualProfit: int = Field(
        description="Actual profit/loss after GE tax (can be negative)"
    )
    reportedAt: str = Field(description="ISO 8601 timestamp when user reported outcome")


class TradeOutcomeResponse(BaseModel):
    """Response after recording a trade outcome."""

    success: bool
    message: str


class FlipInfo(BaseModel):
    """Information about a single flip trade."""

    itemName: str = Field(description="Item name")
    profit: int = Field(description="Profit or loss in gp")


class PeriodComparison(BaseModel):
    """Comparison of stats to previous period."""

    profitDelta: int = Field(description="Profit change vs previous period")
    winRateDelta: float = Field(description="Win rate change vs previous period")


class UserStatsResponse(BaseModel):
    """Aggregated user trade statistics."""

    period: Literal["week", "month", "all"] = Field(description="Stats period")
    startDate: str = Field(description="Period start date (ISO 8601)")
    endDate: str = Field(description="Period end date (ISO 8601)")
    totalProfit: int = Field(description="Total profit/loss in gp")
    totalTrades: int = Field(description="Number of trades")
    winRate: float = Field(description="Fraction of profitable trades (0-1)")
    bestFlip: Optional[FlipInfo] = Field(
        default=None, description="Best performing trade"
    )
    worstFlip: Optional[FlipInfo] = Field(
        default=None, description="Worst performing trade"
    )
    comparisonToPreviousPeriod: Optional[PeriodComparison] = Field(
        default=None, description="Comparison to previous period (null for 'all')"
    )


# Feedback type enum matching database constraint
FeedbackType = Literal[
    "price_too_high",
    "price_too_low",
    "volume_too_low",
    "filled_quickly",
    "filled_slowly",
    "did_not_fill",
    "spread_too_wide",
    "price_manipulation",
    "other",
]

TradeSide = Literal["buy", "sell"]


class FeedbackRequest(BaseModel):
    """Request to submit recommendation feedback."""

    userId: str = Field(description="Hashed user ID (SHA256 of Discord ID)")
    itemId: int = Field(description="OSRS item ID")
    itemName: str = Field(description="Item name")
    feedbackType: FeedbackType = Field(description="Type of feedback")
    recId: Optional[str] = Field(
        default=None,
        description="Recommendation ID (optional - for linking to specific recommendation)",
    )
    side: Optional[TradeSide] = Field(
        default=None, description="Which side had the issue (buy/sell)"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional free-text notes (max 500 chars)",
    )
    recommendedPrice: Optional[int] = Field(
        default=None, ge=1, description="Price from recommendation"
    )
    actualPrice: Optional[int] = Field(
        default=None, ge=1, description="Actual price user saw/got"
    )
    submittedAt: str = Field(
        description="ISO 8601 timestamp when user submitted feedback"
    )


class FeedbackResponse(BaseModel):
    """Response after recording feedback."""

    success: bool
    message: str
    feedbackId: Optional[int] = None


class FeedbackTypeSummary(BaseModel):
    """Summary count for a single feedback type."""

    feedbackType: str
    count: int
    percentage: float


class FeedbackAnalyticsResponse(BaseModel):
    """Analytics summary for feedback."""

    period: Literal["week", "month", "all"]
    startDate: str
    endDate: str
    totalFeedback: int
    byType: list[FeedbackTypeSummary]
    topItems: list[dict]  # [{itemId, itemName, count}]


# Order update request/response models for active order evaluation
class OrderUpdateRequest(BaseModel):
    """Request to evaluate an active order."""

    item_id: int = Field(ge=1, description="OSRS item ID")
    order_type: Literal["buy", "sell"] = Field(description="Order type")
    user_price: int = Field(ge=1, description="User's order price in gp")
    quantity: int = Field(ge=1, description="Order quantity")
    time_elapsed_minutes: int = Field(ge=0, description="Minutes since order placed")
    user_id: Optional[str] = Field(default=None, description="Hashed user ID (SHA256)")


class AdjustPriceRecommendation(BaseModel):
    """Recommendation details for adjust_price action."""

    suggested_price: int = Field(description="Suggested new price")
    new_fill_probability: float = Field(
        ge=0, le=1, description="Fill prob at new price"
    )
    cost_difference: int = Field(description="Cost difference vs current order")


class WaitRecommendation(BaseModel):
    """Recommendation details for wait action."""

    estimated_fill_time_minutes: int = Field(
        ge=0, description="Estimated minutes until fill"
    )


class AlternativeItem(BaseModel):
    """Alternative item for abort_retry recommendation."""

    item_id: int
    item_name: str
    expected_profit: int
    fill_probability: float
    expected_hours: int


class AbortRetryRecommendation(BaseModel):
    """Recommendation details for abort_retry action."""

    alternative_items: list[AlternativeItem] = Field(
        description="Alternative items to try"
    )


class LiquidateRecommendation(BaseModel):
    """Recommendation details for liquidate action."""

    instant_price: int = Field(description="Price for instant execution")
    loss_amount: int = Field(description="Loss amount if liquidating")


class OrderRecommendations(BaseModel):
    """All recommendation options for an order."""

    adjust_price: Optional[AdjustPriceRecommendation] = None
    wait: Optional[WaitRecommendation] = None
    abort_retry: Optional[AbortRetryRecommendation] = None
    liquidate: Optional[LiquidateRecommendation] = None


class OrderUpdateResponse(BaseModel):
    """Response for order evaluation."""

    action: Literal["adjust_price", "wait", "abort_retry", "liquidate"] = Field(
        description="Recommended action"
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in recommendation")
    current_fill_probability: float = Field(
        ge=0, le=1, description="Fill probability at user's current price"
    )
    recommendations: OrderRecommendations = Field(
        description="Details for each possible action"
    )
    reasoning: str = Field(description="Human-readable explanation")


# Simplified guidance models for web UI check-ins
class GuidanceRequest(BaseModel):
    """Request for trade guidance."""

    item_id: int = Field(description="OSRS item ID")
    order_type: Literal["buy", "sell"] = Field(description="Current order type")
    user_price: int = Field(description="User's current order price")
    quantity: int = Field(description="Order quantity")
    time_elapsed_minutes: int = Field(description="Minutes since order placed")
    reported_progress: int = Field(ge=0, le=100, description="User-reported progress 0-100")


class GuidanceAction(str, Enum):
    """Simplified action for web UI."""

    hold = "hold"
    relist = "relist"
    exit = "exit"
    sell_now = "sell_now"


class GuidanceResponse(BaseModel):
    """Simplified guidance for web UI."""

    action: GuidanceAction
    reason: str = Field(description="Brief explanation")
    confidence: float = Field(ge=0, le=1)
    params: Optional[dict] = Field(default=None, description="Action-specific params")
    next_check_in_minutes: int = Field(description="Suggested next check-in")


# Trade updates polling models (Issue #148)
class NewItemRecommendation(BaseModel):
    """Alternative item recommendation for SWITCH_ITEM updates."""

    itemId: int
    item: str
    buyPrice: int
    sellPrice: int
    quantity: int
    expectedProfit: int


class TradeUpdate(BaseModel):
    """A single trade update recommendation."""

    id: str = Field(description="Unique update ID")
    tradeId: str = Field(description="Original trade/recommendation ID")
    type: Literal["ADJUST_PRICE", "SELL_NOW", "SWITCH_ITEM"] = Field(
        description="Update type"
    )
    reason: str = Field(description="Human-readable explanation")
    confidence: float = Field(ge=0, le=1, description="Confidence in recommendation")
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent the update is"
    )
    # Fields for ADJUST_PRICE type
    newSellPrice: Optional[int] = None
    originalSellPrice: Optional[int] = None
    profitDelta: Optional[int] = None
    # Fields for SWITCH_ITEM type
    newItem: Optional[NewItemRecommendation] = None
    # Fields for SELL_NOW type
    adjustedSellPrice: Optional[int] = None


class TradeUpdatesResponse(BaseModel):
    """Response for trade updates polling endpoint."""

    updates: list[TradeUpdate]
    nextCheckIn: int = Field(
        description="Suggested seconds until next poll (0 = no updates needed)"
    )


# Opportunity browsing models (for Opportunity Browser UI)
class OpportunityFilters(BaseModel):
    """Filters for browsing opportunities."""

    min_profit: Optional[int] = Field(
        default=None, description="Minimum expected profit in gp"
    )
    max_profit: Optional[int] = Field(
        default=None, description="Maximum expected profit in gp"
    )
    min_hours: Optional[float] = Field(
        default=None, description="Minimum expected hours"
    )
    max_hours: Optional[float] = Field(
        default=None, description="Maximum expected hours"
    )
    confidence: Optional[list[Literal["low", "medium", "high"]]] = Field(
        default=None, description="Confidence levels to include"
    )
    max_capital: Optional[int] = Field(
        default=None, description="Maximum capital required"
    )
    categories: Optional[list[str]] = Field(
        default=None, description="Item categories to include"
    )
    limit: int = Field(default=50, ge=1, le=200, description="Max results to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    use_beta_model: bool = Field(
        default=False, description="Use beta model predictions if available"
    )


class OpportunityResponse(BaseModel):
    """A browsable opportunity with full details."""

    id: str
    item_id: int
    item_name: str
    icon_url: Optional[str] = None
    buy_price: int
    sell_price: int
    quantity: int
    capital_required: int
    expected_profit: int
    expected_hours: float
    confidence: str
    fill_probability: float
    volume_24h: Optional[int] = None
    trend: Optional[str] = None
    why_chips: list[WhyChip]
    category: Optional[str] = None
    model_id: Optional[str] = None


class OpportunitiesListResponse(BaseModel):
    """Paginated list of opportunities."""

    items: list[OpportunityResponse]
    total: int
    has_more: bool


def _get_uptime_seconds() -> int:
    """Get uptime in seconds since startup."""
    if _startup_time is None:
        return 0
    return int(time.time() - _startup_time)


@app.get("/api/v1/health", response_model=HealthResponse)
@limiter.limit(config.rate_limit_health)
async def health_check(request: Request):
    """Check system health status with full details.

    Returns comprehensive health information including all dependency checks,
    cache stats, and crowding statistics. Use for monitoring dashboards.
    For lightweight probes, use /healthz (liveness) or /ready (readiness).
    """
    # Get cache stats (async)
    cache_stats = await get_cache_stats()
    uptime = _get_uptime_seconds()

    if engine is None:
        return HealthResponse(
            status="error",
            checks=[
                {"status": "error", "component": "engine", "error": "Not initialized"}
            ],
            timestamp="",
            recommendation_store_size=0,
            crowding_stats={},
            cache_stats=cache_stats,
            uptime_seconds=uptime,
        )

    try:
        health = engine.health_check()
        health["cache_stats"] = cache_stats
        health["uptime_seconds"] = uptime
        return HealthResponse(**health)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            checks=[{"status": "error", "component": "health_check", "error": str(e)}],
            timestamp="",
            recommendation_store_size=0,
            crowding_stats={},
            cache_stats=cache_stats,
            uptime_seconds=uptime,
        )


@app.get("/healthz", response_model=LivenessResponse)
async def health_liveness():
    """Lightweight liveness probe for load balancers.

    Returns immediately without any database or dependency checks.
    Use this for frequent health polling from load balancers (e.g., Cloudflare).
    Responds in <10ms.
    """
    return LivenessResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse)
@limiter.limit(config.rate_limit_health)
async def health_readiness(request: Request):
    """Detailed readiness probe with dependency checks.

    Checks all dependencies and returns structured status for each:
    - database: Connection and prediction data availability
    - predictions: Freshness of ML predictions
    - redis: Cache connection status (if configured)

    Returns 200 with status "ok", "degraded", or "error".
    Degraded means non-critical issues (stale predictions, cache unavailable).
    """
    checks: dict[str, dict] = {}
    overall_status = "ok"

    # Database and predictions check
    if engine is None:
        checks["database"] = {"status": "error", "message": "Engine not initialized"}
        overall_status = "error"
    else:
        try:
            start_time = time.monotonic()
            health = engine.health_check()
            latency_ms = int((time.monotonic() - start_time) * 1000)

            checks["database"] = {"status": "ok", "latency_ms": latency_ms}

            # Check prediction freshness
            if health.get("checks"):
                db_check = health["checks"][0]
                pred_age = db_check.get("prediction_age_seconds", 0)
                if pred_age > config.prediction_stale_seconds:
                    checks["predictions"] = {
                        "status": "warning",
                        "message": f"Stale ({int(pred_age)}s old)",
                        "age_seconds": int(pred_age),
                    }
                    if overall_status == "ok":
                        overall_status = "degraded"
                else:
                    checks["predictions"] = {
                        "status": "ok",
                        "age_seconds": int(pred_age),
                    }
            else:
                checks["predictions"] = {"status": "ok"}

        except Exception as e:
            checks["database"] = {"status": "error", "message": str(e)}
            overall_status = "error"

    # Redis cache check
    cache_stats = await get_cache_stats()
    if cache_stats.get("available"):
        checks["redis"] = {"status": "ok", "connected": True}
    elif config.redis_url:
        # Redis was configured but is unavailable
        checks["redis"] = {
            "status": "warning",
            "message": "Unavailable",
            "connected": False,
        }
        if overall_status == "ok":
            overall_status = "degraded"
    else:
        # Redis not configured (using in-memory cache)
        checks["redis"] = {
            "status": "ok",
            "message": "Not configured (in-memory)",
            "connected": False,
        }

    return ReadinessResponse(
        status=overall_status,
        checks=checks,
        uptime_seconds=_get_uptime_seconds(),
    )


@app.get("/startup", response_model=StartupResponse)
async def health_startup():
    """Startup probe for container orchestration.

    Returns 503 until the application is fully initialized and ready to serve.
    Returns 200 once startup is complete.

    Use this in Kubernetes/Docker startup probes.
    """
    if not _is_ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "starting", "message": "Application is starting up"},
        )

    return StartupResponse(status="ok", message="Application ready")


class CacheClearResponse(BaseModel):
    """Response for cache clear endpoint."""

    success: bool
    message: str
    keys_cleared: int


@app.post("/api/v1/cache/clear", response_model=CacheClearResponse)
@limiter.limit("5/minute")
async def clear_cache(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Clear all cached recommendation data.

    Admin endpoint to force cache invalidation after predictions refresh.
    Clears all recommendation and item cache keys.
    """
    try:
        keys_cleared = await cache_clear_all()
        return CacheClearResponse(
            success=True,
            message=f"Cleared {keys_cleared} cache keys",
            keys_cleared=keys_cleared,
        )
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return CacheClearResponse(
            success=False,
            message=f"Failed to clear cache: {str(e)}",
            keys_cleared=0,
        )


@app.post(
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
):
    """Get recommendations formatted for Discord bot.

    This is the main endpoint for the Discord bot to fetch flip recommendations.
    Takes user constraints (capital, style, risk, slots) and returns optimized trades.
    Accounts for active trades to calculate remaining capital and available slots.

    Set include_metadata=true to get freshness metadata with the response.
    Set return_all=true to get all viable flips sorted by score (no slot limit).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

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
        # Convert active trades to dict format
        active_trades = [
            {"itemId": t.itemId, "quantity": t.quantity, "buyPrice": t.buyPrice}
            for t in body.activeTrades
        ]

        # Normalize user_id to ensure it's a valid SHA256 hash
        normalized_user_id = _normalize_user_id(body.userId)

        if body.return_all:
            # Use new method that returns all viable flips sorted by score
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

            # Generate timestamps for response
            generated_at = datetime.now(timezone.utc)
            valid_until = generated_at + timedelta(minutes=5)

            return AllRecommendationsResponse(
                recommendations=recommendations,
                generated_at=generated_at.isoformat(),
                valid_until=valid_until.isoformat(),
                total_count=len(recommendations),
            )

        # Standard slot-limited portfolio optimization
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

        # Return with or without metadata based on request
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


@app.get(
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
    include_metadata: bool = Query(
        default=False,
        description="Include freshness metadata in response",
    ),
    return_all: bool = Query(
        default=False,
        description="If true, return all viable flips sorted by score instead of slot-limited portfolio",
    ),
    use_beta_model: bool = Query(
        default=False,
        description="Use beta model predictions if available",
    ),
):
    """Get recommendations via GET request.

    Alternative endpoint for Discord bot using query parameters.
    Supports exclude parameter to filter out already-seen recommendations.
    Supports exclude_item_ids to filter out specific items (e.g., active trades).
    Supports offset filtering:
    - offset_pct: exact margin target (1.25% to 2.5%)
    - min_offset_pct/max_offset_pct: margin range (can use one or both)
    Supports max_hour_offset to limit time horizon (overrides style default).
    Supports min_ev to set minimum expected value threshold.
    Set include_metadata=true to get freshness metadata with the response.
    Set return_all=true to get all viable flips sorted by score (no slot limit).
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Validate slots based on user tier (only applicable when not using return_all)
    if not return_all:
        max_slots_limit = 20 if user_tier == "unlimited" else 8
        if slots > max_slots_limit:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {max_slots_limit} slots allowed for {user_tier} tier",
            )

    try:
        # Parse exclude list (recommendation IDs)
        exclude_ids = set()
        if exclude:
            exclude_ids = {rid.strip() for rid in exclude.split(",") if rid.strip()}

        # Parse exclude_item_ids list (item IDs)
        excluded_items = set()
        if exclude_item_ids:
            excluded_items = {
                int(id.strip())
                for id in exclude_item_ids.split(",")
                if id.strip().isdigit()
            }

        # Normalize user_id to ensure it's a valid SHA256 hash
        normalized_user_id = _normalize_user_id(user_id)

        if return_all:
            # Use new method that returns all viable flips sorted by score
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

            # Generate timestamps for response
            generated_at = datetime.now(timezone.utc)
            valid_until = generated_at + timedelta(minutes=5)

            return AllRecommendationsResponse(
                recommendations=recommendations,
                generated_at=generated_at.isoformat(),
                valid_until=valid_until.isoformat(),
                total_count=len(recommendations),
            )

        # Standard slot-limited portfolio optimization
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

        # Return with or without metadata based on request
        if include_metadata:
            metadata = engine.get_freshness_metadata()
            return RecommendationsWithMetadata(
                recommendations=recommendations,
                metadata=FreshnessMetadata(**metadata),
            )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/opportunities", response_model=OpportunitiesListResponse)
@limiter.limit(config.rate_limit_recommendations)
async def browse_opportunities(
    request: Request,
    filters: OpportunityFilters,
    _api_key: Optional[str] = Depends(verify_api_key),
) -> OpportunitiesListResponse:
    """Browse all available trading opportunities with filters.

    Unlike /recommendations, this endpoint:
    - Returns more items (up to 200)
    - Supports rich filtering
    - Doesn't require user settings context
    - Includes why_chips for each item
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    start_time = time.monotonic()
    try:
        # Get all current predictions above minimum thresholds
        all_opportunities = engine.get_all_opportunities(
            use_beta_model=filters.use_beta_model
        )

        # Apply filters
        filtered = []
        for opp in all_opportunities:
            # Profit filter (use explicit None checks to allow 0 as valid filter value)
            if filters.min_profit is not None and opp["expected_profit"] < filters.min_profit:
                continue
            if filters.max_profit is not None and opp["expected_profit"] > filters.max_profit:
                continue

            # Time filter
            if filters.min_hours is not None and opp["expected_hours"] < filters.min_hours:
                continue
            if filters.max_hours is not None and opp["expected_hours"] > filters.max_hours:
                continue

            # Confidence filter
            if filters.confidence and opp["confidence"] not in filters.confidence:
                continue

            # Capital filter
            if filters.max_capital is not None and opp["capital_required"] > filters.max_capital:
                continue

            # Category filter
            if filters.categories and opp.get("category") not in filters.categories:
                continue

            filtered.append(opp)

        # Sort by expected profit (can add sort param later)
        filtered.sort(key=lambda x: x["expected_profit"], reverse=True)

        total = len(filtered)

        # Paginate
        paginated = filtered[filters.offset : filters.offset + filters.limit]

        # Determine model_id for attribution
        active_model_id: Optional[str] = None
        if filters.use_beta_model and config.beta_model_id:
            active_model_id = config.beta_model_id
        elif config.preferred_model_id:
            active_model_id = config.preferred_model_id

        # Build response with why_chips
        items = []
        for opp in paginated:
            chips = engine.generate_why_chips(opp)
            # Defence in depth: coerce volume_24h to int|None so Pydantic
            # never receives a float NaN for an Optional[int] field.
            raw_vol = opp.get("volume_24h")
            safe_vol = None if (raw_vol is None or (isinstance(raw_vol, float) and not math.isfinite(raw_vol))) else int(raw_vol)
            items.append(
                OpportunityResponse(
                    id=f"opp-{opp['item_id']}",
                    item_id=opp["item_id"],
                    item_name=opp["item_name"],
                    icon_url=opp.get("icon_url"),
                    buy_price=opp["buy_price"],
                    sell_price=opp["sell_price"],
                    quantity=opp["quantity"],
                    capital_required=opp["capital_required"],
                    expected_profit=opp["expected_profit"],
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


@app.get("/api/v1/recommendations/item/{item_id}")
@limiter.limit(config.rate_limit_items)
async def get_recommendation_by_item_id(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    capital: Optional[int] = Query(
        default=None, ge=1000, description="Available capital in gp"
    ),
    risk: Optional[Literal["low", "medium", "high"]] = Query(
        default=None, description="Risk tolerance"
    ),
    style: Optional[Literal["passive", "hybrid", "active"]] = Query(
        default=None, description="Trading style"
    ),
    slots: Optional[int] = Query(
        default=None, ge=1, le=8, description="Available GE slots"
    ),
    include_price_history: bool = Query(
        default=False, description="Include 24h price history for charts"
    ),
):
    """Get recommendation for a specific item by ID.

    If user context (capital, risk, style) is provided, generates a fresh recommendation
    tailored to user preferences. Otherwise returns cached recommendation.

    Returns full recommendation if item is good to flip, or explanation if not
    recommended. Use include_price_history=true for expanded view with mini chart data.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

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
            raise HTTPException(
                status_code=404, detail="No active recommendation for this item"
            )

        # If it's a "not recommended" response, return 200 with explanation
        return rec

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/recommendations/item/{item_id}/refresh",
    response_model=ItemPriceRefreshResponse,
)
@limiter.limit(config.rate_limit_items)
async def refresh_item_prices(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Refresh buy/sell prices for an item without generating a new recommendation.

    Used when a recommendation is stale (>15 min old) but user wants to check
    current market prices before placing an order. Does not consume rate limit.

    Returns current market prices (not model offset prices):
    - buyPrice: Current instant-buy price (low tick)
    - sellPrice: Current instant-sell price (high tick)
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # Get item name
        item_name = engine.loader.get_item_name(item_id)
        if item_name is None:
            raise HTTPException(status_code=404, detail="Item not found")

        # Get latest price data
        price_data = engine.loader.get_latest_price(item_id)
        if price_data is None:
            raise HTTPException(
                status_code=404, detail="No price data available for this item"
            )

        # Format timestamp as ISO 8601 with Z suffix
        timestamp = price_data["timestamp"]
        if hasattr(timestamp, "isoformat"):
            updated_at = timestamp.isoformat().replace("+00:00", "") + "Z"
        else:
            updated_at = str(timestamp) + "Z"

        return ItemPriceRefreshResponse(
            itemId=item_id,
            itemName=item_name,
            buyPrice=int(price_data["low"]),
            sellPrice=int(price_data["high"]),
            updatedAt=updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing prices for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/items/{item_id}/price",
    response_model=ItemPriceLookupResponse,
)
@limiter.limit(config.rate_limit_items)
async def get_item_price_lookup(
    request: Request,
    item_id: int = Path(..., gt=0, description="OSRS item ID"),
    _api_key: Optional[str] = Depends(verify_api_key),
    side: Literal["buy", "sell"] = Query(
        default="buy", description="Trade side - buy or sell"
    ),
    window: int = Query(
        default=24, ge=1, le=48, description="Target time window in hours"
    ),
    offset: Optional[float] = Query(
        default=None, ge=0.01, le=0.03, description="Target offset percentage"
    ),
    include_price_history: bool = Query(
        default=False, description="Include 24h price history for charts"
    ),
):
    """Get price recommendation for any item, regardless of recommendation status.

    Unlike the recommendation endpoints, this returns price data for ANY supported
    item, even if it doesn't meet the recommendation thresholds. This allows users
    to look up prices for items they want to trade regardless of expected value.

    Use cases:
    - Look up prices for items not in top recommendations
    - Check prices for items acquired through gameplay (drops, rewards)
    - Understand why an item isn't recommended
    - Research flip potential with margin metrics

    Returns:
        Price recommendation with fill probability, flip metrics, and optional price history
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

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
        logger.error(f"Error getting price lookup for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/recommendations/item")
@limiter.limit(config.rate_limit_items)
async def get_recommendation_by_item_name(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    name: str = Query(description="Item name to search for"),
    capital: int = Query(ge=1000, description="Available capital in gp"),
    risk: Literal["low", "medium", "high"] = Query(
        default="medium", description="Risk tolerance"
    ),
    style: Literal["passive", "hybrid", "active"] = Query(
        default="hybrid", description="Trading style"
    ),
    slots: int = Query(default=4, ge=1, le=8, description="Available GE slots"),
):
    """Get recommendation for a specific item by name (with fuzzy matching).

    Searches for items matching the name and returns a recommendation for the best
    match. Returns full recommendation if item is good to flip, or explanation if
    not recommended. If item not found, returns suggestions.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

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

        # Check if it's an error response (item not found)
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
        logger.error(f"Error getting recommendation by name '{name}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/recommendations/{rec_id}", response_model=Optional[RecommendationResponse]
)
@limiter.limit(config.rate_limit_items)
async def get_recommendation_by_id(
    request: Request,
    rec_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Get a specific recommendation by its ID.

    Used when user clicks "Mark Ordered" in Discord bot.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    rec = engine.get_recommendation_by_id(rec_id)
    if rec is None:
        raise HTTPException(
            status_code=404, detail="Recommendation not found or expired"
        )

    return rec


@app.get("/api/v1/predictions/{item_id}", response_model=ItemPredictionResponse)
@limiter.limit(config.rate_limit_items)
async def get_item_predictions(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Get full prediction details for a specific item.

    Returns all hour_offset/offset_pct combinations for the item.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    prediction = engine.get_prediction_for_item(item_id)
    if prediction is None:
        raise HTTPException(
            status_code=404, detail="No predictions found for this item"
        )

    return prediction


@app.get("/api/v1/items/search", response_model=list[ItemSearchResult])
@limiter.limit(config.rate_limit_search)
async def search_items(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=25, description="Maximum number of results"),
    no_cache: bool = Query(default=False, description="Bypass cache"),
):
    """Search for items by name with fuzzy matching.

    Supports case-insensitive search with typo tolerance for Discord autocomplete.
    Returns top N matches sorted by relevance score.

    Args:
        q: Search query string (minimum 1 character)
        limit: Maximum number of results to return (1-25, default 10)
        no_cache: If true, bypass the cache and fetch fresh data

    Returns:
        List of matching items with itemId and name
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    start_time = time.monotonic()
    # Generate cache key
    cache_key = f"search:{q.lower().strip()}:{limit}"

    # Try to get from cache (unless no_cache is set)
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
        # Use engine's built-in fuzzy search
        matches = engine.search_items_by_name(q, limit)

        # Convert to response format
        results = [
            ItemSearchResult(
                itemId=match["item_id"],
                name=match["item_name"],
                category=None,
            )
            for match in matches
        ]

        # Cache the results (serialize to dict for JSON)
        await set_cached(
            cache_key,
            [r.model_dump() for r in results],
            config.cache_ttl_search,
        )

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


@app.get("/api/v1/items/{item_id}/price-history", response_model=PriceHistoryResponse)
@limiter.limit(config.rate_limit_items)
async def get_item_price_history(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    hours: int = Query(
        default=24, ge=1, le=168, description="Hours of history (max 7 days)"
    ),
    no_cache: bool = Query(default=False, description="Bypass cache"),
):
    """Get price history for an item with trend analysis.

    Returns hourly price data for sparkline charts in the web frontend.
    Includes high/low prices and 24h change percentage.

    Args:
        item_id: OSRS item ID
        hours: Number of hours of history (default 24, max 168 = 7 days)
        no_cache: If true, bypass the cache and fetch fresh data
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Generate cache key
    cache_key = f"price_history:{item_id}:{hours}"

    # Try to get from cache (unless no_cache is set)
    if not no_cache:
        cached = await get_cached(cache_key)
        if cached is not None:
            return PriceHistoryResponse(**cached)

    try:
        # Get extended price history from database
        history = engine.loader.get_extended_price_history(item_id, hours=hours)

        if not history:
            # Try to get item name even if no price history
            item_name = engine.loader.get_item_name(item_id)
            if item_name is None:
                raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
            # Return empty history with neutral values
            return PriceHistoryResponse(
                itemId=item_id,
                itemName=item_name,
                history=[],
                trend="Stable",
                change24h=0.0,
            )

        # Get item name
        item_name = engine.loader.get_item_name(item_id)
        if item_name is None:
            item_name = f"Item {item_id}"

        # Get trend
        trend = engine.loader.get_item_trend(item_id)

        # Calculate 24h change percentage
        change_24h = 0.0
        if len(history) >= 2:
            first_point = history[0]
            last_point = history[-1]
            first_avg = (first_point["high"] + first_point["low"]) / 2
            last_avg = (last_point["high"] + last_point["low"]) / 2
            if first_avg > 0:
                change_24h = round(((last_avg - first_avg) / first_avg) * 100, 2)

        # Convert to response model
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

        # Cache the results
        await set_cached(
            cache_key,
            response.model_dump(),
            config.cache_ttl_prices,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting price history for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/recommendations/{rec_id}/outcome", response_model=TradeOutcomeResponse
)
@limiter.limit(config.rate_limit_outcomes)
async def report_trade_outcome(
    request: Request,
    rec_id: str,
    body: TradeOutcomeRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Record a trade outcome for ML feedback loop.

    Users report trade outcomes via Discord bot's /report command or when marking
    trades as "Filled" in active trades. This data feeds back into the prediction
    model to improve accuracy over time.

    Privacy:
    - User IDs must be hashed (SHA256) before sending
    - No Discord IDs or PII stored
    - Data used only for model training

    Args:
        rec_id: Recommendation ID that led to this trade
        body: Trade outcome details
    """
    # Check outcome database availability (separate from predictions database)
    if outcome_db_engine is None:
        raise HTTPException(status_code=503, detail="Outcome database not available")

    # Validate rec_id matches request body
    if rec_id != body.recId:
        raise HTTPException(
            status_code=400, detail="rec_id in URL does not match recId in request body"
        )

    # Validate userId is hashed (should be 64 char hex string for SHA256)
    if len(body.userId) != 64 or not all(
        c in "0123456789abcdef" for c in body.userId.lower()
    ):
        raise HTTPException(
            status_code=400, detail="userId must be SHA256 hash (64 hex characters)"
        )

    # Parse timestamp
    try:
        reported_at = datetime.fromisoformat(body.reportedAt.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid reportedAt timestamp: {e}"
        )

    # Insert into outcome database (gept_bot, separate from predictions osrs_data)
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
            f"Recorded trade outcome: rec_id={rec_id}, item={body.itemName}, "
            f"profit={body.actualProfit:+,}gp"
        )

        return TradeOutcomeResponse(success=True, message="Outcome recorded")

    except Exception as e:
        logger.error(f"Error recording trade outcome: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record trade outcome: {str(e)}"
        )


@app.get("/api/v1/users/{hashed_user_id}/stats", response_model=UserStatsResponse)
@limiter.limit(config.rate_limit_outcomes)
async def get_user_stats(
    request: Request,
    hashed_user_id: str,
    _api_key: Optional[str] = Depends(verify_api_key),
    period: Literal["week", "month", "all"] = Query(
        default="week",
        description="Stats period: week (7 days), month (30 days), or all time",
    ),
):
    """Get aggregated trade statistics for a user.

    Returns profit/loss, win rate, and best/worst flips for the specified period.

    Privacy:
    - User IDs must be hashed (SHA256) before sending
    - No Discord IDs or PII stored

    Args:
        hashed_user_id: SHA256 hash of the user's Discord ID (64 hex chars)
        period: Time period for stats (week, month, or all)
    """
    # Check outcome database availability
    if outcome_db_engine is None:
        raise HTTPException(status_code=503, detail="Outcome database not available")

    # Validate hash format (64 hex chars for SHA256)
    if len(hashed_user_id) != 64 or not all(
        c in "0123456789abcdef" for c in hashed_user_id.lower()
    ):
        raise HTTPException(
            status_code=400,
            detail="hashed_user_id must be SHA256 hash (64 hex characters)",
        )

    # Calculate date ranges based on period
    now = datetime.now(timezone.utc)
    if period == "week":
        start_date = now - timedelta(days=7)
        prev_start = start_date - timedelta(days=7)
        prev_end = start_date
    elif period == "month":
        start_date = now - timedelta(days=30)
        prev_start = start_date - timedelta(days=30)
        prev_end = start_date
    else:  # all
        start_date = None
        prev_start = None
        prev_end = None

    t = trade_outcomes
    try:
        with outcome_db_engine.connect() as conn:
            # Helper: build stats aggregation with optional date filter
            def _stats_query(uid, date_start=None, date_end=None):
                conditions = [t.c.user_id_hash == uid]
                if date_start:
                    conditions.append(t.c.reported_at >= date_start)
                if date_end:
                    conditions.append(t.c.reported_at < date_end)
                return select(
                    func.count().label("total_trades"),
                    func.coalesce(func.sum(t.c.actual_profit), 0).label("total_profit"),
                    func.coalesce(
                        func.sum(case((t.c.actual_profit > 0, 1), else_=0)), 0
                    ).label("winning_trades"),
                ).where(and_(*conditions))

            # Query current period stats
            result = conn.execute(
                _stats_query(hashed_user_id, start_date, now if start_date else None)
            )
            row = result.fetchone()
            total_trades = row[0] if row else 0
            total_profit = row[1] if row else 0
            winning_trades = row[2] if row else 0
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

            # Helper: build flip query (best or worst)
            def _flip_query(uid, order_dir, date_start=None, date_end=None):
                conditions = [t.c.user_id_hash == uid]
                if date_start:
                    conditions.append(t.c.reported_at >= date_start)
                if date_end:
                    conditions.append(t.c.reported_at < date_end)
                order_col = (
                    t.c.actual_profit.desc()
                    if order_dir == "best"
                    else t.c.actual_profit.asc()
                )
                return (
                    select(t.c.item_name, t.c.actual_profit)
                    .where(and_(*conditions))
                    .order_by(order_col)
                    .limit(1)
                )

            # Query best flip
            best_flip = None
            if total_trades > 0:
                best_row = conn.execute(
                    _flip_query(
                        hashed_user_id, "best", start_date, now if start_date else None
                    )
                ).fetchone()
                if best_row:
                    best_flip = FlipInfo(itemName=best_row[0], profit=best_row[1])

            # Query worst flip
            worst_flip = None
            if total_trades > 0:
                worst_row = conn.execute(
                    _flip_query(
                        hashed_user_id, "worst", start_date, now if start_date else None
                    )
                ).fetchone()
                if worst_row:
                    worst_flip = FlipInfo(itemName=worst_row[0], profit=worst_row[1])

            # Query previous period for comparison (only for week/month)
            comparison = None
            if prev_start and prev_end:
                prev_result = conn.execute(
                    _stats_query(hashed_user_id, prev_start, prev_end)
                )
                prev_row = prev_result.fetchone()
                prev_total_trades = prev_row[0] if prev_row else 0
                prev_total_profit = prev_row[1] if prev_row else 0
                prev_winning_trades = prev_row[2] if prev_row else 0
                prev_win_rate = (
                    prev_winning_trades / prev_total_trades
                    if prev_total_trades > 0
                    else 0.0
                )

                comparison = PeriodComparison(
                    profitDelta=int(total_profit) - int(prev_total_profit),
                    winRateDelta=round(win_rate - prev_win_rate, 4),
                )

        # Format dates
        if start_date:
            start_date_str = start_date.strftime("%Y-%m-%d")
        else:
            start_date_str = "all-time"
        end_date_str = now.strftime("%Y-%m-%d")

        logger.info(
            f"User stats: user={hashed_user_id[:8]}..., period={period}, "
            f"trades={total_trades}, profit={total_profit:+,}gp"
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
        logger.error(f"Error fetching user stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch user stats: {str(e)}"
        )


@app.post("/api/v1/feedback", response_model=FeedbackResponse)
@limiter.limit(config.rate_limit_outcomes)
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Submit structured feedback on a recommendation.

    Users can report issues like price being too high/low, volume issues,
    or fill time problems. This data is used for model improvement.

    Privacy:
    - User IDs must be hashed (SHA256) before sending
    - No Discord IDs or PII stored
    - Data used only for model training and analytics

    Args:
        body: Feedback details including type, item, and optional notes
    """
    import re

    # Check outcome database availability
    if outcome_db_engine is None:
        raise HTTPException(status_code=503, detail="Feedback database not available")

    # Validate userId is hashed (64 char hex string for SHA256)
    if len(body.userId) != 64 or not all(
        c in "0123456789abcdef" for c in body.userId.lower()
    ):
        raise HTTPException(
            status_code=400, detail="userId must be SHA256 hash (64 hex characters)"
        )

    # Parse timestamp
    try:
        submitted_at = datetime.fromisoformat(body.submittedAt.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid submittedAt timestamp: {e}"
        )

    # Validate rec_id format if provided (rec_{item_id}_{YYYYMMDDHH})
    if body.recId:
        if not re.match(r"^rec_\d+_\d{10}$", body.recId):
            raise HTTPException(
                status_code=400,
                detail="Invalid recId format. Expected: rec_{item_id}_{YYYYMMDDHH}",
            )

    # Insert into feedback table
    f = recommendation_feedback
    stmt = (
        f.insert()
        .values(
            user_id_hash=body.userId,
            rec_id=body.recId,
            item_id=body.itemId,
            item_name=body.itemName,
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
            f"Recorded feedback: item={body.itemName}, type={body.feedbackType}, "
            f"user={body.userId[:8]}..."
        )

        return FeedbackResponse(
            success=True, message="Feedback recorded", feedbackId=feedback_id
        )

    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to record feedback: {str(e)}"
        )


@app.get("/api/v1/feedback/analytics", response_model=FeedbackAnalyticsResponse)
@limiter.limit(config.rate_limit_outcomes)
async def get_feedback_analytics(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    period: Literal["week", "month", "all"] = Query(
        default="week",
        description="Analytics period: week (7 days), month (30 days), or all time",
    ),
    item_id: Optional[int] = Query(
        default=None,
        description="Filter by specific item ID",
    ),
):
    """Get aggregated feedback analytics.

    Returns summary statistics on feedback by type, top items receiving feedback,
    and trends over time. Useful for identifying systematic issues.

    Args:
        period: Time period for analytics (week, month, or all)
        item_id: Optional filter for specific item
    """
    if outcome_db_engine is None:
        raise HTTPException(status_code=503, detail="Feedback database not available")

    # Calculate date ranges
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

            # Build conditions for type breakdown
            type_conditions = []
            if start_date:
                type_conditions.append(f.c.submitted_at >= start_date)
            if item_id is not None:
                type_conditions.append(f.c.item_id == item_id)

            type_query = (
                select(
                    f.c.feedback_type,
                    func.count().label("count"),
                )
                .group_by(f.c.feedback_type)
                .order_by(func.count().desc())
            )
            if type_conditions:
                type_query = type_query.where(and_(*type_conditions))

            type_rows = conn.execute(type_query).fetchall()

            total_feedback = sum(row[1] for row in type_rows)
            by_type = [
                FeedbackTypeSummary(
                    feedbackType=row[0],
                    count=row[1],
                    percentage=(
                        round(row[1] / total_feedback * 100, 1)
                        if total_feedback > 0
                        else 0
                    ),
                )
                for row in type_rows
            ]

            # Get top items (date filter only, no item filter)
            top_items_query = (
                select(
                    f.c.item_id,
                    f.c.item_name,
                    func.count().label("count"),
                )
                .group_by(f.c.item_id, f.c.item_name)
                .order_by(func.count().desc())
                .limit(10)
            )
            if start_date:
                top_items_query = top_items_query.where(
                    f.c.submitted_at >= start_date
                )

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
        logger.error(f"Error fetching feedback analytics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch analytics: {str(e)}"
        )


@app.post("/api/v1/recommendations/update", response_model=OrderUpdateResponse)
@limiter.limit(config.rate_limit_outcomes)
async def evaluate_active_order(
    request: Request,
    body: OrderUpdateRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Evaluate an active/pending order and recommend action.

    For users with unfilled buy or sell orders, this endpoint analyzes the order's
    current fill probability and market conditions to recommend one of four actions:

    - **wait**: Keep the order as-is, fill is likely soon
    - **adjust_price**: Modify price to improve fill probability
    - **abort_retry**: Cancel and try a different item with better opportunity
    - **liquidate**: Execute immediately at market price (accept loss)

    All four recommendation options are always returned so the Discord bot can
    present them as choices to the user.

    Privacy:
    - User IDs should be hashed (SHA256) before sending
    - No Discord IDs or PII stored
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Validate user_id format if provided (64 char hex for SHA256)
    if body.user_id and (
        len(body.user_id) != 64
        or not all(c in "0123456789abcdef" for c in body.user_id.lower())
    ):
        raise HTTPException(
            status_code=400, detail="user_id must be SHA256 hash (64 hex characters)"
        )

    try:
        result = engine.evaluate_active_order(
            item_id=body.item_id,
            order_type=body.order_type,
            user_price=body.user_price,
            quantity=body.quantity,
            time_elapsed_minutes=body.time_elapsed_minutes,
            user_id=body.user_id,
        )

        # Convert nested dicts to Pydantic models
        recommendations = OrderRecommendations(
            adjust_price=(
                AdjustPriceRecommendation(**result["recommendations"]["adjust_price"])
                if result["recommendations"].get("adjust_price")
                else None
            ),
            wait=(
                WaitRecommendation(**result["recommendations"]["wait"])
                if result["recommendations"].get("wait")
                else None
            ),
            abort_retry=(
                AbortRetryRecommendation(**result["recommendations"]["abort_retry"])
                if result["recommendations"].get("abort_retry")
                else None
            ),
            liquidate=(
                LiquidateRecommendation(**result["recommendations"]["liquidate"])
                if result["recommendations"].get("liquidate")
                else None
            ),
        )

        logger.info(
            f"Order evaluation: item={body.item_id}, type={body.order_type}, "
            f"action={result['action']}, confidence={result['confidence']}"
        )

        return OrderUpdateResponse(
            action=result["action"],
            confidence=result["confidence"],
            current_fill_probability=result["current_fill_probability"],
            recommendations=recommendations,
            reasoning=result["reasoning"],
        )

    except Exception as e:
        logger.error(f"Error evaluating order: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to evaluate order: {str(e)}"
        )


@app.post("/api/v1/guidance", response_model=GuidanceResponse)
@limiter.limit(config.rate_limit_outcomes)
async def get_guidance(
    request: Request,
    body: GuidanceRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
) -> GuidanceResponse:
    """Get prescriptive guidance for an active trade.

    Used by the web UI during check-ins to provide simple,
    actionable recommendations.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")

    try:
        # Use existing OrderAdvisor via engine.evaluate_active_order
        result = engine.evaluate_active_order(
            item_id=body.item_id,
            order_type=body.order_type,
            user_price=body.user_price,
            quantity=body.quantity,
            time_elapsed_minutes=body.time_elapsed_minutes,
        )

        # Map OrderAdvisor action to simplified GuidanceAction
        action_map = {
            "wait": GuidanceAction.hold,
            "adjust_price": GuidanceAction.relist,
            "abort_retry": GuidanceAction.exit,
            "liquidate": GuidanceAction.sell_now,
        }

        action = action_map.get(result["action"], GuidanceAction.hold)

        # Build params based on action
        params = None
        if action == GuidanceAction.relist and result["recommendations"].get(
            "adjust_price"
        ):
            adj = result["recommendations"]["adjust_price"]
            params = {
                "newPrice": adj["suggested_price"],
                "priceDelta": abs(adj["suggested_price"] - body.user_price),
                "expectedSpeedup": "faster fill",
            }

        # Calculate next check-in based on confidence and progress
        wait_rec = result["recommendations"].get("wait", {})
        expected_hours = wait_rec.get("estimated_fill_time_minutes", 60) / 60
        if body.reported_progress < 30:
            next_check = max(30, int(expected_hours * 15))  # 15% of remaining, min 30m
        else:
            next_check = max(60, int(expected_hours * 25))  # 25% of remaining, min 1h

        logger.info(
            f"Guidance: item={body.item_id}, type={body.order_type}, "
            f"action={action.value}, confidence={result['confidence']}"
        )

        return GuidanceResponse(
            action=action,
            reason=result["reasoning"],
            confidence=result["confidence"],
            params=params,
            next_check_in_minutes=next_check,
        )

    except Exception as e:
        logger.error(f"Error getting guidance: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get guidance: {str(e)}"
        )


@app.get("/api/v1/trades/updates", response_model=TradeUpdatesResponse)
@limiter.limit(config.rate_limit_trade_updates)
async def get_trade_updates(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    tradeIds: str = Query(
        ..., description="Comma-separated trade/recommendation IDs to check"
    ),
    user_id: Optional[str] = Query(
        default=None, description="Hashed user ID for crowding awareness"
    ),
):
    """Poll for update recommendations on active trades.

    Batch endpoint to check multiple trades at once. Returns only trades
    with actionable updates (not HOLD status). Designed for efficient polling
    from web frontend.

    Update types:
    - ADJUST_PRICE: Market shifted, recommend adjusting sell price
    - SELL_NOW: Partial fill situation, recommend selling immediately
    - SWITCH_ITEM: Better opportunity found, recommend switching items

    Trades with HOLD (wait) status are not included in the response.

    Args:
        tradeIds: Comma-separated list of trade IDs (e.g., "rec_123,rec_456")
        user_id: Optional hashed user ID for crowding-aware suggestions
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Validate user_id format if provided
    if user_id and (
        len(user_id) != 64 or not all(c in "0123456789abcdef" for c in user_id.lower())
    ):
        raise HTTPException(
            status_code=400, detail="user_id must be SHA256 hash (64 hex characters)"
        )

    # Parse trade IDs
    trade_id_list = [tid.strip() for tid in tradeIds.split(",") if tid.strip()]
    if not trade_id_list:
        raise HTTPException(status_code=400, detail="No valid trade IDs provided")

    if len(trade_id_list) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 trade IDs per request")

    updates: list[TradeUpdate] = []
    update_counter = 0

    for trade_id in trade_id_list:
        try:
            # Get the original recommendation to extract trade details
            rec = engine.get_recommendation_by_id(trade_id)
            if rec is None:
                # Trade not found or expired, skip
                continue

            # For web polling, we assume trades are sell orders waiting to fill
            # The buy was already completed, now waiting for sell
            # Evaluate as a sell order using the recommendation's sell price
            result = engine.evaluate_active_order(
                item_id=rec["itemId"],
                order_type="sell",
                user_price=rec["sellPrice"],
                quantity=rec.get("quantity", 1),
                time_elapsed_minutes=30,  # Default assumption for polling
                user_id=user_id,
            )

            action = result["action"]
            confidence = result["confidence"]

            # Skip HOLD (wait) actions - only return actionable updates
            if action == "wait":
                continue

            # Determine urgency based on confidence
            if confidence >= 0.85:
                urgency = "high"
            elif confidence >= 0.65:
                urgency = "medium"
            else:
                urgency = "low"

            update_counter += 1
            update_id = f"update_{update_counter:03d}"

            if action == "adjust_price" and result["recommendations"].get(
                "adjust_price"
            ):
                adjust = result["recommendations"]["adjust_price"]
                updates.append(
                    TradeUpdate(
                        id=update_id,
                        tradeId=trade_id,
                        type="ADJUST_PRICE",
                        reason=result["reasoning"],
                        confidence=confidence,
                        urgency=urgency,
                        newSellPrice=adjust["suggested_price"],
                        originalSellPrice=rec["sellPrice"],
                        profitDelta=adjust.get("cost_difference", 0),
                    )
                )
            elif action == "abort_retry" and result["recommendations"].get(
                "abort_retry"
            ):
                abort = result["recommendations"]["abort_retry"]
                alternatives = abort.get("alternative_items", [])
                if alternatives:
                    alt = alternatives[0]  # Take the best alternative
                    updates.append(
                        TradeUpdate(
                            id=update_id,
                            tradeId=trade_id,
                            type="SWITCH_ITEM",
                            reason=result["reasoning"],
                            confidence=confidence,
                            urgency=urgency,
                            newItem=NewItemRecommendation(
                                itemId=alt["item_id"],
                                item=alt["item_name"],
                                buyPrice=alt.get("buy_price", rec.get("buyPrice", 0)),
                                sellPrice=alt.get(
                                    "sell_price", rec.get("sellPrice", 0)
                                ),
                                quantity=1,
                                expectedProfit=alt["expected_profit"],
                            ),
                        )
                    )
            elif action == "liquidate" and result["recommendations"].get("liquidate"):
                liq = result["recommendations"]["liquidate"]
                updates.append(
                    TradeUpdate(
                        id=update_id,
                        tradeId=trade_id,
                        type="SELL_NOW",
                        reason=result["reasoning"],
                        confidence=confidence,
                        urgency=urgency,
                        adjustedSellPrice=liq["instant_price"],
                    )
                )

        except Exception as e:
            logger.warning(f"Error evaluating trade {trade_id}: {e}")
            # Continue processing other trades
            continue

    # Calculate next check-in interval based on update urgency
    if any(u.urgency == "high" for u in updates):
        next_check_in = 15  # Check again in 15 seconds for high urgency
    elif updates:
        next_check_in = 30  # Check again in 30 seconds if any updates
    else:
        next_check_in = 60  # No updates, check again in 60 seconds

    logger.info(
        f"Trade updates: checked {len(trade_id_list)} trades, "
        f"returning {len(updates)} updates"
    )

    return TradeUpdatesResponse(updates=updates, nextCheckIn=next_check_in)


# =============================================================================
# Web Frontend Compatibility Routes (Issue #147)
# These are simplified route aliases for the gept-gg web frontend.
# They call the same handlers as the /api/v1/ endpoints.
# =============================================================================


@app.get("/health", response_model=HealthResponse)
@limiter.limit(config.rate_limit_health)
async def health_check_alias(request: Request):
    """Health check endpoint alias for web frontend.

    Calls the same handler as /api/v1/health.
    """
    return await health_check(request)


@app.get(
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
    user_hash: Optional[str] = Query(
        default=None, description="User hash for crowding tracking"
    ),
    capital: int = Query(ge=1000, description="Available capital in gp"),
    style: Literal["passive", "hybrid", "active"] = Query(
        default="hybrid", description="Trading style"
    ),
    risk: Literal["low", "medium", "high"] = Query(
        default="medium", description="Risk tolerance"
    ),
    count: int = Query(default=4, ge=1, le=20, description="Number of recommendations"),
    exclude_items: Optional[str] = Query(
        default=None, description="Comma-separated item IDs to exclude"
    ),
    max_offset_pct: Optional[float] = Query(
        default=None, ge=0.0125, le=0.0250, description="Maximum margin percentage"
    ),
    min_offset_pct: Optional[float] = Query(
        default=None, ge=0.0125, le=0.0250, description="Minimum margin percentage"
    ),
    include_metadata: bool = Query(
        default=False, description="Include freshness metadata"
    ),
):
    """Get recommendations for web frontend.

    Web-friendly endpoint with parameters matching the frontend's expected interface.
    Maps to /api/v1/recommendations internally.
    """
    return await get_recommendations_get(
        request=request,
        _api_key=_api_key,
        style=style,
        capital=capital,
        risk=risk,
        slots=count,
        user_tier="free",  # Web users default to free tier
        exclude=None,
        user_id=user_hash,
        exclude_item_ids=exclude_items,
        offset_pct=None,
        min_offset_pct=min_offset_pct,
        max_offset_pct=max_offset_pct,
        max_hour_offset=None,
        min_ev=None,
        include_metadata=include_metadata,
        return_all=False,  # Web endpoint uses standard slot-limited behavior
    )


@app.get("/item/{item_id}")
@limiter.limit(config.rate_limit_items)
async def get_item_web(
    request: Request,
    item_id: int,
    _api_key: Optional[str] = Depends(verify_api_key),
    capital: Optional[int] = Query(
        default=None, ge=1000, description="Available capital in gp"
    ),
    risk: Optional[Literal["low", "medium", "high"]] = Query(
        default=None, description="Risk tolerance"
    ),
    style: Optional[Literal["passive", "hybrid", "active"]] = Query(
        default=None, description="Trading style"
    ),
    include_price_history: bool = Query(
        default=False, description="Include price history for charts"
    ),
):
    """Get recommendation for a specific item by ID.

    Web-friendly endpoint alias for /api/v1/recommendations/item/{item_id}.
    """
    return await get_recommendation_by_item_id(
        request=request,
        item_id=item_id,
        _api_key=_api_key,
        capital=capital,
        risk=risk,
        style=style,
        slots=None,
        include_price_history=include_price_history,
    )


@app.get("/search-items", response_model=list[ItemSearchResult])
@limiter.limit(config.rate_limit_search)
async def search_items_web(
    request: Request,
    _api_key: Optional[str] = Depends(verify_api_key),
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=25, description="Max results"),
):
    """Search for items by name.

    Web-friendly endpoint alias for /api/v1/items/search.
    """
    return await search_items(request=request, _api_key=_api_key, q=q, limit=limit)


class TradeOutcomeWebRequest(BaseModel):
    """Request body for web frontend trade outcome reporting."""

    rec_id: str = Field(description="Recommendation ID that led to this trade")
    userId: str = Field(description="Hashed user ID (SHA256 of user identifier)")
    itemId: int = Field(description="OSRS item ID")
    itemName: str = Field(description="Item name")
    buyPrice: int = Field(ge=1, description="Buy price in gp")
    sellPrice: int = Field(ge=1, description="Sell price in gp")
    quantity: int = Field(ge=1, description="Quantity traded")
    actualProfit: int = Field(description="Actual profit/loss (can be negative)")
    reportedAt: str = Field(description="ISO 8601 timestamp")


@app.post("/trade-outcome", response_model=TradeOutcomeResponse)
@limiter.limit(config.rate_limit_outcomes)
async def report_trade_outcome_web(
    request: Request,
    body: TradeOutcomeWebRequest,
    _api_key: Optional[str] = Depends(verify_api_key),
):
    """Record a trade outcome from web frontend.

    Web-friendly endpoint with rec_id in the body instead of URL path.
    Maps to /api/v1/recommendations/{rec_id}/outcome internally.
    """
    # Convert to the format expected by the original handler
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
        request=request, rec_id=body.rec_id, body=outcome_body, _api_key=_api_key
    )


# =============================================================================
# Webhook Endpoints (Issue #167)
# These endpoints receive events from the web application
# =============================================================================


class TradeWebhookPayload(BaseModel):
    """Payload data for a trade webhook event."""

    itemId: int = Field(description="OSRS item ID")
    itemName: str = Field(description="Item name")
    buyPrice: int = Field(ge=1, description="Buy price in gp")
    sellPrice: int = Field(ge=1, description="Sell price in gp")
    quantity: int = Field(ge=1, description="Quantity")
    profit: Optional[int] = Field(
        default=None, description="Actual profit (only on TRADE_COMPLETED)"
    )
    recId: Optional[str] = Field(
        default=None, description="Links to original recommendation"
    )
    modelId: Optional[str] = Field(
        default=None, description="Links to model that made recommendation"
    )
    expectedHours: Optional[int] = Field(
        default=None, description="Expected trade duration in hours"
    )
    createdAt: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of trade creation"
    )


class TradeWebhookRequest(BaseModel):
    """Request body for trade webhook events."""

    eventType: Literal[
        "TRADE_CREATED", "TRADE_COMPLETED", "TRADE_CANCELLED", "TRADE_UPDATED"
    ] = Field(description="Type of trade event")
    timestamp: str = Field(description="ISO 8601 timestamp")
    userId: str = Field(description="User identifier")
    tradeId: str = Field(description="Unique trade identifier")
    payload: TradeWebhookPayload = Field(description="Trade details")


class TradeWebhookResponse(BaseModel):
    """Response for trade webhook events."""

    success: bool = Field(description="Whether the event was processed successfully")


@app.post("/webhooks/trades", response_model=TradeWebhookResponse)
async def receive_trade_webhook(
    request: Request,
    body: TradeWebhookRequest,
):
    """Receive trade lifecycle events from the web application.

    This endpoint receives TRADE_CREATED, TRADE_COMPLETED, TRADE_CANCELLED,
    and TRADE_UPDATED events. Events are verified using HMAC-SHA256 signatures.

    Headers:
        X-Webhook-Timestamp: Unix timestamp in milliseconds
        X-Webhook-Signature: HMAC-SHA256 signature of {timestamp}.{body}
    """
    # Get signature headers
    timestamp = request.headers.get("X-Webhook-Timestamp")
    signature = request.headers.get("X-Webhook-Signature")

    if not timestamp or not signature:
        logger.warning("Webhook request missing signature headers")
        raise HTTPException(
            status_code=401,
            detail="Missing X-Webhook-Timestamp or X-Webhook-Signature header",
        )

    # Verify signature
    try:
        raw_body = await request.body()
        verify_webhook_signature(
            body=raw_body.decode("utf-8"),
            timestamp=timestamp,
            signature=signature,
        )
    except WebhookSignatureError as e:
        logger.warning(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    # Check if trade event handler is initialized
    if trade_event_handler is None:
        logger.error("Trade event handler not initialized")
        raise HTTPException(status_code=503, detail="Service not ready")

    # Parse timestamp
    try:
        event_timestamp = datetime.fromisoformat(body.timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    # Build trade event
    event = TradeEvent(
        event_type=TradeEventType(body.eventType),
        timestamp=event_timestamp,
        user_id=body.userId,
        trade_id=body.tradeId,
        payload=TradePayload(
            item_id=body.payload.itemId,
            item_name=body.payload.itemName,
            buy_price=body.payload.buyPrice,
            sell_price=body.payload.sellPrice,
            quantity=body.payload.quantity,
            profit=body.payload.profit,
            rec_id=body.payload.recId,
            model_id=body.payload.modelId,
            expected_hours=body.payload.expectedHours,
            created_at=(
                datetime.fromisoformat(body.payload.createdAt.replace("Z", "+00:00"))
                if body.payload.createdAt
                else None
            ),
        ),
    )

    # Process the event
    try:
        await trade_event_handler.handle_event(event)
    except Exception as e:
        logger.error(f"Error processing trade event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process event")

    logger.info(
        "Trade webhook processed",
        event_type=body.eventType,
        trade_id=body.tradeId,
    )

    return TradeWebhookResponse(success=True)


# =============================================================================
# Root endpoint
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "GePT Recommendation Engine",
        "version": "2.0.0",
        "description": "Transforms raw predictions into trade recommendations",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "recommendations": "POST/GET /api/v1/recommendations",
            "by_id": "GET /api/v1/recommendations/{rec_id}",
            "by_item_id": "GET /api/v1/recommendations/item/{item_id}",
            "by_item_name": "GET /api/v1/recommendations/item?name={name}",
            "evaluate_order": "POST /api/v1/recommendations/update",
            "guidance": "POST /api/v1/guidance",
            "predictions": "GET /api/v1/predictions/{item_id}",
            "item_search": "GET /api/v1/items/search?q=<query>&limit=<limit>",
            "report_outcome": "POST /api/v1/recommendations/{rec_id}/outcome",
            "submit_feedback": "POST /api/v1/feedback",
            "feedback_analytics": "GET /api/v1/feedback/analytics",
        },
        "web_frontend_endpoints": {
            "recommendations": "GET /recommendations",
            "item": "GET /item/{item_id}",
            "search": "GET /search-items",
            "trade_outcome": "POST /trade-outcome",
            "health": "GET /health",
        },
        "webhooks": {
            "trades": "POST /webhooks/trades",
        },
    }


def create_app() -> FastAPI:
    """Create and return the FastAPI app instance."""
    return app
