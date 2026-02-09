"""Pydantic models for the FastAPI surface area.

Keep these definitions out of `src/api.py` to reduce coupling between request
handling and schema definitions.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .app_metadata import APP_VERSION


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
    offsetPct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Offset percentage used for this recommendation (from predictions)",
    )
    quantity: int
    capitalRequired: int
    expectedProfit: int
    expectedProfitEV: Optional[int] = Field(
        default=None,
        description="Conservative expected profit (EV) accounting for bag-holding downside",
    )
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
    offsetPct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Offset percentage used for the underlying recommendation (optional)",
    )
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
    offsetPct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Offset percentage used for the underlying recommendation (optional)",
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
    # Optional model selection
    use_beta_model: bool = Field(
        default=False,
        description="Use beta model predictions if available (falls back to preferred model if not configured)",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Explicit model_id to use for predictions (overrides use_beta_model when it matches configured models)",
    )


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
    reported_progress: int = Field(
        ge=0, le=100, description="User-reported progress 0-100"
    )


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
    offset_pct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Offset percentage used for this opportunity (from predictions)",
    )
    quantity: int
    capital_required: int
    expected_profit: int
    expected_profit_ev: Optional[int] = Field(
        default=None,
        description="Conservative expected profit (EV) accounting for bag-holding downside",
    )
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


class CacheClearResponse(BaseModel):
    """Response for cache clear endpoint."""

    success: bool
    message: str
    keys_cleared: int


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


class TradeWebhookPayload(BaseModel):
    """Payload data for a trade webhook event."""

    itemId: int = Field(description="OSRS item ID")
    itemName: str = Field(description="Item name")
    buyPrice: int = Field(ge=1, description="Buy price in gp")
    sellPrice: int = Field(ge=1, description="Sell price in gp")
    offsetPct: Optional[float] = Field(
        default=None,
        ge=0.0125,
        le=0.0250,
        description="Offset percentage used for the underlying recommendation (optional)",
    )
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
