"""Database schema constants for type-safe SQL query building.

Defines table and column names as Python constants so that:
- Column name typos become AttributeError at import time, not runtime SQL failures
- Schema drift is caught by validation tests (see tests/test_schema.py)
- IDE autocomplete and "find usages" work for all column references

Usage:
    from .schema import Predictions, PriceData5Min

    query = text(f"SELECT {Predictions.ITEM_ID} FROM {Predictions.TABLE}")
"""


class Predictions:
    """predictions table — pre-computed model predictions."""

    TABLE = "predictions"

    TIME = "time"
    ITEM_ID = "item_id"
    ITEM_NAME = "item_name"
    HOUR_OFFSET = "hour_offset"
    OFFSET_PCT = "offset_pct"
    FILL_PROBABILITY = "fill_probability"
    EXPECTED_VALUE = "expected_value"
    BUY_PRICE = "buy_price"
    SELL_PRICE = "sell_price"
    CURRENT_HIGH = "current_high"
    CURRENT_LOW = "current_low"
    CONFIDENCE = "confidence"
    MODEL_ID = "model_id"


class PriceData5Min:
    """price_data_5min table — 5-minute OHLCV price snapshots."""

    TABLE = "price_data_5min"

    ITEM_ID = "item_id"
    TIMESTAMP = "timestamp"
    HIGH_PRICE = "high_price"
    LOW_PRICE = "low_price"
    HIGH_PRICE_VOLUME = "high_price_volume"
    LOW_PRICE_VOLUME = "low_price_volume"
    AVG_HIGH_PRICE = "avg_high_price"
    AVG_LOW_PRICE = "avg_low_price"


class Items:
    """items table — OSRS item metadata."""

    TABLE = "items"

    ITEM_ID = "item_id"
    BUY_LIMIT = "buy_limit"


class ModelRegistry:
    """model_registry table — trained model lifecycle tracking."""

    TABLE = "model_registry"

    MODEL_ID = "model_id"
    ITEM_ID = "item_id"
    STATUS = "status"
    MEAN_AUC = "mean_auc"
    TRAINED_AT = "trained_at"


class PricesLatest1M:
    """prices_latest_1m table — most recent 1-minute price ticks."""

    TABLE = "prices_latest_1m"

    ITEM_ID = "item_id"
    TIMESTAMP = "timestamp"
    HIGH = "high"
    LOW = "low"
    HIGH_TIME = "high_time"
    LOW_TIME = "low_time"


class TradeOutcomes:
    """trade_outcomes table — recorded user trade results."""

    TABLE = "trade_outcomes"

    USER_ID_HASH = "user_id_hash"
    REC_ID = "rec_id"
    ITEM_ID = "item_id"
    ITEM_NAME = "item_name"
    BUY_PRICE = "buy_price"
    SELL_PRICE = "sell_price"
    QUANTITY = "quantity"
    ACTUAL_PROFIT = "actual_profit"
    REPORTED_AT = "reported_at"


class RecommendationFeedback:
    """recommendation_feedback table — user feedback on recommendations."""

    TABLE = "recommendation_feedback"

    ID = "id"
    USER_ID_HASH = "user_id_hash"
    REC_ID = "rec_id"
    ITEM_ID = "item_id"
    ITEM_NAME = "item_name"
    FEEDBACK_TYPE = "feedback_type"
    SIDE = "side"
    NOTES = "notes"
    RECOMMENDED_PRICE = "recommended_price"
    ACTUAL_PRICE = "actual_price"
    SUBMITTED_AT = "submitted_at"


# All schema classes for validation tests
ALL_TABLES = [
    Predictions,
    PriceData5Min,
    Items,
    ModelRegistry,
    PricesLatest1M,
    TradeOutcomes,
    RecommendationFeedback,
]
