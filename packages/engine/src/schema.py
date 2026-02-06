"""Database schema for type-safe SQLAlchemy Core query building.

Defines tables as SQLAlchemy Table objects so that:
- Queries use Column objects instead of f-string interpolation
- Column name typos become AttributeError at import time
- Schema drift is caught by validation tests (see tests/test_schema.py)
- IDE autocomplete and "find usages" work for all column references

Usage:
    from .schema import predictions, price_data_5min

    query = select(predictions.c.item_id).where(predictions.c.model_id == mid)
"""

from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    column as sa_column,
    table as sa_table,
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Domain tables
# ---------------------------------------------------------------------------

predictions = Table(
    "predictions",
    metadata,
    Column("time", DateTime(timezone=True), nullable=False),
    Column("item_id", Integer, nullable=False),
    Column("item_name", String),
    Column("hour_offset", Integer, nullable=False),
    Column("offset_pct", Numeric(5, 4), nullable=False),
    Column("fill_probability", Numeric(7, 6)),
    Column("expected_value", Numeric(8, 6)),
    Column("buy_price", Numeric(12, 2)),
    Column("sell_price", Numeric(12, 2)),
    Column("current_high", Numeric(12, 2)),
    Column("current_low", Numeric(12, 2)),
    Column("confidence", Text),
    Column("model_id", Text),
)

price_data_5min = Table(
    "price_data_5min",
    metadata,
    Column("item_id", Integer, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("high_price_volume", Integer),
    Column("low_price_volume", Integer),
    Column("avg_high_price", Integer),
    Column("avg_low_price", Integer),
)

items = Table(
    "items",
    metadata,
    Column("item_id", Integer, primary_key=True),
    Column("buy_limit", Integer),
)

model_registry = Table(
    "model_registry",
    metadata,
    Column("model_id", Text, primary_key=True),
    Column("item_id", Integer),
    Column("status", Text),
    Column("mean_auc", Numeric),
    Column("trained_at", DateTime(timezone=True)),
)

prices_latest_1m = Table(
    "prices_latest_1m",
    metadata,
    Column("item_id", Integer),
    Column("timestamp", DateTime(timezone=True)),
    Column("high", Integer),
    Column("low", Integer),
    Column("high_time", Integer),
    Column("low_time", Integer),
)

trade_outcomes = Table(
    "trade_outcomes",
    metadata,
    Column("user_id_hash", Text),
    Column("rec_id", Text),
    Column("item_id", Integer),
    Column("item_name", Text),
    Column("buy_price", Numeric(12, 2)),
    Column("sell_price", Numeric(12, 2)),
    Column("quantity", Integer),
    Column("actual_profit", Numeric(12, 2)),
    Column("reported_at", DateTime(timezone=True)),
)

recommendation_feedback = Table(
    "recommendation_feedback",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id_hash", Text),
    Column("rec_id", Text),
    Column("item_id", Integer),
    Column("item_name", Text),
    Column("feedback_type", Text),
    Column("side", Text),
    Column("notes", Text),
    Column("recommended_price", Numeric(12, 2)),
    Column("actual_price", Numeric(12, 2)),
    Column("submitted_at", DateTime(timezone=True)),
)

# ---------------------------------------------------------------------------
# Materialized view references (lightweight, no metadata binding)
# ---------------------------------------------------------------------------

mv_volume_24h = sa_table(
    "mv_volume_24h",
    sa_column("item_id"),
    sa_column("total_volume"),
)

mv_volume_1h = sa_table(
    "mv_volume_1h",
    sa_column("item_id"),
    sa_column("total_volume"),
)

# ---------------------------------------------------------------------------
# Registry for validation tests
# ---------------------------------------------------------------------------

ALL_TABLES = [
    predictions,
    price_data_5min,
    items,
    model_registry,
    prices_latest_1m,
    trade_outcomes,
    recommendation_feedback,
]
