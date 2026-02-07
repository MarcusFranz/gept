"""Prediction loader for fetching pre-computed predictions from PostgreSQL."""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import and_, case, create_engine, func, literal_column, select, text
from sqlalchemy.pool import QueuePool

from .config import Config
from .schema import (
    items,
    model_registry,
    mv_volume_1h,
    mv_volume_24h,
    predictions,
    price_data_5min,
    prices_latest_1m,
)
from .osrs.acronyms import expand_acronym
from .wiki_api import get_wiki_api_client

logger = logging.getLogger(__name__)


# Shorthand aliases for table references used in query building
p = predictions
v = price_data_5min
i = items
m = model_registry
latest = prices_latest_1m


def _prediction_columns():
    """Standard prediction columns used across multiple queries."""
    return [
        p.c.item_id,
        p.c.item_name,
        p.c.hour_offset,
        p.c.offset_pct,
        p.c.fill_probability,
        p.c.expected_value,
        p.c.buy_price,
        p.c.sell_price,
        p.c.current_high,
        p.c.current_low,
        p.c.confidence,
        p.c.time.label("prediction_time"),
    ]


class PredictionLoader:
    """Fetches pre-computed predictions from the Ampere server's predictions table.

    The inference engine runs every 5 minutes on the Ampere server, storing
    predictions in the `predictions` table. This loader queries those predictions
    and filters them based on user constraints.
    """

    def __init__(
        self,
        db_connection_string: str,
        pool_size: int = 5,
        preferred_model_id: str = "",
        config: Optional[Config] = None,
    ):
        """Initialize database connection.

        Args:
            db_connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            preferred_model_id: If set, only serve predictions from this model_id
            config: Application config (created from env vars if not provided)
        """
        self.engine = create_engine(
            db_connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=3,
            pool_pre_ping=True,
        )
        self.preferred_model_id = preferred_model_id
        self.config = config or Config()
        if preferred_model_id:
            logger.info(f"Model filter active: preferred_model_id={preferred_model_id}")

    def _max_time_subquery(self):
        """Return a scalar subquery for MAX(time), scoped to preferred model if configured."""
        q = select(func.max(p.c.time))
        if self.preferred_model_id:
            q = q.where(p.c.model_id == self.preferred_model_id)
        return q.scalar_subquery()

    def _matview_exists(self, view_name: str) -> bool:
        """Check if a materialized view exists (cached per instance)."""
        if not hasattr(self, "_matview_cache"):
            self._matview_cache: dict[str, bool] = {}
        if view_name not in self._matview_cache:
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(
                        text("SELECT 1 FROM pg_matviews WHERE matviewname = :name"),
                        {"name": view_name},
                    ).fetchone()
                    self._matview_cache[view_name] = result is not None
            except Exception as e:
                logger.warning(f"Error checking matview '{view_name}': {e}")
                self._matview_cache[view_name] = False
        return self._matview_cache[view_name]

    def get_latest_predictions(
        self,
        min_fill_prob: float = 0.03,
        min_ev: float = 0.005,
        max_hour_offset: Optional[int] = None,
        item_ids: Optional[list[int]] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch latest predictions filtered by constraints.

        Args:
            min_fill_prob: Minimum fill probability (filter out unlikely fills)
            min_ev: Minimum expected value threshold
            max_hour_offset: Maximum hours ahead (for active vs passive trading)
            item_ids: Optional list of specific item IDs to fetch
            limit: Maximum number of predictions to return

        Returns:
            DataFrame with predictions sorted by expected_value descending
        """
        conditions = [
            p.c.time == self._max_time_subquery(),
            p.c.fill_probability >= min_fill_prob,
            p.c.expected_value >= min_ev,
        ]

        if max_hour_offset is not None:
            conditions.append(p.c.hour_offset <= max_hour_offset)

        if item_ids:
            conditions.append(p.c.item_id.in_(item_ids))

        query = (
            select(*_prediction_columns())
            .where(and_(*conditions))
            .order_by(p.c.expected_value.desc())
            .limit(limit)
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)

            if df.empty:
                logger.warning("No predictions found matching criteria")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error fetching predictions: {e}")
            return pd.DataFrame()

    def get_predictions_for_item(self, item_id: int) -> pd.DataFrame:
        """Get all latest predictions for a specific item.

        Args:
            item_id: OSRS item ID

        Returns:
            DataFrame with all hour_offset/offset_pct combinations for the item
        """
        query = (
            select(*_prediction_columns())
            .where(
                and_(
                    p.c.time == self._max_time_subquery(),
                    p.c.item_id == item_id,
                )
            )
            .order_by(p.c.hour_offset, p.c.offset_pct)
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            logger.error(f"Error fetching predictions for item {item_id}: {e}")
            return pd.DataFrame()

    def get_best_prediction_per_item(
        self,
        min_fill_prob: float = 0.03,
        min_ev: float = 0.005,
        min_hour_offset: Optional[int] = None,
        max_hour_offset: Optional[int] = None,
        min_offset_pct: Optional[float] = None,
        max_offset_pct: Optional[float] = None,
        limit: int = 100,
        min_volume_24h: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get the single best prediction for each item.

        Selects the hour_offset/offset_pct combination with highest EV per item.

        Args:
            min_fill_prob: Minimum fill probability
            min_ev: Minimum expected value
            min_hour_offset: Minimum hours ahead (for filtering short horizons)
            max_hour_offset: Maximum hours ahead
            min_offset_pct: Minimum offset percentage (e.g., 0.0125 for 1.25%)
            max_offset_pct: Maximum offset percentage (e.g., 0.0250 for 2.5%)
            limit: Maximum number of items
            min_volume_24h: Minimum 24-hour volume (excludes illiquid items)

        Returns:
            DataFrame with one row per item (best configuration)
        """
        max_time = self._max_time_subquery()
        needs_volume = min_volume_24h is not None and min_volume_24h > 0

        # Base WHERE conditions
        conditions = [
            p.c.time == max_time,
            p.c.fill_probability >= min_fill_prob,
            p.c.expected_value >= min_ev,
        ]
        if min_hour_offset is not None:
            conditions.append(p.c.hour_offset >= int(min_hour_offset))
        if max_hour_offset is not None:
            conditions.append(p.c.hour_offset <= int(max_hour_offset))
        if min_offset_pct is not None:
            conditions.append(p.c.offset_pct >= min_offset_pct)
        if max_offset_pct is not None:
            conditions.append(p.c.offset_pct <= max_offset_pct)

        # Build column list with optional volume
        rank_cols = list(_prediction_columns())

        rn_col = func.row_number().over(
            partition_by=p.c.item_id,
            order_by=p.c.expected_value.desc(),
        ).label("rn")

        if needs_volume:
            # Build volume source (matview or live aggregation)
            if self._matview_exists("mv_volume_24h"):
                vol_cte = (
                    select(
                        mv_volume_24h.c.item_id,
                        mv_volume_24h.c.total_volume,
                    )
                    .cte("volume_24h")
                )
            else:
                vol_cte = (
                    select(
                        v.c.item_id,
                        (
                            func.coalesce(func.sum(v.c.high_price_volume), 0)
                            + func.coalesce(func.sum(v.c.low_price_volume), 0)
                        ).label("total_volume"),
                    )
                    .where(v.c.timestamp >= func.now() - text("INTERVAL '24 hours'"))
                    .group_by(v.c.item_id)
                    .cte("volume_24h")
                )

            rank_cols.append(
                func.coalesce(vol_cte.c.total_volume, 0).label("volume_24h")
            )
            rank_cols.append(rn_col)

            # Price-tiered volume threshold
            vol_threshold = case(
                (func.coalesce(p.c.buy_price, 0) > 100_000_000, 100),
                (func.coalesce(p.c.buy_price, 0) > 10_000_000, 500),
                (
                    func.coalesce(p.c.buy_price, 0) > 1_000_000,
                    func.least(min_volume_24h, 2000),
                ),
                else_=min_volume_24h,
            )
            conditions.append(
                func.coalesce(vol_cte.c.total_volume, 0) >= vol_threshold
            )

            ranked = (
                select(*rank_cols)
                .select_from(
                    p.outerjoin(vol_cte, vol_cte.c.item_id == p.c.item_id)
                )
                .where(and_(*conditions))
                .cte("ranked")
            )
        else:
            # Fast path: skip volume calculation entirely
            rank_cols.append(literal_column("0").label("volume_24h"))
            rank_cols.append(rn_col)

            ranked = (
                select(*rank_cols)
                .where(and_(*conditions))
                .cte("ranked")
            )

        query = (
            select(ranked)
            .where(ranked.c.rn == 1)
            .order_by(ranked.c.expected_value.desc())
            .limit(limit)
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)

            # Drop internal columns (keep volume_24h for stability filter)
            for col in ["rn"]:
                if col in df.columns:
                    df = df.drop(col, axis=1)

            return df

        except Exception as e:
            logger.error(f"Error fetching best predictions per item: {e}")
            return pd.DataFrame()

    def get_latest_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the most recent predictions.

        Returns:
            Most recent prediction timestamp, or None if no data
        """
        query = select(func.max(p.c.time))
        if self.preferred_model_id:
            query = query.where(p.c.model_id == self.preferred_model_id)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result and result[0]:
                return result[0]
            return None

        except Exception as e:
            logger.error(f"Error fetching latest timestamp: {e}")
            return None

    def get_prediction_age_seconds(self) -> float:
        """Get age of the latest predictions in seconds.

        Returns:
            Age in seconds, or infinity if no predictions
        """
        latest = self.get_latest_timestamp()
        if latest is None:
            return float("inf")

        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)

        return (datetime.now(timezone.utc) - latest).total_seconds()

    def get_item_name(self, item_id: int) -> Optional[str]:
        """Get an item's human-readable name.

        The engine schema does not maintain an authoritative item-name mapping
        table today; the best available source is the predictions table's
        denormalized `item_name` column. If the item has no predictions (for
        example, newly added items), fall back to the OSRS Wiki mapping API.

        Returns:
            Item name if found, otherwise None.
        """
        query = select(p.c.item_name).where(p.c.item_id == item_id)
        if self.preferred_model_id:
            query = query.where(p.c.model_id == self.preferred_model_id)
        query = query.order_by(p.c.time.desc()).limit(1)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result and result[0]:
                return str(result[0])
            return self._get_wiki_item_name(item_id)

        except Exception as e:
            logger.debug(f"Could not fetch item name for item {item_id}: {e}")
            return self._get_wiki_item_name(item_id)

    def get_item_buy_limit(self, item_id: int) -> Optional[int]:
        """Get GE buy limit for an item.

        Tries OSRS Wiki API first (authoritative source), falls back to database.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit or None if not found
        """
        # Try Wiki API first (authoritative source)
        wiki_limit = self._get_wiki_buy_limit(item_id)
        if wiki_limit is not None:
            return wiki_limit

        # Fallback to database
        db_limit = self._get_db_buy_limit(item_id)
        if db_limit is not None:
            logger.debug(f"Using database buy limit for item {item_id}: {db_limit}")
            return db_limit

        return None

    def _get_db_buy_limit(self, item_id: int) -> Optional[int]:
        """Query database for item buy limit.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit from database or None if not found
        """
        query = select(i.c.buy_limit).where(i.c.item_id == item_id)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result and result[0]:
                return int(result[0])
            return None

        except Exception as e:
            logger.debug(f"Could not fetch buy limit from DB for item {item_id}: {e}")
            return None

    def _get_wiki_buy_limit(self, item_id: int) -> Optional[int]:
        """Get buy limit from OSRS Wiki API.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit from Wiki API or None if not available
        """
        if not self.config.wiki_api_enabled:
            return None

        try:
            wiki_client = get_wiki_api_client()
            return wiki_client.get_buy_limit(item_id)
        except Exception as e:
            logger.debug(
                f"Could not fetch buy limit from Wiki API for item {item_id}: {e}"
            )
            return None

    def _get_wiki_item_name(self, item_id: int) -> Optional[str]:
        """Get item name from OSRS Wiki API mapping.

        Args:
            item_id: OSRS item ID

        Returns:
            Item name from Wiki mapping, or None if not available.
        """
        if not self.config.wiki_api_enabled:
            return None

        try:
            wiki_client = get_wiki_api_client()
            return wiki_client.get_item_name(item_id)
        except Exception as e:
            logger.debug(
                f"Could not fetch item name from Wiki API for item {item_id}: {e}"
            )
            return None

    def get_batch_buy_limits(self, item_ids: list[int]) -> dict[int, int]:
        """Get GE buy limits for multiple items in batch.

        Fetches buy limits from Wiki API first (batched), then fills in
        remaining items from database in a single query.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to buy_limit (only includes items with known limits)
        """
        if not item_ids:
            return {}

        result: dict[int, int] = {}
        remaining_ids = set(item_ids)

        # Try Wiki API first (it has a cache, so this is efficient)
        result, remaining_ids = self._get_batch_wiki_buy_limits(list(remaining_ids))

        # Fetch remaining from database in one query
        if remaining_ids:
            db_limits = self._get_batch_db_buy_limits(list(remaining_ids))
            result.update(db_limits)

        return result

    def _get_batch_db_buy_limits(self, item_ids: list[int]) -> dict[int, int]:
        """Query database for buy limits in batch.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to buy_limit
        """
        if not item_ids:
            return {}

        query = (
            select(i.c.item_id, i.c.buy_limit)
            .where(
                and_(
                    i.c.item_id.in_(item_ids),
                    i.c.buy_limit.is_not(None),
                )
            )
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()

            return {int(row[0]): int(row[1]) for row in result if row[1]}

        except Exception as e:
            logger.debug(f"Could not fetch batch buy limits from DB: {e}")
            return {}

    def _get_batch_wiki_buy_limits(
        self, item_ids: list[int]
    ) -> tuple[dict[int, int], set[int]]:
        """Get buy limits from Wiki API for multiple items.

        Uses the wiki client's cache for efficiency.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Tuple of (dict mapping item_id to buy_limit, set of unfound item_ids)
        """
        if not self.config.wiki_api_enabled:
            return {}, set(item_ids)

        result: dict[int, int] = {}
        remaining: set[int] = set()

        try:
            wiki_client = get_wiki_api_client()
            for item_id in item_ids:
                limit = wiki_client.get_buy_limit(item_id)
                if limit is not None:
                    result[item_id] = limit
                else:
                    remaining.add(item_id)
        except Exception as e:
            logger.debug(f"Could not fetch batch buy limits from Wiki API: {e}")
            remaining = set(item_ids) - set(result.keys())

        return result, remaining

    def get_item_volume_24h(self, item_id: int) -> Optional[int]:
        """Get 24-hour trade volume for an item.

        Sums buy and sell volumes from the last 24 hours of 5-minute data.

        Args:
            item_id: OSRS item ID

        Returns:
            Total 24h volume (0 if no data), or None on query failure
        """
        if self._matview_exists("mv_volume_24h"):
            query = (
                select(mv_volume_24h.c.total_volume)
                .where(mv_volume_24h.c.item_id == item_id)
            )
        else:
            query = select(
                (
                    func.coalesce(func.sum(v.c.high_price_volume), 0)
                    + func.coalesce(func.sum(v.c.low_price_volume), 0)
                ).label("total_volume")
            ).where(
                and_(
                    v.c.item_id == item_id,
                    v.c.timestamp >= func.now() - text("INTERVAL '24 hours'"),
                )
            )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result and result[0] is not None:
                return int(result[0])
            return 0

        except Exception as e:
            logger.warning(f"Could not fetch 24h volume for item {item_id}: {e}")
            return None

    def get_item_volume_1h(self, item_id: int) -> Optional[int]:
        """Get last hour of trade volume from 5-minute data.

        Args:
            item_id: OSRS item ID

        Returns:
            Total 1h volume (0 if no data), or None on query failure
        """
        if self._matview_exists("mv_volume_1h"):
            query = (
                select(mv_volume_1h.c.total_volume)
                .where(mv_volume_1h.c.item_id == item_id)
            )
        else:
            query = select(
                (
                    func.coalesce(func.sum(v.c.high_price_volume), 0)
                    + func.coalesce(func.sum(v.c.low_price_volume), 0)
                ).label("total_volume")
            ).where(
                and_(
                    v.c.item_id == item_id,
                    v.c.timestamp >= func.now() - text("INTERVAL '1 hour'"),
                )
            )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result and result[0] is not None:
                return int(result[0])
            return 0

        except Exception as e:
            logger.warning(f"Could not fetch 1h volume for item {item_id}: {e}")
            return None

    def get_batch_volumes_24h(self, item_ids: list[int]) -> dict[int, int]:
        """Get 24-hour trade volumes for multiple items in a single query.

        Optimized batch fetch to eliminate N+1 queries.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to total 24h volume
        """
        if not item_ids:
            return {}

        if self._matview_exists("mv_volume_24h"):
            query = (
                select(mv_volume_24h.c.item_id, mv_volume_24h.c.total_volume)
                .where(mv_volume_24h.c.item_id.in_(item_ids))
            )
        else:
            query = (
                select(
                    v.c.item_id,
                    (
                        func.coalesce(func.sum(v.c.high_price_volume), 0)
                        + func.coalesce(func.sum(v.c.low_price_volume), 0)
                    ).label("total_volume"),
                )
                .where(
                    and_(
                        v.c.item_id.in_(item_ids),
                        v.c.timestamp >= func.now() - text("INTERVAL '24 hours'"),
                    )
                )
                .group_by(v.c.item_id)
            )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()

            return {int(row[0]): int(row[1]) for row in result}

        except Exception as e:
            logger.warning(f"Could not fetch batch 24h volumes: {e}")
            return {}

    def get_batch_volumes_1h(self, item_ids: list[int]) -> dict[int, int]:
        """Get last hour trade volumes for multiple items in a single query.

        Optimized batch fetch to eliminate N+1 queries.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to total 1h volume
        """
        if not item_ids:
            return {}

        if self._matview_exists("mv_volume_1h"):
            query = (
                select(mv_volume_1h.c.item_id, mv_volume_1h.c.total_volume)
                .where(mv_volume_1h.c.item_id.in_(item_ids))
            )
        else:
            query = (
                select(
                    v.c.item_id,
                    (
                        func.coalesce(func.sum(v.c.high_price_volume), 0)
                        + func.coalesce(func.sum(v.c.low_price_volume), 0)
                    ).label("total_volume"),
                )
                .where(
                    and_(
                        v.c.item_id.in_(item_ids),
                        v.c.timestamp >= func.now() - text("INTERVAL '1 hour'"),
                    )
                )
                .group_by(v.c.item_id)
            )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()

            return {int(row[0]): int(row[1]) for row in result}

        except Exception as e:
            logger.warning(f"Could not fetch batch 1h volumes: {e}")
            return {}

    def get_price_history(self, item_id: int, hours: int = 24) -> list[dict]:
        """Get price history for an item.

        Returns hourly price data for mini chart display.

        Args:
            item_id: OSRS item ID
            hours: Number of hours of history (default 24)

        Returns:
            List of dicts with 'timestamp' and 'price' (midpoint) keys
        """
        bucket = func.date_trunc("hour", v.c.timestamp).label("bucket")
        query = (
            select(
                bucket,
                func.avg(v.c.avg_high_price).label("avg_high_price"),
                func.avg(v.c.avg_low_price).label("avg_low_price"),
            )
            .where(
                and_(
                    v.c.item_id == item_id,
                    v.c.timestamp >= func.now() - func.make_interval(  # years, months, weeks, days, hours
                        0, 0, 0, 0, hours
                    ),
                )
            )
            .group_by(bucket)
            # Take most-recent buckets, then reverse in Python so the caller gets asc timestamps.
            .order_by(bucket.desc())
            .limit(hours)
        )

        try:
            with self.engine.connect() as conn:
                result = list(reversed(conn.execute(query).fetchall()))

            history = []
            for row in result:
                timestamp, high, low = row
                if high is not None and low is not None:
                    midpoint = int((float(high) + float(low)) / 2)
                    history.append(
                        {
                            "timestamp": (
                                timestamp.isoformat()
                                if hasattr(timestamp, "isoformat")
                                else str(timestamp)
                            ),
                            "price": midpoint,
                        }
                    )

            return history

        except Exception as e:
            logger.warning(f"Could not fetch price history for item {item_id}: {e}")
            return []

    def get_extended_price_history(self, item_id: int, hours: int = 24) -> list[dict]:
        """Get extended price history with high/low data for sparkline charts.

        Returns hourly price data with all price points for web frontend.

        Args:
            item_id: OSRS item ID
            hours: Number of hours of history (default 24)

        Returns:
            List of dicts with 'timestamp', 'high', 'low', 'avgHigh', 'avgLow' keys
        """
        bucket = func.date_trunc("hour", v.c.timestamp).label("bucket")
        query = (
            select(
                bucket,
                func.max(v.c.avg_high_price).label("high"),
                func.min(v.c.avg_low_price).label("low"),
                func.avg(v.c.avg_high_price).label("avg_high_price"),
                func.avg(v.c.avg_low_price).label("avg_low_price"),
            )
            .where(
                and_(
                    v.c.item_id == item_id,
                    v.c.timestamp >= func.now() - func.make_interval(  # years, months, weeks, days, hours
                        0, 0, 0, 0, hours
                    ),
                )
            )
            .group_by(bucket)
            # Take most-recent buckets, then reverse in Python so the caller gets asc timestamps.
            .order_by(bucket.desc())
            .limit(hours)
        )

        try:
            with self.engine.connect() as conn:
                result = list(reversed(conn.execute(query).fetchall()))

            history = []
            for row in result:
                timestamp, high, low, avg_high, avg_low = row
                # Use available price data, falling back as needed.
                actual_high = high if high is not None else avg_high
                actual_low = low if low is not None else avg_low
                actual_avg_high = avg_high if avg_high is not None else actual_high
                actual_avg_low = avg_low if avg_low is not None else actual_low

                if actual_high is not None and actual_low is not None:
                    history.append(
                        {
                            "timestamp": (
                                timestamp.isoformat()
                                if hasattr(timestamp, "isoformat")
                                else str(timestamp)
                            ),
                            "high": int(round(float(actual_high))),
                            "low": int(round(float(actual_low))),
                            "avgHigh": (
                                int(round(float(actual_avg_high)))
                                if actual_avg_high is not None
                                else int(round(float(actual_high)))
                            ),
                            "avgLow": (
                                int(round(float(actual_avg_low)))
                                if actual_avg_low is not None
                                else int(round(float(actual_low)))
                            ),
                        }
                    )

            return history

        except Exception as e:
            logger.warning(
                f"Could not fetch extended price history for item {item_id}: {e}"
            )
            return []

    def get_item_trend(self, item_id: int) -> str:
        """Determine price trend for an item over last 4 hours.

        Args:
            item_id: OSRS item ID

        Returns:
            'Rising', 'Falling', or 'Stable'
        """
        query = (
            select(v.c.avg_high_price, v.c.avg_low_price)
            .where(v.c.item_id == item_id)
            .order_by(v.c.timestamp.desc())
            .limit(4)
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()

            if len(result) < 2:
                return "Stable"

            # Compare current vs 4h ago using midpoint
            current_mid = (
                (result[0][0] + result[0][1]) / 2
                if result[0][0] and result[0][1]
                else None
            )
            oldest_mid = (
                (result[-1][0] + result[-1][1]) / 2
                if result[-1][0] and result[-1][1]
                else None
            )

            if current_mid is None or oldest_mid is None or oldest_mid == 0:
                return "Stable"

            change_pct = (current_mid - oldest_mid) / oldest_mid

            if change_pct > 0.02:  # > 2% increase
                return "Rising"
            elif change_pct < -0.02:  # > 2% decrease
                return "Falling"
            else:
                return "Stable"

        except Exception as e:
            logger.warning(f"Could not fetch trend for item {item_id}: {e}")
            return "Stable"

    def get_batch_trends(self, item_ids: list[int]) -> dict[int, str]:
        """Determine price trends for multiple items in a single query.

        Fetches the last 4 hours of price data for all items and calculates
        trends based on the price change percentage.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to trend ('Rising', 'Falling', or 'Stable')
        """
        if not item_ids:
            return {}

        rn = func.row_number().over(
            partition_by=v.c.item_id,
            order_by=v.c.timestamp.desc(),
        ).label("rn")

        ranked = (
            select(
                v.c.item_id,
                v.c.avg_high_price,
                v.c.avg_low_price,
                v.c.timestamp,
                rn,
            )
            .where(
                and_(
                    v.c.item_id.in_(item_ids),
                    v.c.timestamp >= func.now() - text("INTERVAL '4 hours'"),
                )
            )
            .cte("ranked_prices")
        )

        query = (
            select(
                ranked.c.item_id,
                ranked.c.avg_high_price,
                ranked.c.avg_low_price,
                ranked.c.rn,
            )
            .where(ranked.c.rn.in_([1, 4]))
            .order_by(ranked.c.item_id, ranked.c.rn)
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()

            # Group by item_id and calculate trends
            item_prices: dict[int, dict] = {}
            for row in result:
                item_id_val, high, low, rn_val = row
                item_id_val = int(item_id_val)
                if item_id_val not in item_prices:
                    item_prices[item_id_val] = {}

                if rn_val == 1:  # Current (most recent)
                    item_prices[item_id_val]["current"] = (high, low)
                elif rn_val == 4:  # 4 hours ago
                    item_prices[item_id_val]["oldest"] = (high, low)

            # Calculate trends
            trends: dict[int, str] = {}
            for item_id_val, prices in item_prices.items():
                current = prices.get("current")
                oldest = prices.get("oldest")

                # Need both prices to calculate trend
                if not current or not oldest:
                    trends[item_id_val] = "Stable"
                    continue

                current_mid = (
                    (current[0] + current[1]) / 2 if current[0] and current[1] else None
                )
                oldest_mid = (
                    (oldest[0] + oldest[1]) / 2 if oldest[0] and oldest[1] else None
                )

                if current_mid is None or oldest_mid is None or oldest_mid == 0:
                    trends[item_id_val] = "Stable"
                    continue

                change_pct = (current_mid - oldest_mid) / oldest_mid

                if change_pct > 0.02:  # > 2% increase
                    trends[item_id_val] = "Rising"
                elif change_pct < -0.02:  # > 2% decrease
                    trends[item_id_val] = "Falling"
                else:
                    trends[item_id_val] = "Stable"

            # Default to Stable for items with no data
            for item_id_val in item_ids:
                if item_id_val not in trends:
                    trends[item_id_val] = "Stable"

            return trends

        except Exception as e:
            logger.warning(f"Could not fetch batch trends: {e}")
            # Return Stable for all items on error
            return {item_id_val: "Stable" for item_id_val in item_ids}

    def health_check(self) -> dict:
        """Check database connection and prediction freshness.

        Returns:
            Health status dictionary
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            age = self.get_prediction_age_seconds()

            if age == float("inf"):
                status = "error"
                message = "No predictions found"
            elif age > 360:  # > 6 min = missed a 5-min cycle
                status = "warning"
                message = f"Predictions are {age:.0f}s old (stale)"
            else:
                status = "ok"
                message = f"Predictions are {age:.0f}s old"

            return {
                "status": status,
                "component": "prediction_loader",
                "message": message,
                "prediction_age_seconds": age if age != float("inf") else None,
                "connected": True,
            }

        except Exception as e:
            return {
                "status": "error",
                "component": "prediction_loader",
                "error": str(e),
                "connected": False,
            }

    def search_items_by_name(self, query_str: str, limit: int = 10) -> list[dict]:
        """Search for items by name with fuzzy matching.

        Supports OSRS acronym expansion (e.g., "ags" -> "armadyl godsword").

        Args:
            query_str: Search query (may be an acronym)
            limit: Maximum number of results

        Returns:
            List of dicts with item_id and item_name
        """
        # Expand acronym if recognized
        expanded_query = expand_acronym(query_str)

        # Escape LIKE wildcard characters in search term
        safe_query = expanded_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        max_time = self._max_time_subquery()

        # Subquery: distinct items matching the search
        items_sub = (
            select(p.c.item_id, p.c.item_name)
            .distinct()
            .where(
                and_(
                    p.c.time == max_time,
                    func.lower(p.c.item_name).like(
                        func.lower(f"%{safe_query}%"), escape="\\"
                    ),
                )
            )
            .subquery("items")
        )

        # Order by match quality: exact > starts-with > contains
        order_priority = case(
            (func.lower(items_sub.c.item_name) == func.lower(expanded_query), 0),
            (
                func.lower(items_sub.c.item_name).like(
                    func.lower(f"{safe_query}%"), escape="\\"
                ),
                1,
            ),
            else_=2,
        )

        sql = (
            select(items_sub.c.item_id, items_sub.c.item_name)
            .order_by(order_priority, items_sub.c.item_name)
            .limit(limit)
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql)
                return [{"item_id": row[0], "item_name": row[1]} for row in result]
        except Exception as e:
            logger.error(f"Error searching items: {e}")
            return []

    def close(self):
        """Close database connections."""
        self.engine.dispose()

    # === Model Registry Methods ===

    def get_active_model_ids(self) -> set[int]:
        """Get model_ids with ACTIVE status.

        Returns:
            Set of model_ids that are currently active
        """
        query = select(m.c.model_id).where(m.c.status == "ACTIVE")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.error(f"Error fetching active model IDs: {e}")
            return set()

    def get_model_status(self, model_id: int) -> Optional[str]:
        """Get status for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model status string or None if not found
        """
        query = select(m.c.status).where(m.c.model_id == model_id)

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error fetching model status for {model_id}: {e}")
            return None

    def get_models_by_item(
        self,
        item_id: int,
        status_filter: Optional[str] = None,
    ) -> list[dict]:
        """Get models for an item, optionally filtered by status.

        Args:
            item_id: OSRS item ID
            status_filter: Optional status to filter by (e.g., 'ACTIVE')

        Returns:
            List of model dicts with model_id, status, mean_auc, trained_at
        """
        cols = [m.c.model_id, m.c.item_id, m.c.status, m.c.mean_auc, m.c.trained_at]
        conditions = [m.c.item_id == item_id]
        if status_filter:
            conditions.append(m.c.status == status_filter)

        query = (
            select(*cols)
            .where(and_(*conditions))
            .order_by(m.c.trained_at.desc())
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            return [
                {
                    "model_id": row[0],
                    "item_id": row[1],
                    "status": row[2],
                    "mean_auc": float(row[3]) if row[3] else None,
                    "trained_at": row[4].isoformat() if row[4] else None,
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error fetching models for item {item_id}: {e}")
            return []

    def get_active_model_for_item(self, item_id: int) -> Optional[dict]:
        """Get the active model for an item.

        Args:
            item_id: OSRS item ID

        Returns:
            Model dict or None if no active model exists
        """
        models = self.get_models_by_item(item_id, status_filter="ACTIVE")
        return models[0] if models else None

    def get_model_registry_stats(self) -> dict:
        """Get statistics about model registry status.

        Returns:
            Dict with counts by status
        """
        query = (
            select(m.c.status, func.count().label("count"))
            .group_by(m.c.status)
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            stats = {row[0]: row[1] for row in result}
            return {
                "active": stats.get("ACTIVE", 0),
                "deprecated": stats.get("DEPRECATED", 0),
                "sunset": stats.get("SUNSET", 0),
                "archived": stats.get("ARCHIVED", 0),
                "total": sum(stats.values()),
            }
        except Exception as e:
            logger.error(f"Error fetching model registry stats: {e}")
            return {
                "active": 0,
                "deprecated": 0,
                "sunset": 0,
                "archived": 0,
                "total": 0,
            }

    def get_items_with_active_models(self) -> set[int]:
        """Get item IDs that have at least one ACTIVE model.

        Returns:
            Set of item_ids with active models
        """
        query = select(m.c.item_id).distinct().where(m.c.status == "ACTIVE")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.error(f"Error fetching items with active models: {e}")
            return set()

    def get_latest_price(self, item_id: int) -> Optional[dict]:
        """Get most recent price tick for an item from 1-minute data.

        Args:
            item_id: OSRS item ID

        Returns:
            Dictionary with keys: timestamp, high, low, high_time, low_time
            Returns None if no data found
        """
        query = (
            select(latest.c.timestamp, latest.c.high, latest.c.low, latest.c.high_time, latest.c.low_time)
            .where(latest.c.item_id == item_id)
            .order_by(latest.c.timestamp.desc())
            .limit(1)
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result is None:
                return None

            return {
                "timestamp": result[0],
                "high": result[1],
                "low": result[2],
                "high_time": result[3],
                "low_time": result[4],
            }

        except Exception as e:
            logger.error(f"Error fetching latest price for item {item_id}: {e}")
            return None

    def get_predictions_for_items(self, item_ids: list[int]) -> pd.DataFrame:
        """Get all latest predictions for multiple items.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            DataFrame with all predictions for the items
            (all hour_offset/offset_pct combinations)
        """
        if not item_ids:
            return pd.DataFrame()

        query = (
            select(*_prediction_columns())
            .where(
                and_(
                    p.c.time == self._max_time_subquery(),
                    p.c.item_id.in_(item_ids),
                )
            )
            .order_by(p.c.item_id, p.c.hour_offset, p.c.offset_pct)
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            logger.error(f"Error fetching predictions for items: {e}")
            return pd.DataFrame()
