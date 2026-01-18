"""Data loader for fetching price data from PostgreSQL."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


class DataLoader:
    """Fetches price data from PostgreSQL database."""

    def __init__(self, db_connection_string: str, pool_size: int = 10):
        """Initialize database connection.

        Args:
            db_connection_string: PostgreSQL connection string
            pool_size: Connection pool size
        """
        self.engine = create_engine(
            db_connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=5,
            pool_pre_ping=True,
        )
        self._items_cache: Optional[pd.DataFrame] = None
        self._items_cache_time: Optional[datetime] = None

    def get_recent_prices(
        self, item_id: int, minutes: int = 1440
    ) -> Optional[pd.DataFrame]:
        """Fetch last N minutes of price data for one item.

        Args:
            item_id: OSRS item ID
            minutes: Number of minutes of history to fetch (default 1440 = 24h)

        Returns:
            DataFrame with columns: timestamp, high, low, high_volume, low_volume
            Returns None if no data found
        """
        query = text(
            """
            SELECT
                timestamp,
                high,
                low,
                high_volume,
                low_volume
            FROM prices_1m
            WHERE item_id = :item_id
              AND timestamp >= NOW() - make_interval(mins => :minutes)
            ORDER BY timestamp ASC
        """
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    query, conn, params={"item_id": item_id, "minutes": minutes}
                )

            if df.empty:
                logger.warning(f"No price data found for item {item_id}")
                return None

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")

            # Fill any gaps in the data
            df = df.astype(
                {
                    "high": "float64",
                    "low": "float64",
                    "high_volume": "float64",
                    "low_volume": "float64",
                }
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching prices for item {item_id}: {e}")
            return None

    def get_all_recent_prices(self, minutes: int = 1440) -> dict[int, pd.DataFrame]:
        """Fetch last N minutes for all items.

        Args:
            minutes: Number of minutes of history to fetch

        Returns:
            Dictionary mapping item_id to DataFrame
        """
        query = text(
            """
            SELECT
                item_id,
                timestamp,
                high,
                low,
                high_volume,
                low_volume
            FROM prices_1m
            WHERE timestamp >= NOW() - make_interval(mins => :minutes)
            ORDER BY item_id, timestamp ASC
        """
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"minutes": minutes})

            if df.empty:
                logger.warning("No price data found")
                return {}

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.astype(
                {
                    "high": "float64",
                    "low": "float64",
                    "high_volume": "float64",
                    "low_volume": "float64",
                }
            )

            # Group by item_id
            result = {}
            for item_id, group in df.groupby("item_id"):
                item_df = group.drop("item_id", axis=1).set_index("timestamp")
                result[int(item_id)] = item_df

            return result

        except Exception as e:
            logger.error(f"Error fetching all prices: {e}")
            return {}

    def get_latest_price(self, item_id: int) -> Optional[dict]:
        """Get most recent price tick for one item.

        Args:
            item_id: OSRS item ID

        Returns:
            Dictionary with keys: timestamp, high, low, high_volume, low_volume
            Returns None if no data found
        """
        query = text(
            """
            SELECT
                timestamp,
                high,
                low,
                high_volume,
                low_volume
            FROM prices_1m
            WHERE item_id = :item_id
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_id": item_id}).fetchone()

            if result is None:
                return None

            return {
                "timestamp": result[0],
                "high": result[1],
                "low": result[2],
                "high_volume": result[3],
                "low_volume": result[4],
            }

        except Exception as e:
            logger.error(f"Error fetching latest price for item {item_id}: {e}")
            return None

    def get_latest_timestamp(self) -> Optional[datetime]:
        """Get the most recent timestamp in the database.

        Returns:
            Most recent timestamp, or None if no data
        """
        query = text(
            """
            SELECT MAX(timestamp) FROM prices_1m
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()

            if result and result[0]:
                return result[0]
            return None

        except Exception as e:
            logger.error(f"Error fetching latest timestamp: {e}")
            return None

    def get_items(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get item metadata.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            DataFrame with item metadata
        """
        # Use cached data if available and less than 1 hour old
        if (
            not force_refresh
            and self._items_cache is not None
            and self._items_cache_time is not None
            and datetime.now(timezone.utc) - self._items_cache_time < timedelta(hours=1)
        ):
            return self._items_cache

        query = text(
            """
            SELECT
                item_id,
                item_name,
                buy_limit,
                members,
                high_alch,
                low_alch
            FROM items
            ORDER BY item_id
        """
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)

            self._items_cache = df
            self._items_cache_time = datetime.now(timezone.utc)
            return df

        except Exception as e:
            logger.error(f"Error fetching items: {e}")
            return pd.DataFrame()

    def get_item_name(self, item_id: int) -> Optional[str]:
        """Get item name by ID.

        Args:
            item_id: OSRS item ID

        Returns:
            Item name or None if not found
        """
        items = self.get_items()
        if items.empty:
            return None

        match = items[items["item_id"] == item_id]
        if match.empty:
            return None

        return match.iloc[0]["item_name"]

    def get_item_buy_limit(self, item_id: int) -> Optional[int]:
        """Get GE buy limit for an item.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit or None if not found
        """
        items = self.get_items()
        if items.empty:
            return None

        match = items[items["item_id"] == item_id]
        if match.empty:
            return None

        buy_limit = match.iloc[0]["buy_limit"]
        return int(buy_limit) if pd.notna(buy_limit) else None

    def health_check(self) -> dict:
        """Check database connection health.

        Returns:
            Dictionary with health status
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            latest = self.get_latest_timestamp()
            if latest:
                age = (
                    datetime.now(timezone.utc) - latest.replace(tzinfo=timezone.utc)
                ).total_seconds()
            else:
                age = float("inf")

            return {
                "status": "ok" if age < 300 else "warning",
                "component": "database",
                "latest_data_age_seconds": age,
                "connected": True,
            }

        except Exception as e:
            return {
                "status": "error",
                "component": "database",
                "error": str(e),
                "connected": False,
            }

    def close(self):
        """Close database connections."""
        self.engine.dispose()
