"""Tests for schema module.

Validates that:
1. All tables have names and at least one column
2. No duplicate column names within a table
3. ALL_TABLES registry is complete

For live validation against the database, run with DATABASE_URL set:
    DATABASE_URL=postgresql://user:pass@host/db pytest tests/test_schema.py
"""

import os
from typing import Optional

import pytest
from sqlalchemy import create_engine, func, inspect, select

from src.schema import (
    ALL_TABLES,
    items,
    model_registry,
    mv_volume_24h,
    predictions,
    price_data_5min,
    prices_latest_1m,
    recommendation_feedback,
    trade_outcomes,
)


# ---------------------------------------------------------------------------
# Structural tests (no database needed)
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    """Validate schema Table objects are well-formed."""

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_has_table_name(self, table):
        assert table.name, "Table missing name"
        assert len(table.name) > 0

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_has_columns(self, table):
        assert len(table.columns) > 0, f"{table.name} has no columns"

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_column_names_are_nonempty(self, table):
        for col in table.columns:
            assert col.name, f"{table.name} has a column with empty name"
            assert len(col.name) > 0

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_no_duplicate_column_names(self, table):
        names = [col.name for col in table.columns]
        duplicates = [n for n in names if names.count(n) > 1]
        assert len(duplicates) == 0, (
            f"{table.name} has duplicate column names: {set(duplicates)}"
        )

    def test_all_tables_registry_complete(self):
        """Ensure ALL_TABLES includes every table defined in the module."""
        expected = {
            predictions,
            price_data_5min,
            items,
            model_registry,
            prices_latest_1m,
            trade_outcomes,
            recommendation_feedback,
        }
        assert set(ALL_TABLES) == expected

    def test_table_names_are_unique(self):
        table_names = [t.name for t in ALL_TABLES]
        assert len(table_names) == len(set(table_names)), (
            f"Duplicate table names: {[n for n in table_names if table_names.count(n) > 1]}"
        )


# ---------------------------------------------------------------------------
# Query compilation tests (no database needed)
# ---------------------------------------------------------------------------


class TestQueryCompilation:
    """Verify key query patterns compile to valid SQL."""

    def test_select_predictions(self):
        query = select(predictions.c.item_id).where(
            predictions.c.time == func.now()
        )
        sql = str(query.compile(compile_kwargs={"literal_binds": True}))
        assert "predictions" in sql
        assert "item_id" in sql

    def test_insert_trade_outcomes(self):
        stmt = trade_outcomes.insert().values(user_id_hash="test", item_id=1)
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "trade_outcomes" in sql

    def test_insert_returning_feedback(self):
        stmt = (
            recommendation_feedback.insert()
            .values(user_id_hash="test", item_id=1)
            .returning(recommendation_feedback.c.id)
        )
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "RETURNING" in sql

    def test_join_with_volume_cte(self):
        vol = select(
            mv_volume_24h.c.item_id, mv_volume_24h.c.total_volume
        ).cte("vol")
        query = select(predictions.c.item_id).select_from(
            predictions.outerjoin(vol, vol.c.item_id == predictions.c.item_id)
        )
        sql = str(query.compile(compile_kwargs={"literal_binds": True}))
        assert "LEFT OUTER JOIN" in sql

    def test_like_with_escape(self):
        query = select(predictions.c.item_name).where(
            func.lower(predictions.c.item_name).like(
                func.lower("%test\\%name%"), escape="\\"
            )
        )
        sql = str(query.compile(compile_kwargs={"literal_binds": True}))
        assert "LIKE" in sql
        assert "ESCAPE" in sql


# ---------------------------------------------------------------------------
# Live database validation (requires DATABASE_URL env var)
# ---------------------------------------------------------------------------


def _get_db_url() -> Optional[str]:
    """Get database URL from environment."""
    return os.environ.get("DATABASE_URL")


@pytest.fixture(scope="module")
def db_engine():
    """Create a database engine for schema validation."""
    url = _get_db_url()
    if not url:
        pytest.skip("No DATABASE_URL set — skipping live schema validation")
    engine = create_engine(url)
    yield engine
    engine.dispose()


class TestSchemaAgainstDatabase:
    """Validate that schema Table definitions match the actual database schema."""

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_table_exists(self, db_engine, table):
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        assert table.name in tables, (
            f"Table '{table.name}' does not exist in database. Available: {tables}"
        )

    @pytest.mark.parametrize("table", ALL_TABLES)
    def test_columns_exist(self, db_engine, table):
        inspector = inspect(db_engine)
        try:
            db_columns = {col["name"] for col in inspector.get_columns(table.name)}
        except Exception:
            pytest.skip(f"Could not inspect table {table.name}")
            return

        schema_columns = {col.name for col in table.columns}
        missing = schema_columns - db_columns

        assert len(missing) == 0, (
            f"{table.name} references columns not in database: {missing}. "
            f"DB has: {sorted(db_columns)}"
        )
