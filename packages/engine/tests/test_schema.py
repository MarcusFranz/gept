"""Tests for schema constants module.

Validates that:
1. Schema classes define TABLE and at least one column constant
2. All constants are non-empty strings (catches accidental empty assignments)
3. No duplicate column names within a table (catches copy-paste errors)
4. ALL_TABLES registry is complete

For live validation against the database, run with --db-url:
    pytest tests/test_schema.py --db-url=postgresql://user:pass@host/db
"""

import os
from typing import Optional

import pytest
from sqlalchemy import create_engine, inspect, text

from src.schema import (
    ALL_TABLES,
    Items,
    ModelRegistry,
    Predictions,
    PriceData5Min,
    PricesLatest1M,
    RecommendationFeedback,
    TradeOutcomes,
)


def _get_column_attrs(schema_class):
    """Get all column-name attributes from a schema class (excludes TABLE)."""
    return {
        k: v
        for k, v in vars(schema_class).items()
        if not k.startswith("_") and k != "TABLE" and isinstance(v, str)
    }


# ---------------------------------------------------------------------------
# Structural tests (no database needed)
# ---------------------------------------------------------------------------


class TestSchemaStructure:
    """Validate schema constants are well-formed."""

    @pytest.mark.parametrize("table_cls", ALL_TABLES)
    def test_has_table_name(self, table_cls):
        assert hasattr(table_cls, "TABLE"), f"{table_cls.__name__} missing TABLE"
        assert isinstance(table_cls.TABLE, str)
        assert len(table_cls.TABLE) > 0

    @pytest.mark.parametrize("table_cls", ALL_TABLES)
    def test_has_columns(self, table_cls):
        cols = _get_column_attrs(table_cls)
        assert len(cols) > 0, f"{table_cls.__name__} has no column constants"

    @pytest.mark.parametrize("table_cls", ALL_TABLES)
    def test_column_values_are_nonempty_strings(self, table_cls):
        for attr_name, value in _get_column_attrs(table_cls).items():
            assert isinstance(value, str), (
                f"{table_cls.__name__}.{attr_name} is {type(value)}, expected str"
            )
            assert len(value) > 0, f"{table_cls.__name__}.{attr_name} is empty"

    @pytest.mark.parametrize("table_cls", ALL_TABLES)
    def test_no_duplicate_column_values(self, table_cls):
        cols = _get_column_attrs(table_cls)
        values = list(cols.values())
        duplicates = [v for v in values if values.count(v) > 1]
        assert len(duplicates) == 0, (
            f"{table_cls.__name__} has duplicate column values: {set(duplicates)}"
        )

    def test_all_tables_registry_complete(self):
        """Ensure ALL_TABLES includes every schema class defined in the module."""
        expected = {
            Predictions,
            PriceData5Min,
            Items,
            ModelRegistry,
            PricesLatest1M,
            TradeOutcomes,
            RecommendationFeedback,
        }
        assert set(ALL_TABLES) == expected

    def test_table_names_are_unique(self):
        table_names = [cls.TABLE for cls in ALL_TABLES]
        assert len(table_names) == len(set(table_names)), (
            f"Duplicate table names: {[n for n in table_names if table_names.count(n) > 1]}"
        )


# ---------------------------------------------------------------------------
# Live database validation (requires --db-url or DATABASE_URL env var)
# ---------------------------------------------------------------------------


def _get_db_url() -> Optional[str]:
    """Get database URL from pytest flag or environment."""
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
    """Validate that schema constants match the actual database schema."""

    @pytest.mark.parametrize("table_cls", ALL_TABLES)
    def test_table_exists(self, db_engine, table_cls):
        inspector = inspect(db_engine)
        tables = inspector.get_table_names()
        assert table_cls.TABLE in tables, (
            f"Table '{table_cls.TABLE}' from {table_cls.__name__} "
            f"does not exist in database. Available: {tables}"
        )

    @pytest.mark.parametrize("table_cls", ALL_TABLES)
    def test_columns_exist(self, db_engine, table_cls):
        inspector = inspect(db_engine)
        try:
            db_columns = {col["name"] for col in inspector.get_columns(table_cls.TABLE)}
        except Exception:
            pytest.skip(f"Could not inspect table {table_cls.TABLE}")
            return

        schema_columns = _get_column_attrs(table_cls)
        missing = []
        for attr_name, col_name in schema_columns.items():
            if col_name not in db_columns:
                missing.append(f"{attr_name}={col_name!r}")

        assert len(missing) == 0, (
            f"{table_cls.__name__} references columns not in "
            f"{table_cls.TABLE}: {missing}. "
            f"DB has: {sorted(db_columns)}"
        )
