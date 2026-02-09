"""Tests for model_id filtering behavior in PredictionLoader.

These tests do not require a real database. We intercept pandas.read_sql to
capture the SQLAlchemy query object and then assert the generated SQL includes
an outer `predictions.model_id = ...` filter when preferred_model_id is set.
"""

import re

import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

import src.prediction_loader as pl


def _compile_sql(query) -> str:
    return str(
        query.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_latest_predictions", {"limit": 5}),
        (
            "get_best_prediction_per_item",
            {"limit": 25, "min_volume_24h": 0},
        ),
    ],
)
def test_model_filter_applies_to_outer_query(monkeypatch, method: str, kwargs: dict):
    captured = {}

    def fake_read_sql(query, conn, *args, **kwargs2):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(pl.pd, "read_sql", fake_read_sql)

    loader = pl.PredictionLoader(
        db_connection_string="sqlite:///:memory:",
        preferred_model_id="gept_v2",
    )

    # Trigger query construction (execution is intercepted by fake_read_sql).
    getattr(loader, method)(**kwargs)

    q = captured.get("query")
    assert q is not None, "Expected pandas.read_sql to be called"
    sql = _compile_sql(q)

    # We expect the model_id filter to appear twice:
    # 1) in the MAX(time) subquery
    # 2) in the outer query WHERE clause
    matches = re.findall(r"\"?predictions\"?\.model_id\s*=\s*'gept_v2'", sql)
    assert len(matches) == 2, f"Expected outer + subquery model_id filter, got SQL:\n{sql}"


def test_no_model_filter_when_preferred_model_id_empty(monkeypatch):
    captured = {}

    def fake_read_sql(query, conn, *args, **kwargs):
        captured["query"] = query
        return pd.DataFrame()

    monkeypatch.setattr(pl.pd, "read_sql", fake_read_sql)

    loader = pl.PredictionLoader(
        db_connection_string="sqlite:///:memory:",
        preferred_model_id="",
    )

    loader.get_latest_predictions(limit=1)

    q = captured.get("query")
    assert q is not None, "Expected pandas.read_sql to be called"
    sql = _compile_sql(q)

    assert "model_id" not in sql, f"Did not expect model_id filtering, got SQL:\n{sql}"
