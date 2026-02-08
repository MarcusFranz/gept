"""Tests for seeding active trades directly from the web app database.

These tests use SQLite to avoid requiring Neon/Postgres credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text

from src.api_lifespan import _resync_active_trades_from_db
from src.config import config
from src.trade_events import TradeEventHandler, TradeEventType


def _init_sqlite_active_trades(db_url: str) -> None:
    engine = create_engine(db_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE active_trades (
                      id TEXT PRIMARY KEY,
                      user_id TEXT NOT NULL,
                      item_id INTEGER NOT NULL,
                      item_name TEXT NOT NULL,
                      buy_price INTEGER NOT NULL,
                      sell_price INTEGER NOT NULL,
                      quantity INTEGER NOT NULL,
                      rec_id TEXT NULL,
                      model_id TEXT NULL,
                      expected_hours INTEGER NULL,
                      created_at TEXT NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO active_trades (
                      id, user_id, item_id, item_name, buy_price, sell_price, quantity,
                      rec_id, model_id, expected_hours, created_at
                    ) VALUES (
                      :id, :user_id, :item_id, :item_name, :buy_price, :sell_price, :quantity,
                      :rec_id, :model_id, :expected_hours, :created_at
                    )
                    """
                ),
                {
                    "id": "trade_1",
                    "user_id": "user_1",
                    "item_id": 4151,
                    "item_name": "Abyssal whip",
                    "buy_price": 100,
                    "sell_price": 110,
                    "quantity": 2,
                    "rec_id": "rec_1",
                    "model_id": "model_1",
                    "expected_hours": 3,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_resync_active_trades_from_db_seeds_trade(monkeypatch, tmp_path):
    db_url = f"sqlite:///{tmp_path / 'active_trades.db'}"
    _init_sqlite_active_trades(db_url)

    monkeypatch.setattr(config, "active_trades_db_connection_string", db_url)

    handler = TradeEventHandler()
    ok, seeded = await _resync_active_trades_from_db(handler)

    assert ok is True
    assert seeded == 1

    active = handler.get_active_trades()
    assert "trade_1" in active
    assert active["trade_1"].event_type == TradeEventType.TRADE_CREATED
    assert active["trade_1"].payload.item_id == 4151


@pytest.mark.asyncio
async def test_resync_active_trades_from_db_missing_table_returns_not_ok(monkeypatch, tmp_path):
    # SQLite DB without the active_trades table should trigger an error and be reported as not-ok.
    db_url = f"sqlite:///{tmp_path / 'empty.db'}"
    monkeypatch.setattr(config, "active_trades_db_connection_string", db_url)

    handler = TradeEventHandler()
    ok, seeded = await _resync_active_trades_from_db(handler)

    assert ok is False
    assert seeded == 0
