"""Runtime state for the FastAPI server.

We keep mutable runtime resources (DB engines, background tasks, etc.) in a
single place and attach an instance to `app.state.runtime`.
"""


import asyncio
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.engine import Engine

from .alert_dispatcher import AlertDispatcher
from .recommendation_engine import RecommendationEngine
from .trade_events import TradeEventHandler


@dataclass(slots=True)
class RuntimeState:
    """Mutable runtime resources for the API process."""

    engine: Optional[RecommendationEngine] = None
    outcome_db_engine: Optional[Engine] = None
    cleanup_task: Optional[asyncio.Task] = None
    monitor_task: Optional[asyncio.Task] = None
    alert_dispatcher: Optional[AlertDispatcher] = None
    startup_time: Optional[float] = None
    is_ready: bool = False
    trade_event_handler: Optional[TradeEventHandler] = None
