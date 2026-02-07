"""FastAPI dependencies for shared runtime resources."""


from fastapi import HTTPException, Request
from sqlalchemy.engine import Engine

from .api_state import RuntimeState
from .recommendation_engine import RecommendationEngine
from .trade_events import TradeEventHandler


def get_state(request: Request) -> RuntimeState:
    state = getattr(request.app.state, "runtime", None)
    if state is None:
        # Should never happen in production; `src.api` always sets this.
        raise HTTPException(status_code=503, detail="Runtime not initialized")
    return state


def get_engine(request: Request) -> RecommendationEngine:
    state = get_state(request)
    if state.engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return state.engine


def get_outcome_db_engine(request: Request) -> Engine:
    # Prefer explicit module-level overrides for test patching.
    import src.api as api_module

    if hasattr(api_module, "outcome_db_engine") and hasattr(
        api_module, "_OUTCOME_DB_ENGINE_UNSET"
    ):
        if api_module.outcome_db_engine is not api_module._OUTCOME_DB_ENGINE_UNSET:
            if api_module.outcome_db_engine is None:
                raise HTTPException(
                    status_code=503, detail="Outcome database not available"
                )
            return api_module.outcome_db_engine

    state = get_state(request)
    if state.outcome_db_engine is None:
        # Preserve legacy API error message for clients/tests.
        raise HTTPException(status_code=503, detail="Outcome database not available")
    return state.outcome_db_engine


def get_feedback_db_engine(request: Request) -> Engine:
    # Prefer explicit module-level overrides for test patching.
    import src.api as api_module

    if hasattr(api_module, "outcome_db_engine") and hasattr(
        api_module, "_OUTCOME_DB_ENGINE_UNSET"
    ):
        if api_module.outcome_db_engine is not api_module._OUTCOME_DB_ENGINE_UNSET:
            if api_module.outcome_db_engine is None:
                raise HTTPException(
                    status_code=503, detail="Feedback database not available"
                )
            return api_module.outcome_db_engine

    state = get_state(request)
    if state.outcome_db_engine is None:
        # Feedback endpoints historically returned a different error message.
        raise HTTPException(status_code=503, detail="Feedback database not available")
    return state.outcome_db_engine


def get_trade_event_handler(request: Request) -> TradeEventHandler:
    state = get_state(request)
    if state.trade_event_handler is None:
        raise HTTPException(status_code=503, detail="Trade event handler not initialized")
    return state.trade_event_handler
