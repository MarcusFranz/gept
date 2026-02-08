"""FastAPI lifespan (startup/shutdown) for the recommendation engine API."""


import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from sqlalchemy import create_engine, text

from .alert_dispatcher import AlertDispatcher
from .api_state import RuntimeState
from .cache import close_cache, init_cache
from .config import config
from .logging_config import get_logger
from .recommendation_engine import RecommendationEngine
from .trade_events import TradeEventHandler
from .trade_price_monitor import TradePriceMonitor

logger = get_logger(__name__)

# Cleanup interval in seconds (run every 5 minutes)
CROWDING_CLEANUP_INTERVAL_SECONDS = 300


async def _crowding_cleanup_loop(app: FastAPI) -> None:
    """Background task that periodically cleans up the crowding tracker."""
    while True:
        try:
            await asyncio.sleep(CROWDING_CLEANUP_INTERVAL_SECONDS)
            state: RuntimeState = app.state.runtime
            if state.engine is not None:
                state.engine.crowding_tracker.cleanup_all()
                logger.debug("Crowding tracker cleanup completed")
        except asyncio.CancelledError:
            logger.info("Crowding cleanup task cancelled")
            break
        except Exception as e:
            logger.error("Error in crowding cleanup task", error=str(e))


async def _resync_active_trades() -> None:
    """Trigger a resync of active trades from the web app."""
    if not config.trade_webhooks_enabled:
        logger.info("Active trade resync skipped", reason="TRADE_WEBHOOKS_ENABLED=false")
        return

    if not config.web_app_resync_url:
        logger.info("Active trade resync skipped", reason="WEB_APP_RESYNC_URL not configured")
        return

    if not config.webhook_secret:
        logger.warning("Active trade resync skipped", reason="WEBHOOK_SECRET not configured")
        return

    headers = {
        "Authorization": f"Bearer {config.webhook_secret}",
        # Some WAF/CDN setups strip Authorization headers. Include a fallback header
        # that the web app can verify.
        "X-Gept-Webhook-Secret": config.webhook_secret,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Cloudflare/WAF can block POSTs to internal endpoints; GET is sufficient here
            # because the endpoint is still authenticated via Authorization header.
            response = await client.get(config.web_app_resync_url, headers=headers)

        if 200 <= response.status_code < 300:
            dispatched = None
            try:
                payload = response.json()
                dispatched = payload.get("data", {}).get("dispatched")
            except Exception:
                dispatched = None
            logger.info(
                "Active trade resync completed",
                status_code=response.status_code,
                dispatched=dispatched,
            )
            return

        logger.warning("Active trade resync failed", status_code=response.status_code)
    except Exception as e:
        logger.warning("Active trade resync error", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Ensure runtime is present even for tests that import `src.api` directly.
    if not hasattr(app.state, "runtime") or app.state.runtime is None:
        app.state.runtime = RuntimeState()

    state: RuntimeState = app.state.runtime

    # Startup
    state.startup_time = time.time()
    logger.info("Initializing recommendation engine...")

    errors = config.validate()
    if errors:
        logger.error("Configuration errors", errors=errors)
        raise RuntimeError(f"Configuration errors: {errors}")

    state.engine = RecommendationEngine(
        db_connection_string=config.db_connection_string,
        config=config,
    )

    # Test connection
    health = state.engine.health_check()
    if health["status"] == "error":
        logger.error("Engine health check failed", health=health)
        raise RuntimeError("Failed to connect to predictions database")

    pred_age = health["checks"][0].get("prediction_age_seconds", "unknown")
    logger.info("Engine initialized", prediction_age_seconds=pred_age)

    # Initialize trade event handler for webhook integration
    state.trade_event_handler = TradeEventHandler(
        crowding_tracker=state.engine.crowding_tracker,
        recommendation_engine=state.engine,
    )
    logger.info("Trade event handler initialized")

    if config.trade_webhooks_enabled:
        # Resync active trades after engine restart
        await _resync_active_trades()
    else:
        logger.warning("Trade webhooks disabled", reason="TRADE_WEBHOOKS_ENABLED=false")

    if config.price_drop_monitor_enabled:
        # Initialize alert dispatcher for price monitoring
        state.alert_dispatcher = AlertDispatcher()
        logger.info("Alert dispatcher initialized")

        # Start trade price monitor background task
        trade_monitor = TradePriceMonitor(
            trade_event_handler=state.trade_event_handler,
            prediction_loader=state.engine.loader,
            alert_dispatcher=state.alert_dispatcher,
            config=config,
        )
        state.monitor_task = asyncio.create_task(trade_monitor.run())
        logger.info("Trade price monitor started")
    else:
        logger.warning("Price drop monitor disabled", reason="PRICE_DROP_MONITOR_ENABLED=false")

    # Initialize outcome database connection (optional - for ML feedback loop)
    if config.outcome_db_connection_string:
        try:
            state.outcome_db_engine = create_engine(
                config.outcome_db_connection_string,
                pool_size=config.outcome_db_pool_size,
                pool_pre_ping=True,
            )
            # Test connection
            with state.outcome_db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Outcome database connection established")
        except Exception as e:
            logger.warning(
                "Outcome database connection failed; disabling trade outcome recording",
                error=str(e),
            )
            state.outcome_db_engine = None
    else:
        logger.info(
            "Outcome database not configured; trade outcome recording disabled",
            reason="OUTCOME_DB_CONNECTION_STRING missing",
        )

    # Start background cleanup task for crowding tracker
    state.cleanup_task = asyncio.create_task(_crowding_cleanup_loop(app))
    logger.info(
        "Crowding cleanup task started",
        interval_seconds=CROWDING_CLEANUP_INTERVAL_SECONDS,
    )

    # Initialize Redis cache (optional - for response caching)
    cache_available = await init_cache()
    if cache_available:
        logger.info("Redis response cache initialized")
    else:
        logger.info("Redis response cache not available, using direct database queries")

    # Mark as ready for startup probes
    state.is_ready = True
    logger.info("Application startup complete, ready to serve requests")

    yield

    # Mark as not ready during shutdown
    state.is_ready = False

    # Shutdown
    # Cancel the trade price monitor task
    if state.monitor_task is not None:
        state.monitor_task.cancel()
        try:
            await state.monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("Trade price monitor stopped")

    if state.alert_dispatcher is not None:
        await state.alert_dispatcher.close()
        logger.info("Alert dispatcher closed")

    # Cancel the cleanup task
    if state.cleanup_task is not None:
        state.cleanup_task.cancel()
        try:
            await state.cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Crowding cleanup task stopped")

    # Close cache connection
    await close_cache()

    if state.outcome_db_engine is not None:
        state.outcome_db_engine.dispose()
        logger.info("Outcome database connection closed")

    if state.engine is not None:
        state.engine.close()

    logger.info("Recommendation engine shut down")
