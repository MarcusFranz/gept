"""
Shared Health Check Module for GePT Collectors

Provides consistent health check endpoints that verify:
1. Database connection (if applicable)
2. Last successful collection time
3. API/service reachability
4. Overall service status

Usage:
    from shared.health import HealthChecker, create_health_app

    # Create health checker with checks
    health_checker = HealthChecker(
        service_name='latest',
        collection_interval=60,
        last_collection_gauge=LAST_COLLECTION,  # Prometheus gauge
    )

    # Add database check
    health_checker.set_db_connection(conn)

    # Add API check
    health_checker.set_api_url("https://prices.runescape.wiki/api/v1/osrs/latest")

    # Create Flask app for /health endpoint
    app = create_health_app(health_checker, metrics_port=9103)
"""

import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Default staleness thresholds (in seconds)
# Data is considered stale if older than 2x the collection interval
DEFAULT_STALE_MULTIPLIER = 2.0


class HealthChecker:
    """
    Health checker for collector services.

    Provides methods to check:
    - Database connectivity
    - Data freshness (last collection time)
    - External API availability
    """

    def __init__(
        self,
        service_name: str,
        collection_interval: int = 60,
        last_collection_gauge: Optional[Any] = None,
        stale_multiplier: float = DEFAULT_STALE_MULTIPLIER,
    ):
        """
        Initialize health checker.

        Args:
            service_name: Name of the service (for logging/status)
            collection_interval: Expected collection interval in seconds
            last_collection_gauge: Prometheus Gauge tracking last collection timestamp
            stale_multiplier: Multiplier for staleness threshold (default 2.0)
        """
        self.service_name = service_name
        self.collection_interval = collection_interval
        self.last_collection_gauge = last_collection_gauge
        self.stale_threshold = collection_interval * stale_multiplier

        # Optional components
        self._db_connection: Optional[Any] = None
        self._db_check_query: str = "SELECT 1"
        self._api_url: Optional[str] = None
        self._api_check_func: Optional[Callable[[], bool]] = None

        # Custom check functions
        self._custom_checks: Dict[str, Callable[[], bool]] = {}

        # Manual last collection tracking (if no gauge)
        self._last_collection_time: Optional[float] = None

    def set_db_connection(
        self, conn: Any, check_query: str = "SELECT 1"
    ) -> None:
        """Set database connection for health checks."""
        self._db_connection = conn
        self._db_check_query = check_query

    def set_api_url(self, url: str) -> None:
        """Set API URL for connectivity check."""
        self._api_url = url

    def set_api_check_func(self, func: Callable[[], bool]) -> None:
        """Set custom API check function."""
        self._api_check_func = func

    def add_custom_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Add a custom health check."""
        self._custom_checks[name] = check_func

    def record_collection(self) -> None:
        """Record a successful collection (if not using Prometheus gauge)."""
        self._last_collection_time = time.time()

    def check_db_connection(self) -> bool:
        """Check if database connection is alive."""
        if self._db_connection is None:
            return True  # No DB configured, skip check

        try:
            # Check if connection has a cursor method (psycopg2-style)
            if hasattr(self._db_connection, 'cursor'):
                with self._db_connection.cursor() as cur:
                    cur.execute(self._db_check_query)
                    cur.fetchone()
                return True
            # Check for execute method (duckdb-style)
            elif hasattr(self._db_connection, 'execute'):
                self._db_connection.execute(self._db_check_query).fetchone()
                return True
            return False
        except Exception as e:
            logger.warning(f"DB health check failed: {e}")
            return False

    def check_data_freshness(self) -> tuple[bool, float]:
        """
        Check if data collection is recent.

        Returns:
            Tuple of (is_fresh, age_seconds)
        """
        last_ts = None

        # Try Prometheus gauge first
        if self.last_collection_gauge is not None:
            try:
                # prometheus_client Gauge has ._value
                last_ts = self.last_collection_gauge._value.get()
            except Exception:
                pass

        # Fall back to manual tracking
        if last_ts is None or last_ts == 0:
            last_ts = self._last_collection_time

        if last_ts is None or last_ts == 0:
            # Never collected - stale
            return False, float('inf')

        age = time.time() - last_ts
        is_fresh = age <= self.stale_threshold

        return is_fresh, age

    def check_api_health(self) -> bool:
        """Check if external API is reachable."""
        # Custom check function takes priority
        if self._api_check_func is not None:
            try:
                return self._api_check_func()
            except Exception as e:
                logger.warning(f"API health check failed: {e}")
                return False

        # URL-based check
        if self._api_url is None:
            return True  # No API configured, skip check

        try:
            import httpx
            with httpx.Client(timeout=10.0) as client:
                resp = client.head(self._api_url)
                return resp.status_code < 500
        except Exception as e:
            logger.warning(f"API health check failed: {e}")
            return False

    def get_health_status(self) -> dict:
        """
        Get comprehensive health status.

        Returns:
            Dictionary with health status details
        """
        status = {
            "service": self.service_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {},
            "healthy": True,
        }

        # Database check
        db_ok = self.check_db_connection()
        status["checks"]["db_connected"] = db_ok
        if not db_ok:
            status["healthy"] = False

        # Data freshness check
        is_fresh, age_seconds = self.check_data_freshness()
        status["checks"]["data_fresh"] = is_fresh
        status["checks"]["last_collection_age_seconds"] = (
            age_seconds if age_seconds != float('inf') else None
        )
        status["checks"]["stale_threshold_seconds"] = self.stale_threshold
        if not is_fresh:
            status["healthy"] = False

        # API check
        api_ok = self.check_api_health()
        status["checks"]["api_reachable"] = api_ok
        if not api_ok:
            status["healthy"] = False

        # Custom checks
        for name, check_func in self._custom_checks.items():
            try:
                result = check_func()
                status["checks"][name] = result
                if not result:
                    status["healthy"] = False
            except Exception as e:
                logger.warning(f"Custom check '{name}' failed: {e}")
                status["checks"][name] = False
                status["healthy"] = False

        return status


class HealthHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoint."""

    health_checker: Optional[HealthChecker] = None

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.handle_health()
        elif self.path == '/ready':
            self.handle_ready()
        else:
            self.send_error(404)

    def handle_health(self):
        """Handle /health endpoint."""
        if self.health_checker is None:
            self.send_error(500, "Health checker not configured")
            return

        status = self.health_checker.get_health_status()
        http_code = 200 if status["healthy"] else 503

        response = json.dumps(status, indent=2)
        self.send_response(http_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response.encode())

    def handle_ready(self):
        """Handle /ready endpoint (kubernetes-style readiness)."""
        if self.health_checker is None:
            self.send_error(500, "Health checker not configured")
            return

        # For readiness, only check if service can accept requests
        # (simpler than full health check)
        db_ok = self.health_checker.check_db_connection()
        http_code = 200 if db_ok else 503

        self.send_response(http_code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK' if db_ok else b'NOT READY')


def start_health_server(
    health_checker: HealthChecker,
    port: int,
    host: str = '0.0.0.0'
) -> HTTPServer:
    """
    Start a health check HTTP server on a separate thread.

    Args:
        health_checker: HealthChecker instance
        port: Port to listen on
        host: Host to bind to (default 0.0.0.0)

    Returns:
        HTTPServer instance (already running in background)
    """
    # Create handler class with health_checker bound
    class Handler(HealthHTTPHandler):
        pass
    Handler.health_checker = health_checker

    server = HTTPServer((host, port), Handler)

    # Start in daemon thread
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    logger.info(f"Health server started on {host}:{port}")
    return server


def add_health_routes_to_prometheus(
    health_checker: HealthChecker,
    metrics_registry=None
) -> None:
    """
    Add /health endpoint alongside Prometheus /metrics endpoint.

    This patches the prometheus_client HTTP handler to also serve /health.

    Note: This is a convenience function. For more control, use start_health_server
    on a separate port.
    """
    from prometheus_client import exposition

    # Store original handler
    original_handler_class = exposition.MetricsHandler

    class CombinedHandler(original_handler_class):
        """Handler that serves both /metrics and /health."""

        _health_checker = health_checker

        def do_GET(self):
            if self.path == '/health':
                self._handle_health()
            elif self.path == '/ready':
                self._handle_ready()
            else:
                super().do_GET()

        def _handle_health(self):
            status = self._health_checker.get_health_status()
            http_code = 200 if status["healthy"] else 503

            response = json.dumps(status, indent=2).encode()
            self.send_response(http_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response))
            self.end_headers()
            self.wfile.write(response)

        def _handle_ready(self):
            db_ok = self._health_checker.check_db_connection()
            http_code = 200 if db_ok else 503

            self.send_response(http_code)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK' if db_ok else b'NOT READY')

    # Monkey-patch the handler
    exposition.MetricsHandler = CombinedHandler
