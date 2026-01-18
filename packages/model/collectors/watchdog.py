#!/usr/bin/env python3
"""Watchdog - Container Health Monitoring and Automatic Recovery Service.

Monitors Docker container health status and automatically restarts unhealthy
containers. Alerts if recovery fails after max attempts.

Requires Docker socket access (/var/run/docker.sock).
"""

import logging
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import docker
from docker.errors import DockerException, NotFound
from prometheus_client import start_http_server

# Add shared module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared.metrics import MonitorMetrics
from shared.health import HealthChecker, add_health_routes_to_prometheus

# Configuration
METRICS_PORT = int(os.getenv("METRICS_PORT", "9108"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))
MAX_RESTART_ATTEMPTS = int(os.getenv("MAX_RESTART_ATTEMPTS", "3"))
RESTART_WINDOW_MINUTES = int(os.getenv("RESTART_WINDOW_MINUTES", "30"))
COOLDOWN_MINUTES = int(os.getenv("COOLDOWN_MINUTES", "5"))

# Container name patterns to monitor (prefix matching)
MONITORED_CONTAINERS = [
    "osrs-ge-collector",
    "osrs-hourly-collector",
    "osrs-news-collector",
    "osrs-latest-1m",
    "osrs-dashboard",
]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize metrics
metrics = MonitorMetrics("watchdog")

# Health checker for /health endpoint
health_checker = HealthChecker(
    service_name='watchdog',
    collection_interval=CHECK_INTERVAL,
)

# Graceful Shutdown
shutdown_requested = False


def signal_handler(signum: int, frame: Any) -> None:
    global shutdown_requested
    logger.info("Shutdown requested...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class ContainerState:
    """Track state for a monitored container."""

    def __init__(self, name: str):
        self.name = name
        self.restart_times: List[datetime] = []
        self.consecutive_unhealthy = 0
        self.last_restart: Optional[datetime] = None
        self.recovery_failed = False

    def record_restart(self) -> None:
        """Record a restart attempt."""
        now = datetime.now()
        self.restart_times.append(now)
        self.last_restart = now

        # Clean up old restart times
        cutoff = now - timedelta(minutes=RESTART_WINDOW_MINUTES)
        self.restart_times = [t for t in self.restart_times if t > cutoff]

    def get_restart_count(self) -> int:
        """Get number of restarts in the window."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=RESTART_WINDOW_MINUTES)
        return len([t for t in self.restart_times if t > cutoff])

    def can_restart(self) -> bool:
        """Check if we can attempt another restart."""
        # Check max attempts
        if self.get_restart_count() >= MAX_RESTART_ATTEMPTS:
            return False

        # Check cooldown
        if self.last_restart:
            cooldown_end = self.last_restart + timedelta(minutes=COOLDOWN_MINUTES)
            if datetime.now() < cooldown_end:
                return False

        return True


class Watchdog:
    """Container health watchdog."""

    def __init__(self):
        self.client: Optional[docker.DockerClient] = None
        self.container_states: Dict[str, ContainerState] = {}

    def connect(self) -> bool:
        """Connect to Docker daemon."""
        try:
            self.client = docker.from_env()
            self.client.ping()
            logger.info("Connected to Docker daemon")
            return True
        except DockerException as e:
            logger.error(f"Failed to connect to Docker: {e}")
            return False

    def get_monitored_containers(self) -> Dict[str, Any]:
        """Get list of monitored containers and their status."""
        if not self.client:
            return {}

        containers = {}
        try:
            for container in self.client.containers.list(all=True):
                name = container.name
                if any(name.startswith(prefix) for prefix in MONITORED_CONTAINERS):
                    containers[name] = {
                        "id": container.short_id,
                        "status": container.status,
                        "health": self._get_health_status(container),
                    }
        except DockerException as e:
            logger.error(f"Error listing containers: {e}")

        return containers

    def _get_health_status(self, container) -> str:
        """Get health status from container."""
        try:
            container.reload()
            health = container.attrs.get("State", {}).get("Health", {})
            return health.get("Status", "unknown")
        except Exception:
            return "unknown"

    def check_container_health(self, container_name: str) -> bool:
        """Check if a container is healthy."""
        if not self.client:
            return False

        try:
            container = self.client.containers.get(container_name)
            health = self._get_health_status(container)

            # Consider healthy if running and (healthy or no health check)
            is_running = container.status == "running"
            is_healthy = health in ("healthy", "unknown")

            return is_running and is_healthy
        except NotFound:
            logger.warning(f"Container {container_name} not found")
            return False
        except DockerException as e:
            logger.error(f"Error checking {container_name}: {e}")
            return False

    def restart_container(self, container_name: str, timeout: int = 30) -> bool:
        """Restart a container."""
        if not self.client:
            return False

        try:
            container = self.client.containers.get(container_name)
            logger.info(f"Restarting container: {container_name}")
            container.restart(timeout=timeout)

            # Wait for container to start
            time.sleep(5)

            # Verify restart succeeded
            container.reload()
            if container.status == "running":
                logger.info(f"Container {container_name} restarted successfully")
                return True
            else:
                logger.error(
                    f"Container {container_name} failed to start "
                    f"(status: {container.status})"
                )
                return False

        except NotFound:
            logger.error(f"Container {container_name} not found for restart")
            return False
        except DockerException as e:
            logger.error(f"Error restarting {container_name}: {e}")
            return False

    def process_container(self, container_name: str, status: dict) -> None:
        """Process health check for a single container."""
        # Get or create state
        if container_name not in self.container_states:
            self.container_states[container_name] = ContainerState(container_name)
        state = self.container_states[container_name]

        # Check health
        is_healthy = status["status"] == "running" and status["health"] in (
            "healthy",
            "unknown",
        )

        # Record health metric
        metrics.record_container_health(container_name, is_healthy)

        if is_healthy:
            # Container is healthy, reset state
            if state.consecutive_unhealthy > 0:
                logger.info(f"Container {container_name} is now healthy")
            state.consecutive_unhealthy = 0
            state.recovery_failed = False
            metrics.set_recovery_failed(container_name, False)
            return

        # Container is unhealthy
        state.consecutive_unhealthy += 1
        logger.warning(
            f"Container {container_name} unhealthy "
            f"(status: {status['status']}, health: {status['health']}, "
            f"consecutive: {state.consecutive_unhealthy})"
        )

        # Need 2 consecutive unhealthy checks before action
        if state.consecutive_unhealthy < 2:
            return

        # Check if we can restart
        if not state.can_restart():
            restart_count = state.get_restart_count()
            if restart_count >= MAX_RESTART_ATTEMPTS:
                if not state.recovery_failed:
                    logger.error(
                        f"Container {container_name} recovery failed after "
                        f"{MAX_RESTART_ATTEMPTS} attempts"
                    )
                    state.recovery_failed = True
                    metrics.set_recovery_failed(container_name, True)
            else:
                logger.info(
                    f"Container {container_name} in cooldown, skipping restart"
                )
            return

        # Attempt restart
        success = self.restart_container(container_name)
        state.record_restart()
        metrics.record_restart(container_name)

        if success:
            state.consecutive_unhealthy = 0
            logger.info(f"Container {container_name} recovered")
        else:
            logger.error(f"Failed to restart container {container_name}")

    def run_check(self) -> None:
        """Run a health check cycle."""
        containers = self.get_monitored_containers()

        if not containers:
            logger.warning("No monitored containers found")
            return

        logger.info(f"Checking {len(containers)} containers")

        for name, status in containers.items():
            if shutdown_requested:
                break
            self.process_container(name, status)


def main():
    logger.info(
        f"Starting Watchdog (port={METRICS_PORT}, interval={CHECK_INTERVAL}s)"
    )
    logger.info(f"Monitoring containers: {MONITORED_CONTAINERS}")
    logger.info(
        f"Recovery config: max_attempts={MAX_RESTART_ATTEMPTS}, "
        f"window={RESTART_WINDOW_MINUTES}m, cooldown={COOLDOWN_MINUTES}m"
    )

    # Add /health endpoint to Prometheus server
    add_health_routes_to_prometheus(health_checker)

    # Start Prometheus metrics server (now also serves /health)
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server started on port {METRICS_PORT} (also serves /health)")

    # Create watchdog
    watchdog = Watchdog()

    # Connect to Docker
    if not watchdog.connect():
        logger.error("Failed to connect to Docker, exiting")
        sys.exit(1)

    # Add Docker connectivity check to health checker
    def check_docker_connected():
        if watchdog.client is None:
            return False
        try:
            watchdog.client.ping()
            return True
        except Exception:
            return False

    health_checker.add_custom_check("docker_connected", check_docker_connected)

    # Main monitoring loop
    consecutive_errors = 0
    max_consecutive_errors = 5

    while not shutdown_requested:
        try:
            start_time = time.time()

            watchdog.run_check()
            metrics.record_success()
            health_checker.record_collection()  # Update health checker
            consecutive_errors = 0

            duration = time.time() - start_time
            logger.info(f"Health check completed in {duration:.2f}s")

        except DockerException as e:
            logger.error(f"Docker error: {e}")
            consecutive_errors += 1
            metrics.record_error("docker")

            # Try to reconnect
            if not watchdog.connect():
                logger.error("Failed to reconnect to Docker")

        except Exception as e:
            logger.error(f"Watchdog error: {e}")
            consecutive_errors += 1
            metrics.record_error("unknown")

        # Mark unhealthy after too many errors
        if consecutive_errors >= max_consecutive_errors:
            metrics.set_unhealthy()
            logger.error(
                f"Watchdog unhealthy: {consecutive_errors} consecutive errors"
            )

        # Sleep until next check
        elapsed = time.time() - start_time
        sleep_time = max(0, CHECK_INTERVAL - elapsed)
        time.sleep(sleep_time)

    logger.info("Watchdog stopped")


if __name__ == "__main__":
    main()
