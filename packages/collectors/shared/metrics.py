"""
Standardized Prometheus Metrics for GePT Data Collectors

Provides consistent metric naming and structure across all collectors.

Usage:
    from shared.metrics import CollectorMetrics

    metrics = CollectorMetrics('5m')
    metrics.start_server(9100)

    # In collection loop
    with metrics.collection_duration.time():
        data = fetch_data()
        metrics.requests_total.labels(status='success').inc()
        metrics.items_collected.inc(len(data))
        metrics.last_collection.set_to_current_time()
"""

import time
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
        REGISTRY,
        CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class CollectorMetrics:
    """
    Standard metrics for data collectors.

    Creates a consistent set of metrics for any collector service:
    - requests_total: Counter of API requests by status
    - items_collected: Counter of items/records collected
    - collection_duration: Histogram of collection time
    - last_collection: Gauge of last successful collection timestamp
    - errors_total: Counter of errors by type
    """

    def __init__(self, service_name: str, registry: Optional['CollectorRegistry'] = None):
        """
        Initialize metrics for a collector service.

        Args:
            service_name: Short name for the service (e.g., '5m', 'hourly', 'news')
            registry: Optional custom Prometheus registry
        """
        self.service_name = service_name
        self.enabled = PROMETHEUS_AVAILABLE

        if not self.enabled:
            return

        reg = registry or REGISTRY

        # Request counter by status
        self.requests_total = Counter(
            f'gept_{service_name}_requests_total',
            f'Total API requests for {service_name} collector',
            ['status'],
            registry=reg
        )

        # Items/records collected counter
        self.items_collected = Counter(
            f'gept_{service_name}_items_total',
            f'Total items collected by {service_name} collector',
            registry=reg
        )

        # Collection duration histogram
        self.collection_duration = Histogram(
            f'gept_{service_name}_duration_seconds',
            f'Collection duration for {service_name} collector',
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=reg
        )

        # Last successful collection timestamp
        self.last_collection = Gauge(
            f'gept_{service_name}_last_timestamp',
            f'Timestamp of last successful {service_name} collection',
            registry=reg
        )

        # Error counter by type
        self.errors_total = Counter(
            f'gept_{service_name}_errors_total',
            f'Total errors in {service_name} collector',
            ['type'],
            registry=reg
        )

        # Health status gauge (1 = healthy, 0 = unhealthy)
        self.health_status = Gauge(
            f'gept_{service_name}_health',
            f'Health status of {service_name} collector (1=healthy, 0=unhealthy)',
            registry=reg
        )
        self.health_status.set(1)  # Start healthy

    def start_server(self, port: int) -> None:
        """
        Start the Prometheus metrics HTTP server.

        Args:
            port: Port number to listen on
        """
        if self.enabled:
            start_http_server(port)

    def record_success(self, items_count: int = 1) -> None:
        """Record a successful collection."""
        if not self.enabled:
            return
        self.requests_total.labels(status='success').inc()
        self.items_collected.inc(items_count)
        self.last_collection.set(time.time())
        self.health_status.set(1)

    def record_error(self, error_type: str = 'unknown') -> None:
        """Record an error."""
        if not self.enabled:
            return
        self.requests_total.labels(status='error').inc()
        self.errors_total.labels(type=error_type).inc()

    def set_unhealthy(self) -> None:
        """Mark the collector as unhealthy."""
        if self.enabled:
            self.health_status.set(0)

    def set_healthy(self) -> None:
        """Mark the collector as healthy."""
        if self.enabled:
            self.health_status.set(1)


class MonitorMetrics(CollectorMetrics):
    """
    Extended metrics for monitoring services (collector_monitor, gap_detector, watchdog).

    Adds data freshness and gap detection metrics on top of base CollectorMetrics.
    """

    def __init__(self, service_name: str, registry: Optional['CollectorRegistry'] = None):
        """
        Initialize metrics for a monitoring service.

        Args:
            service_name: Short name for the service (e.g., 'monitor', 'gap_detector', 'watchdog')
            registry: Optional custom Prometheus registry
        """
        super().__init__(service_name, registry)

        if not self.enabled:
            return

        reg = registry or REGISTRY

        # Data freshness metrics (for collector_monitor)
        self.data_age = Gauge(
            'gept_data_age_seconds',
            'Age of most recent data in seconds',
            ['table'],
            registry=reg
        )

        self.freshness_threshold = Gauge(
            'gept_data_freshness_threshold_seconds',
            'Freshness threshold per table in seconds',
            ['table'],
            registry=reg
        )

        self.freshness_ok = Gauge(
            'gept_data_freshness_ok',
            'Data freshness status (1=OK, 0=STALE)',
            ['table'],
            registry=reg
        )

        # Gap detection metrics (for gap_detector)
        self.gaps_detected = Counter(
            'gept_gaps_detected_total',
            'Total number of data gaps detected',
            ['table'],
            registry=reg
        )

        self.gaps_backfilled = Counter(
            'gept_gaps_backfilled_total',
            'Total number of gaps successfully backfilled',
            ['table'],
            registry=reg
        )

        self.gaps_unrecoverable = Gauge(
            'gept_gaps_unrecoverable_total',
            'Number of gaps that cannot be backfilled',
            ['table'],
            registry=reg
        )

        self.backfill_errors = Counter(
            'gept_gaps_backfill_errors_total',
            'Total number of backfill errors',
            ['table', 'error_type'],
            registry=reg
        )

        self.rows_recovered = Counter(
            'gept_gaps_rows_recovered_total',
            'Total number of rows recovered via backfill',
            ['table'],
            registry=reg
        )

        # Watchdog metrics (for watchdog)
        self.container_health = Gauge(
            'gept_watchdog_container_health',
            'Container health status (1=healthy, 0=unhealthy)',
            ['container'],
            registry=reg
        )

        self.restarts_total = Counter(
            'gept_watchdog_restarts_total',
            'Total number of container restarts triggered',
            ['container'],
            registry=reg
        )

        self.recovery_failed = Gauge(
            'gept_watchdog_recovery_failed',
            'Recovery failure status (1=failed, 0=ok)',
            ['container'],
            registry=reg
        )

    def record_data_age(self, table: str, age_seconds: float, threshold_seconds: float) -> None:
        """Record data freshness for a table."""
        if not self.enabled:
            return
        self.data_age.labels(table=table).set(age_seconds)
        self.freshness_threshold.labels(table=table).set(threshold_seconds)
        is_fresh = 1 if age_seconds <= threshold_seconds else 0
        self.freshness_ok.labels(table=table).set(is_fresh)

    def record_gap_detected(self, table: str) -> None:
        """Record a detected gap."""
        if not self.enabled:
            return
        self.gaps_detected.labels(table=table).inc()

    def record_gap_backfilled(self, table: str, rows_recovered: int) -> None:
        """Record a successful backfill."""
        if not self.enabled:
            return
        self.gaps_backfilled.labels(table=table).inc()
        self.rows_recovered.labels(table=table).inc(rows_recovered)

    def record_backfill_error(self, table: str, error_type: str) -> None:
        """Record a backfill error."""
        if not self.enabled:
            return
        self.backfill_errors.labels(table=table, error_type=error_type).inc()

    def set_unrecoverable_gaps(self, table: str, count: int) -> None:
        """Set the number of unrecoverable gaps for a table."""
        if not self.enabled:
            return
        self.gaps_unrecoverable.labels(table=table).set(count)

    def record_container_health(self, container: str, is_healthy: bool) -> None:
        """Record container health status."""
        if not self.enabled:
            return
        self.container_health.labels(container=container).set(1 if is_healthy else 0)

    def record_restart(self, container: str) -> None:
        """Record a container restart."""
        if not self.enabled:
            return
        self.restarts_total.labels(container=container).inc()

    def set_recovery_failed(self, container: str, failed: bool) -> None:
        """Set recovery failure status for a container."""
        if not self.enabled:
            return
        self.recovery_failed.labels(container=container).set(1 if failed else 0)


class DataQualityMetrics:
    """
    Data quality metrics for collectors.

    Tracks data validation errors, null values, duplicates, and other
    quality issues that may indicate slow data degradation.

    Usage:
        from shared.metrics import DataQualityMetrics

        dq_metrics = DataQualityMetrics('5m')

        # Track null values
        dq_metrics.record_null_value('avg_high_price')

        # Track duplicates
        dq_metrics.record_duplicates(5)

        # Track validation errors
        dq_metrics.record_validation_error('invalid_range')
    """

    def __init__(self, collector_name: str, registry: Optional['CollectorRegistry'] = None):
        """
        Initialize data quality metrics for a collector.

        Args:
            collector_name: Short name for the collector (e.g., '5m', 'hourly', 'news')
            registry: Optional custom Prometheus registry
        """
        self.collector_name = collector_name
        self.enabled = PROMETHEUS_AVAILABLE

        if not self.enabled:
            return

        reg = registry or REGISTRY

        # Validation errors by type (null_value, invalid_range, schema_mismatch)
        self.validation_errors = Counter(
            'gept_collector_validation_errors_total',
            'Number of data validation errors',
            ['collector', 'error_type'],
            registry=reg
        )

        # Duplicate records detected (from ON CONFLICT DO NOTHING)
        self.duplicate_records = Counter(
            'gept_collector_duplicate_records_total',
            'Number of duplicate records detected',
            ['collector'],
            registry=reg
        )

        # Null/missing values by field
        self.null_values = Counter(
            'gept_collector_null_values_total',
            'Number of null/missing values detected',
            ['collector', 'field'],
            registry=reg
        )

        # Per-item collection duration (for hourly collector)
        self.item_duration = Histogram(
            'gept_collector_item_duration_seconds',
            'Time to collect single item',
            ['collector'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=reg
        )

        # Database commit failures
        self.db_commit_failures = Counter(
            'gept_collector_db_commit_failures_total',
            'Database transaction commit failures',
            ['collector'],
            registry=reg
        )

        # API response validation failures
        self.api_validation_failures = Counter(
            'gept_collector_api_validation_failures_total',
            'API response validation failures',
            ['collector', 'reason'],
            registry=reg
        )

    def record_null_value(self, field: str) -> None:
        """Record a null/missing value for a field."""
        if not self.enabled:
            return
        self.null_values.labels(collector=self.collector_name, field=field).inc()

    def record_null_values(self, field: str, count: int) -> None:
        """Record multiple null/missing values for a field."""
        if not self.enabled or count <= 0:
            return
        self.null_values.labels(collector=self.collector_name, field=field).inc(count)

    def record_duplicates(self, count: int) -> None:
        """Record duplicate records detected."""
        if not self.enabled or count <= 0:
            return
        self.duplicate_records.labels(collector=self.collector_name).inc(count)

    def record_validation_error(self, error_type: str) -> None:
        """Record a data validation error."""
        if not self.enabled:
            return
        self.validation_errors.labels(
            collector=self.collector_name,
            error_type=error_type
        ).inc()

    def record_db_commit_failure(self) -> None:
        """Record a database commit failure."""
        if not self.enabled:
            return
        self.db_commit_failures.labels(collector=self.collector_name).inc()

    def record_api_validation_failure(self, reason: str) -> None:
        """Record an API response validation failure."""
        if not self.enabled:
            return
        self.api_validation_failures.labels(
            collector=self.collector_name,
            reason=reason
        ).inc()

    def observe_item_duration(self, duration_seconds: float) -> None:
        """Record the duration to collect a single item."""
        if not self.enabled:
            return
        self.item_duration.labels(collector=self.collector_name).observe(duration_seconds)

    def time_item(self):
        """Context manager to time item collection."""
        if not self.enabled:
            return _DummyTimer()
        return self.item_duration.labels(collector=self.collector_name).time()


class _DummyTimer:
    """Dummy timer context manager when Prometheus is not available."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class DummyMetrics:
    """
    Dummy metrics class when Prometheus is not available.

    Provides the same interface but does nothing.
    """

    def __init__(self, *args, **kwargs):
        pass

    def start_server(self, port: int) -> None:
        pass

    def record_success(self, items_count: int = 1) -> None:
        pass

    def record_error(self, error_type: str = 'unknown') -> None:
        pass

    def set_unhealthy(self) -> None:
        pass

    def set_healthy(self) -> None:
        pass


class DummyDataQualityMetrics:
    """
    Dummy data quality metrics class when Prometheus is not available.

    Provides the same interface but does nothing.
    """

    def __init__(self, *args, **kwargs):
        pass

    def record_null_value(self, field: str) -> None:
        pass

    def record_null_values(self, field: str, count: int) -> None:
        pass

    def record_duplicates(self, count: int) -> None:
        pass

    def record_validation_error(self, error_type: str) -> None:
        pass

    def record_db_commit_failure(self) -> None:
        pass

    def record_api_validation_failure(self, reason: str) -> None:
        pass

    def observe_item_duration(self, duration_seconds: float) -> None:
        pass

    def time_item(self):
        return _DummyTimer()


def get_metrics(service_name: str) -> CollectorMetrics:
    """
    Factory function to get metrics instance.

    Returns a real CollectorMetrics if Prometheus is available,
    otherwise returns a DummyMetrics that does nothing.
    """
    if PROMETHEUS_AVAILABLE:
        return CollectorMetrics(service_name)
    else:
        return DummyMetrics(service_name)


def get_data_quality_metrics(collector_name: str) -> DataQualityMetrics:
    """
    Factory function to get data quality metrics instance.

    Returns a real DataQualityMetrics if Prometheus is available,
    otherwise returns a DummyDataQualityMetrics that does nothing.
    """
    if PROMETHEUS_AVAILABLE:
        return DataQualityMetrics(collector_name)
    else:
        return DummyDataQualityMetrics(collector_name)
