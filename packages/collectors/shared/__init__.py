"""
Shared modules for GePT data collectors.

This package provides common functionality for all collectors:
- config: Centralized configuration management
- metrics: Standardized Prometheus metrics
"""

from .config import CollectorConfig
from .metrics import CollectorMetrics

__all__ = ['CollectorConfig', 'CollectorMetrics']
