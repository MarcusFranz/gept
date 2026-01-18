"""
Centralized Configuration for GePT Data Collectors

All collectors should use this module for configuration to ensure
consistency and easy environment-based overrides.

Usage:
    from shared.config import CollectorConfig

    # Get database connection
    conn = psycopg2.connect(**CollectorConfig.get_db_params())

    # Access configuration values
    interval = CollectorConfig.INTERVAL_5M
"""

import os
from typing import Dict, Any, Optional


def _require_env(key: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable {key} is not set. "
            f"See .env.example for required variables."
        )
    return value


class CollectorConfig:
    """Configuration for data collectors."""

    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_NAME: str = os.getenv('DB_NAME', 'osrs_data')
    DB_USER: str = os.getenv('DB_USER', 'osrs_user')
    DB_PASS: str = _require_env('DB_PASS')

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    USER_AGENT: str = os.getenv(
        'USER_AGENT',
        'GePT-Collector/3.0 (https://github.com/gept) Contact: @marcusfranz'
    )

    # API Endpoints
    WIKI_API_BASE: str = "https://prices.runescape.wiki/api/v1/osrs"
    OSRS_HOMEPAGE: str = "https://oldschool.runescape.com/"

    # ==========================================================================
    # Collection Intervals (seconds)
    # ==========================================================================
    INTERVAL_1M: int = int(os.getenv('COLLECTION_INTERVAL_1M', '60'))
    INTERVAL_5M: int = int(os.getenv('COLLECTION_INTERVAL_5M', '300'))
    INTERVAL_HOURLY: int = int(os.getenv('COLLECTION_INTERVAL_HOURLY', '3600'))
    INTERVAL_DAILY: int = int(os.getenv('COLLECTION_INTERVAL_DAILY', '86400'))
    INTERVAL_NEWS: int = int(os.getenv('COLLECTION_INTERVAL_NEWS', '1800'))

    # ==========================================================================
    # Metrics Ports
    # ==========================================================================
    METRICS_PORT_5M: int = int(os.getenv('METRICS_PORT_5M', '9100'))
    METRICS_PORT_HOURLY: int = int(os.getenv('METRICS_PORT_HOURLY', '9101'))
    METRICS_PORT_NEWS: int = int(os.getenv('METRICS_PORT_NEWS', '9102'))
    METRICS_PORT_1M: int = int(os.getenv('METRICS_PORT_1M', '9103'))
    METRICS_PORT_PLAYER_COUNT: int = int(os.getenv('METRICS_PORT_PLAYER_COUNT', '9104'))
    METRICS_PORT_ITEMS: int = int(os.getenv('METRICS_PORT_ITEMS', '9105'))

    # ==========================================================================
    # Data Paths
    # ==========================================================================
    DATA_DIR: str = os.getenv('DATA_DIR', '/data')

    @classmethod
    def get_db_params(cls) -> Dict[str, Any]:
        """Get database connection parameters as a dictionary."""
        return {
            'host': cls.DB_HOST,
            'port': cls.DB_PORT,
            'dbname': cls.DB_NAME,
            'user': cls.DB_USER,
            'password': cls.DB_PASS
        }

    @classmethod
    def get_api_headers(cls) -> Dict[str, str]:
        """Get standard headers for API requests."""
        return {
            'User-Agent': cls.USER_AGENT,
            'Accept': 'application/json'
        }

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (for debugging)."""
        print("=== Collector Configuration ===")
        print(f"  DB_HOST: {cls.DB_HOST}")
        print(f"  DB_PORT: {cls.DB_PORT}")
        print(f"  DB_NAME: {cls.DB_NAME}")
        print(f"  DB_USER: {cls.DB_USER}")
        print(f"  DB_PASS: {'*' * len(cls.DB_PASS)}")
        print(f"  USER_AGENT: {cls.USER_AGENT}")
        print(f"  DATA_DIR: {cls.DATA_DIR}")
        print("================================")


# Convenience function for getting connection params
def get_db_params() -> Dict[str, Any]:
    """Shortcut to get database parameters."""
    return CollectorConfig.get_db_params()
