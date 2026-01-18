"""
Database Utilities - Centralized Database Connection Management
================================================================

This module provides a unified interface for database connections across
the GePT codebase. All database access should use this module instead of
hardcoding connection parameters.

Usage:
    from src.db_utils import get_connection, release_connection, get_simple_connection

    # Pooled connection (recommended for short operations)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ...")
    finally:
        release_connection(conn)

    # Simple connection (for long-running operations)
    conn = get_simple_connection()
    try:
        # ... long operation ...
    finally:
        conn.close()
"""

import os
import psycopg2
from psycopg2 import pool
from typing import Optional, Dict, Any
from contextlib import contextmanager


def _get_required_env(key: str, default: Optional[str] = None) -> str:
    """Get environment variable, raising error if not set and no default."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable {key} is not set. "
            f"See .env.example for required variables."
        )
    return value


# Database configuration from environment variables
# DB_PASS is required - no default for security
DEFAULT_DB_CONFIG: Dict[str, Any] = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'dbname': os.getenv('DB_NAME', 'osrs_data'),
    'user': os.getenv('DB_USER', 'osrs_user'),
    'password': _get_required_env('DB_PASS')
}


# Connection pool singleton
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_db_config() -> Dict[str, Any]:
    """Get the current database configuration."""
    return DEFAULT_DB_CONFIG.copy()


def get_connection_pool(minconn: int = 2, maxconn: int = 10) -> pool.ThreadedConnectionPool:
    """Get or create the connection pool singleton."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn,
            maxconn,
            **DEFAULT_DB_CONFIG
        )
    return _connection_pool


def get_connection():
    """
    Get a database connection from the pool.

    Returns:
        A psycopg2 connection object from the pool.

    Note:
        Always call release_connection() when done.
    """
    return get_connection_pool().getconn()


def release_connection(conn) -> None:
    """
    Release a connection back to the pool.

    Args:
        conn: The connection to release.
    """
    if _connection_pool is not None:
        _connection_pool.putconn(conn)


def get_simple_connection():
    """
    Get a simple (non-pooled) database connection.

    Use this for long-running operations that shouldn't block the pool.

    Returns:
        A new psycopg2 connection object.

    Note:
        The caller is responsible for closing this connection.
    """
    return psycopg2.connect(**DEFAULT_DB_CONFIG)


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.

    Automatically handles getting and releasing connections from the pool.

    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    conn = get_connection()
    try:
        yield conn
    finally:
        release_connection(conn)


@contextmanager
def get_db_cursor(commit: bool = True):
    """
    Context manager for database cursors with automatic commit.

    Args:
        commit: If True, commits the transaction on successful completion.

    Usage:
        with get_db_cursor() as cur:
            cur.execute("INSERT INTO ...")
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            yield cur
            if commit:
                conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def close_pool() -> None:
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None


class ConnectionPool:
    """
    Legacy-compatible connection pool class.

    This class provides backwards compatibility with existing code that uses
    the ConnectionPool class pattern. New code should use the module-level
    functions instead.
    """
    _instance = None
    _pool = None

    @classmethod
    def get_pool(cls, minconn: int = 2, maxconn: int = 5) -> pool.ThreadedConnectionPool:
        """Get or create the connection pool."""
        if cls._pool is None:
            cls._pool = get_connection_pool(minconn, maxconn)
        return cls._pool

    @classmethod
    def get_conn(cls):
        """Get a connection from the pool."""
        return cls.get_pool().getconn()

    @classmethod
    def put_conn(cls, conn) -> None:
        """Return a connection to the pool."""
        cls.get_pool().putconn(conn)


# Convenience exports for backwards compatibility
CONN_PARAMS = DEFAULT_DB_CONFIG.copy()
# Legacy alias - use 'dbname' consistently but support 'database' for old code
CONN_PARAMS['database'] = CONN_PARAMS['dbname']
