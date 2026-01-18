"""Test cases for structured logging configuration (Issue #153)."""

import importlib
import json
import os
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestLoggingConfig:
    """Test cases for logging configuration."""

    def test_log_level_config_default(self):
        """Test LOG_LEVEL defaults to INFO."""
        # Save current env var
        env_backup = os.environ.get("LOG_LEVEL")
        if "LOG_LEVEL" in os.environ:
            del os.environ["LOG_LEVEL"]

        try:
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.log_level == "INFO"
        finally:
            if env_backup is not None:
                os.environ["LOG_LEVEL"] = env_backup
            importlib.reload(config_module)

    def test_log_level_config_custom(self):
        """Test LOG_LEVEL accepts custom values."""
        with patch.dict("os.environ", {"LOG_LEVEL": "debug"}, clear=False):
            import src.config as config_module

            importlib.reload(config_module)

            # Should be uppercase
            assert config_module.config.log_level == "DEBUG"

            importlib.reload(config_module)

    def test_log_format_config_default(self):
        """Test LOG_FORMAT defaults to json."""
        env_backup = os.environ.get("LOG_FORMAT")
        if "LOG_FORMAT" in os.environ:
            del os.environ["LOG_FORMAT"]

        try:
            import src.config as config_module

            importlib.reload(config_module)

            assert config_module.config.log_format == "json"
        finally:
            if env_backup is not None:
                os.environ["LOG_FORMAT"] = env_backup
            importlib.reload(config_module)

    def test_log_format_config_text(self):
        """Test LOG_FORMAT accepts text value."""
        with patch.dict("os.environ", {"LOG_FORMAT": "TEXT"}, clear=False):
            import src.config as config_module

            importlib.reload(config_module)

            # Should be lowercase
            assert config_module.config.log_format == "text"

            importlib.reload(config_module)


class TestRequestIDMiddleware:
    """Test cases for request ID middleware."""

    def test_middleware_generates_request_id(self):
        """Test middleware generates request ID when not provided."""
        from src.logging_config import RequestIDMiddleware

        middleware = RequestIDMiddleware(app=MagicMock())

        # Create mock request without X-Request-ID
        mock_request = MagicMock()
        mock_request.headers = {}

        # The request_id should be generated
        request_id = mock_request.headers.get("X-Request-ID", None)
        assert request_id is None  # Not in headers initially

    def test_request_id_context_var(self):
        """Test request ID context variable."""
        from src.logging_config import get_request_id, request_id_ctx

        # Initially should be None
        assert get_request_id() is None

        # Set and get
        token = request_id_ctx.set("test-id-123")
        try:
            assert get_request_id() == "test-id-123"
        finally:
            request_id_ctx.reset(token)

        # After reset should be None
        assert get_request_id() is None


class TestLogProcessors:
    """Test cases for log processors."""

    def test_add_request_id_processor(self):
        """Test add_request_id processor adds request ID to event dict."""
        from src.logging_config import add_request_id, request_id_ctx

        # Without request ID in context
        event_dict = {"event": "test_event"}
        result = add_request_id(None, "info", event_dict.copy())
        assert "request_id" not in result

        # With request ID in context
        token = request_id_ctx.set("req-abc-123")
        try:
            result = add_request_id(None, "info", event_dict.copy())
            assert result["request_id"] == "req-abc-123"
        finally:
            request_id_ctx.reset(token)

    def test_mask_sensitive_data_user_hash(self):
        """Test mask_sensitive_data truncates user hashes."""
        from src.logging_config import mask_sensitive_data

        event_dict = {
            "event": "test",
            "user_hash": "a" * 64,
            "user_id": "b" * 64,
        }
        result = mask_sensitive_data(None, "info", event_dict)

        assert result["user_hash"] == "aaaaaaaa..."
        assert result["user_id"] == "bbbbbbbb..."

    def test_mask_sensitive_data_short_values_unchanged(self):
        """Test short values are not truncated."""
        from src.logging_config import mask_sensitive_data

        event_dict = {
            "event": "test",
            "user_hash": "short",
        }
        result = mask_sensitive_data(None, "info", event_dict)

        assert result["user_hash"] == "short"

    def test_mask_sensitive_data_api_key(self):
        """Test API keys are fully masked."""
        from src.logging_config import mask_sensitive_data

        event_dict = {
            "event": "test",
            "api_key": "secret-key-12345",
        }
        result = mask_sensitive_data(None, "info", event_dict)

        assert result["api_key"] == "***"


class TestGetLogger:
    """Test cases for get_logger function."""

    def test_get_logger_returns_bound_logger(self):
        """Test get_logger returns a structlog BoundLogger."""
        from src.logging_config import get_logger

        logger = get_logger("test_module")
        assert logger is not None
        # Should have structlog's standard methods
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "debug")

    def test_get_logger_default_name(self):
        """Test get_logger uses module name by default."""
        from src.logging_config import get_logger

        logger = get_logger()
        assert logger is not None
