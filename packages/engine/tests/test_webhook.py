"""Tests for webhook utilities."""

import hashlib
import hmac
import time

import pytest

from src.webhook import (
    WebhookSignatureError,
    generate_webhook_signature,
    verify_webhook_signature,
)


class TestWebhookSignature:
    """Tests for webhook signature verification."""

    def test_verify_valid_signature(self):
        """Test that valid signatures are accepted."""
        secret = "test_secret_key"
        body = '{"eventType": "TRADE_CREATED"}'
        timestamp = str(int(time.time() * 1000))

        # Generate signature
        payload = f"{timestamp}.{body}"
        signature = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        # Verify should pass
        result = verify_webhook_signature(body, timestamp, signature, secret)
        assert result is True

    def test_reject_invalid_signature(self):
        """Test that invalid signatures are rejected."""
        secret = "test_secret_key"
        body = '{"eventType": "TRADE_CREATED"}'
        timestamp = str(int(time.time() * 1000))

        with pytest.raises(WebhookSignatureError, match="Signature mismatch"):
            verify_webhook_signature(body, timestamp, "invalid_signature", secret)

    def test_reject_stale_timestamp(self):
        """Test that stale timestamps are rejected."""
        secret = "test_secret_key"
        body = '{"eventType": "TRADE_CREATED"}'
        # Timestamp from 10 minutes ago
        timestamp = str(int((time.time() - 600) * 1000))

        payload = f"{timestamp}.{body}"
        signature = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        with pytest.raises(WebhookSignatureError, match="Timestamp outside tolerance"):
            verify_webhook_signature(body, timestamp, signature, secret)

    def test_reject_invalid_timestamp_format(self):
        """Test that invalid timestamp formats are rejected."""
        secret = "test_secret_key"
        body = '{"eventType": "TRADE_CREATED"}'

        with pytest.raises(WebhookSignatureError, match="Invalid timestamp format"):
            verify_webhook_signature(body, "not_a_number", "signature", secret)

    def test_reject_missing_secret(self):
        """Test that missing secret raises an error."""
        body = '{"eventType": "TRADE_CREATED"}'
        timestamp = str(int(time.time() * 1000))

        with pytest.raises(
            WebhookSignatureError, match="Webhook secret not configured"
        ):
            verify_webhook_signature(body, timestamp, "signature", "")


class TestGenerateSignature:
    """Tests for webhook signature generation."""

    def test_generate_signature(self):
        """Test signature generation."""
        secret = "test_secret_key"
        body = '{"eventType": "TRADE_CREATED"}'

        timestamp, signature = generate_webhook_signature(body, secret)

        # Verify the generated signature is valid
        assert len(signature) == 64  # SHA256 hex string
        assert timestamp.isdigit()

        # Verify it can be validated
        result = verify_webhook_signature(body, timestamp, signature, secret)
        assert result is True

    def test_generate_signature_uses_config_secret(self):
        """Test signature generation falls back to config when secret is None."""
        body = '{"eventType": "TRADE_CREATED"}'
        config_secret = "config_secret_key"

        import src.config as config_module

        original = config_module.config.webhook_secret
        config_module.config.webhook_secret = config_secret
        try:
            timestamp, signature = generate_webhook_signature(body)
            result = verify_webhook_signature(body, timestamp, signature)
            assert result is True
        finally:
            config_module.config.webhook_secret = original

    def test_generate_signature_no_secret(self):
        """Test that missing secret raises an error."""
        body = '{"eventType": "TRADE_CREATED"}'

        with pytest.raises(
            WebhookSignatureError, match="Webhook secret not configured"
        ):
            generate_webhook_signature(body, "")


class TestRoundTrip:
    """Tests for full signature round-trip."""

    def test_signature_round_trip(self):
        """Test generating and verifying a signature."""
        secret = "my_secure_webhook_secret_12345"
        body = '{"eventType":"TRADE_CREATED","userId":"user123","tradeId":"trade456"}'

        # Generate
        timestamp, signature = generate_webhook_signature(body, secret)

        # Verify
        result = verify_webhook_signature(body, timestamp, signature, secret)
        assert result is True

    def test_modified_body_fails(self):
        """Test that modifying the body invalidates the signature."""
        secret = "my_secure_webhook_secret_12345"
        original_body = '{"eventType":"TRADE_CREATED"}'
        modified_body = '{"eventType":"TRADE_COMPLETED"}'

        timestamp, signature = generate_webhook_signature(original_body, secret)

        with pytest.raises(WebhookSignatureError, match="Signature mismatch"):
            verify_webhook_signature(modified_body, timestamp, signature, secret)
