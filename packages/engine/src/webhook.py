"""Webhook utilities for secure communication with web app."""

import hashlib
import hmac
import time
from typing import Optional

from .config import config
from .logging_config import get_logger

logger = get_logger(__name__)


class WebhookSignatureError(Exception):
    """Raised when webhook signature verification fails."""

    pass


def verify_webhook_signature(
    body: str,
    timestamp: str,
    signature: str,
    secret: Optional[str] = None,
) -> bool:
    """Verify HMAC-SHA256 webhook signature.

    Args:
        body: Raw request body as string
        timestamp: Unix timestamp in milliseconds from X-Webhook-Timestamp header
        signature: HMAC-SHA256 signature from X-Webhook-Signature header
        secret: Webhook secret (defaults to config.webhook_secret)

    Returns:
        True if signature is valid

    Raises:
        WebhookSignatureError: If signature is invalid or timestamp is stale
    """
    if secret is None:
        secret = config.webhook_secret

    if not secret:
        raise WebhookSignatureError("Webhook secret not configured")

    # Validate timestamp is within tolerance
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        raise WebhookSignatureError("Invalid timestamp format")

    current_time_ms = int(time.time() * 1000)
    tolerance_ms = config.webhook_timestamp_tolerance_ms

    if abs(current_time_ms - ts) > tolerance_ms:
        raise WebhookSignatureError(
            f"Timestamp outside tolerance window ({tolerance_ms}ms)"
        )

    # Compute expected signature
    payload = f"{timestamp}.{body}"
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(signature.lower(), expected.lower()):
        raise WebhookSignatureError("Signature mismatch")

    return True


def generate_webhook_signature(
    body: str,
    secret: Optional[str] = None,
) -> tuple[str, str]:
    """Generate HMAC-SHA256 webhook signature for outgoing requests.

    Args:
        body: Request body as string
        secret: Webhook secret (defaults to config.webhook_secret)

    Returns:
        Tuple of (timestamp, signature)
    """
    if secret is None:
        secret = config.webhook_secret

    if not secret:
        raise WebhookSignatureError("Webhook secret not configured")

    timestamp = str(int(time.time() * 1000))
    payload = f"{timestamp}.{body}"
    signature = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()

    return timestamp, signature
