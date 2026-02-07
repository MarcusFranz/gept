"""API security and rate limiting helpers."""


from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter

import src.config as config_module
from .logging_config import get_logger

logger = get_logger(__name__)


def get_rate_limit_key(request: Request) -> str:
    """Get rate limit key based on user identifier or IP.

    Priority:
    1. User hash from X-User-Hash header (authenticated web users)
    2. User hash from user_id query param
    3. X-Forwarded-For header (proxied requests)
    4. Client IP (direct requests)
    """
    # Try user hash from header (web frontend with auth)
    user_hash = request.headers.get("X-User-Hash")
    if user_hash and len(user_hash) == 64:
        return f"user:{user_hash}"

    # Try user_id from query params
    user_id = request.query_params.get("user_id")
    if user_id and len(user_id) == 64:
        return f"user:{user_id}"

    # Only trust X-Forwarded-For from known proxies (localhost, Docker network)
    # Prevents rate limit bypass via spoofed headers from direct connections
    client_host = request.client.host if request.client else None
    trusted_proxies = {"127.0.0.1", "::1", "172.17.0.1"}
    if client_host in trusted_proxies:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
            return f"ip:{client_ip}"

    # Use direct connection IP (not spoofable)
    if client_host:
        return f"ip:{client_host}"

    return "ip:unknown"


limiter = Limiter(key_func=get_rate_limit_key)


# API Key Authentication
# Security scheme for OpenAPI documentation
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    x_api_key: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """Verify API key for protected endpoints.

    INTERNAL_API_KEY is required at startup (enforced by config.validate()).
    The bypass below is unreachable in production but kept for test ergonomics.

    Returns:
        The API key if valid, None if auth is not configured (unreachable in prod)

    Raises:
        HTTPException: 401 if API key is missing or invalid
    """
    config = config_module.config

    # Unreachable in production — config.validate() blocks startup without key.
    # Kept for test fixtures that don't go through the lifespan startup path.
    if not config.internal_api_key:
        return None

    # API key is required
    if not x_api_key:
        logger.warning("API request rejected: missing X-API-Key header")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate API key
    if x_api_key != config.internal_api_key:
        logger.warning("API request rejected: invalid API key")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key
