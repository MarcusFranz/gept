"""Small helpers shared across API route modules."""


import hashlib
from typing import Optional

from .logging_config import get_logger

logger = get_logger(__name__)


def is_valid_sha256(value: str) -> bool:
    """Check if a string is a valid SHA256 hash (64 hex characters)."""
    if len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


def normalize_user_id(user_id: Optional[str]) -> Optional[str]:
    """Normalize user_id to ensure it's a valid SHA256 hash.

    If the provided user_id is already a valid SHA256 hash (64 hex chars),
    returns it as-is. Otherwise, hashes the value to create a valid SHA256.
    This ensures raw Discord IDs are never stored in the crowding tracker.
    """
    if not user_id:
        return None

    user_id = user_id.strip()
    if not user_id:
        return None

    # If already a valid SHA256 hash, return as-is (lowercase)
    if is_valid_sha256(user_id):
        return user_id.lower()

    # Otherwise, hash it server-side to protect privacy
    logger.warning(
        "Received non-hashed user_id, hashing server-side",
        user_id_length=len(user_id),
    )
    return hashlib.sha256(user_id.encode()).hexdigest()
