"""Configuration management for the recommendation engine."""

from dataclasses import dataclass, field
from os import environ


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Database - connects to Ampere server's predictions table (osrs_data)
    # Requires DB_CONNECTION_STRING env var (no default for security)
    db_connection_string: str = field(
        default_factory=lambda: environ.get("DB_CONNECTION_STRING", "")
    )
    db_pool_size: int = field(
        default_factory=lambda: int(environ.get("DB_POOL_SIZE", "5"))
    )

    # Outcome Database - connects to Ampere server's gept_bot database
    # Used for storing trade outcomes for ML feedback loop
    # Optional: if not set, trade outcome recording is disabled
    outcome_db_connection_string: str = field(
        default_factory=lambda: environ.get("OUTCOME_DB_CONNECTION_STRING", "")
    )
    outcome_db_pool_size: int = field(
        default_factory=lambda: int(environ.get("OUTCOME_DB_POOL_SIZE", "5"))
    )

    # Active Trades Database (optional) - connects to the web app database (Neon/Postgres)
    # If set, the engine will load active trades directly on startup to seed the in-memory
    # trade monitor, avoiding dependency on web resync endpoints (which may be blocked by WAF/CDN).
    active_trades_db_connection_string: str = field(
        default_factory=lambda: environ.get("ACTIVE_TRADES_DB_CONNECTION_STRING", "")
    )

    # Thresholds
    min_ev_threshold: float = field(
        default_factory=lambda: float(environ.get("MIN_EV_THRESHOLD", "0.005"))
    )
    data_stale_seconds: int = field(
        default_factory=lambda: int(environ.get("DATA_STALE_SECONDS", "600"))
    )
    prediction_stale_seconds: int = field(
        default_factory=lambda: int(environ.get("PREDICTION_STALE_SECONDS", "300"))
    )

    # Minimum 24-hour volume threshold for item recommendations
    # Volume = total quantity traded (not number of trades)
    # Items with volume below this are excluded (prevents illiquid item recommendations)
    # Set to 0 to disable volume filtering
    # Default 1000 filters out extremely low-volume items (single-digit daily trades)
    min_volume_24h: int = field(
        default_factory=lambda: int(environ.get("MIN_VOLUME_24H", "1000"))
    )

    # Manipulation detection thresholds
    # Items with spread > max_spread_pct are flagged as potentially manipulated
    # Default 10% filters volatile/illiquid items like "Oak longbow (u)"
    max_spread_pct: float = field(
        default_factory=lambda: float(environ.get("MAX_SPREAD_PCT", "0.10"))
    )
    # Items where 1h volume > max_volume_concentration of 24h volume are flagged
    # Indicates possible pump/manipulation activity (volume spike in recent hour)
    # Default 80% flags items where most daily volume is in last hour
    max_volume_concentration: float = field(
        default_factory=lambda: float(environ.get("MAX_VOLUME_CONCENTRATION", "0.80"))
    )
    # Higher volume threshold for cheap items (< 1000gp)
    # Low-value items are more prone to manipulation and need higher liquidity
    # Default 10000 requires cheap items to have significant daily volume
    min_volume_for_low_value: int = field(
        default_factory=lambda: int(environ.get("MIN_VOLUME_FOR_LOW_VALUE", "10000"))
    )
    # Maximum ratio of buy_limit to daily volume (anti-manipulation)
    # If buy_limit / volume_24h > this ratio, item is filtered out
    # E.g., 0.05 means if buy_limit is >5% of daily volume, too illiquid/manipulable
    # Default 0.05 filters items where filling your buy limit would significantly impact the market
    max_buy_limit_volume_ratio: float = field(
        default_factory=lambda: float(environ.get("MAX_BUY_LIMIT_VOLUME_RATIO", "0.05"))
    )

    # API Response Cache (Redis)
    # TTL values in seconds for different cache types
    cache_ttl_recommendations: int = field(
        default_factory=lambda: int(environ.get("CACHE_TTL_RECOMMENDATIONS", "30"))
    )
    cache_ttl_items: int = field(
        default_factory=lambda: int(environ.get("CACHE_TTL_ITEMS", "60"))
    )
    cache_ttl_search: int = field(
        default_factory=lambda: int(environ.get("CACHE_TTL_SEARCH", "300"))
    )
    cache_ttl_prices: int = field(
        default_factory=lambda: int(environ.get("CACHE_TTL_PRICES", "30"))
    )
    cache_ttl_stats: int = field(
        default_factory=lambda: int(environ.get("CACHE_TTL_STATS", "300"))
    )
    # Enable/disable caching (for debugging or development)
    cache_enabled: bool = field(
        default_factory=lambda: environ.get("CACHE_ENABLED", "true").lower() == "true"
    )

    # Discord (optional)
    discord_webhook_url: str = field(
        default_factory=lambda: environ.get("DISCORD_WEBHOOK_URL", "")
    )

    # Redis for crowding tracking (optional)
    # If not set, uses in-memory tracker (data lost on restart, per-instance)
    # Format: redis://[:password@]host:port/db (e.g., redis://localhost:6379/0)
    redis_url: str = field(default_factory=lambda: environ.get("REDIS_URL", ""))
    # Whether to fall back to in-memory tracker if Redis is unavailable
    redis_fallback_to_memory: bool = field(
        default_factory=lambda: environ.get("REDIS_FALLBACK_TO_MEMORY", "true").lower()
        == "true"
    )

    # OSRS Wiki API (for buy limit fallback)
    # Used when database doesn't have buy limit data for an item
    wiki_api_enabled: bool = field(
        default_factory=lambda: environ.get("WIKI_API_ENABLED", "true").lower()
        == "true"
    )
    wiki_api_cache_ttl: int = field(
        default_factory=lambda: int(environ.get("WIKI_API_CACHE_TTL", "3600"))
    )
    wiki_api_user_agent: str = field(
        default_factory=lambda: environ.get(
            "WIKI_API_USER_AGENT",
            "GePT-Recommendation-Engine/1.0 (https://github.com/MarcusFranz/gept-recommendation-engine)",
        )
    )

    # API
    api_host: str = field(default_factory=lambda: environ.get("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(environ.get("API_PORT", "8000")))

    # API Authentication
    # Internal API key for server-to-server authentication (Discord bot, web frontend)
    # If not set, authentication is disabled (for development/testing)
    # Generate a secure key: python -c "import secrets; print(secrets.token_urlsafe(32))"
    internal_api_key: str = field(
        default_factory=lambda: environ.get("INTERNAL_API_KEY", "")
    )

    # CORS - configurable allowed origins
    # Default is localhost only for security (server-to-server API)
    # Set CORS_ORIGINS to comma-separated list of allowed origins
    # For production, set to:
    # CORS_ORIGINS=https://gept.gg,https://www.gept.gg,http://localhost:3000,http://localhost:8080
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in environ.get(
                "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
            ).split(",")
            if o.strip()
        ]
    )

    # CORS regex pattern for Vercel preview deployments
    # Set CORS_ORIGIN_REGEX to allow pattern-matched origins
    # Example: https://gept-gg-.*\.vercel\.app
    cors_origin_regex: str = field(
        default_factory=lambda: environ.get("CORS_ORIGIN_REGEX", "")
    )

    # Rate limiting
    # Format: "requests/period" e.g. "60/minute", "10/second;120/minute" (burst + sustained)
    # Higher limits for web frontend traffic, lower for unauthenticated
    rate_limit_recommendations: str = field(
        default_factory=lambda: environ.get("RATE_LIMIT_RECOMMENDATIONS", "60/minute")
    )
    rate_limit_search: str = field(
        default_factory=lambda: environ.get("RATE_LIMIT_SEARCH", "10/second;200/minute")
    )
    rate_limit_items: str = field(
        default_factory=lambda: environ.get("RATE_LIMIT_ITEMS", "120/minute")
    )
    rate_limit_health: str = field(
        default_factory=lambda: environ.get("RATE_LIMIT_HEALTH", "60/minute")
    )
    rate_limit_trade_updates: str = field(
        default_factory=lambda: environ.get("RATE_LIMIT_TRADE_UPDATES", "120/minute")
    )
    rate_limit_outcomes: str = field(
        default_factory=lambda: environ.get("RATE_LIMIT_OUTCOMES", "30/minute")
    )

    # Model selection - filter predictions by model_id
    # Set to a specific model_id (e.g., "patchtst_patchtst") to only serve that model's predictions
    # Leave empty to serve predictions from any model (default, backwards compatible)
    preferred_model_id: str = field(
        default_factory=lambda: environ.get("PREFERRED_MODEL_ID", "")
    )

    # Beta model ID - shadow/experimental model for opt-in users
    # Set to a specific model_id to enable the beta model toggle
    # Leave empty to disable the beta model feature
    beta_model_id: str = field(
        default_factory=lambda: environ.get("BETA_MODEL_ID", "")
    )

    # Item blocklist - items that should never appear in recommendations
    # These are excluded at the engine level from all endpoints
    blocked_item_ids: set[int] = field(default_factory=lambda: {
        13190,  # Old school bond - requires conversion fee, not actually profitable
    })

    # Logging configuration
    # LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    # LOG_FORMAT: json (for production), text (for local development)
    log_level: str = field(
        default_factory=lambda: environ.get("LOG_LEVEL", "INFO").upper()
    )
    log_format: str = field(
        default_factory=lambda: environ.get("LOG_FORMAT", "json").lower()
    )

    # Price buffer configuration
    # Applies a random buffer to buy/sell prices to reduce competition on exact prices
    # Buffer is a random percentage of the margin (sell_price - buy_price)
    # Buy price moves UP (toward market), sell price moves DOWN (toward market)
    # If 1% of margin < 1gp, no buffer is applied (prevents sub-gp adjustments)
    price_buffer_enabled: bool = field(
        default_factory=lambda: environ.get("PRICE_BUFFER_ENABLED", "true").lower()
        == "true"
    )
    price_buffer_min_pct: float = field(
        default_factory=lambda: float(environ.get("PRICE_BUFFER_MIN_PCT", "1.0"))
    )
    price_buffer_max_pct: float = field(
        default_factory=lambda: float(environ.get("PRICE_BUFFER_MAX_PCT", "4.0"))
    )

    # Price drop monitor thresholds
    price_drop_monitor_enabled: bool = field(
        default_factory=lambda: environ.get("PRICE_DROP_MONITOR_ENABLED", "true").lower()
        == "true"
    )
    price_drop_monitor_interval: int = field(
        default_factory=lambda: int(environ.get("PRICE_DROP_MONITOR_INTERVAL", "300"))
    )
    price_drop_min_pct: float = field(
        default_factory=lambda: float(environ.get("PRICE_DROP_MIN_PCT", "0.02"))
    )
    price_drop_medium_pct: float = field(
        default_factory=lambda: float(environ.get("PRICE_DROP_MEDIUM_PCT", "0.05"))
    )
    price_drop_high_pct: float = field(
        default_factory=lambda: float(environ.get("PRICE_DROP_HIGH_PCT", "0.10"))
    )
    price_drop_cooldown_low: int = field(
        default_factory=lambda: int(environ.get("PRICE_DROP_COOLDOWN_LOW", "1800"))
    )
    price_drop_cooldown_medium: int = field(
        default_factory=lambda: int(environ.get("PRICE_DROP_COOLDOWN_MEDIUM", "900"))
    )
    price_drop_cooldown_high: int = field(
        default_factory=lambda: int(environ.get("PRICE_DROP_COOLDOWN_HIGH", "600"))
    )

    # Webhook configuration for web app integration
    # Shared secret for HMAC-SHA256 signature verification
    # Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"
    trade_webhooks_enabled: bool = field(
        default_factory=lambda: environ.get("TRADE_WEBHOOKS_ENABLED", "true").lower()
        == "true"
    )
    webhook_secret: str = field(
        default_factory=lambda: environ.get("WEBHOOK_SECRET", "")
    )
    # Timestamp tolerance for webhook signature verification (milliseconds)
    webhook_timestamp_tolerance_ms: int = field(
        default_factory=lambda: int(
            environ.get("WEBHOOK_TIMESTAMP_TOLERANCE_MS", "300000")
        )
    )
    # Web app webhook URL for sending alerts
    web_app_webhook_url: str = field(
        default_factory=lambda: environ.get("WEB_APP_WEBHOOK_URL", "")
    )
    # Web app endpoint for resyncing active trades after engine restart
    web_app_resync_url: str = field(
        default_factory=lambda: environ.get("WEB_APP_RESYNC_URL", "")
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.db_connection_string:
            errors.append("DB_CONNECTION_STRING is required")
        if not self.internal_api_key:
            errors.append(
                "INTERNAL_API_KEY is required. "
                "Generate one: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        if self.trade_webhooks_enabled or self.price_drop_monitor_enabled:
            if not self.webhook_secret:
                errors.append(
                    "WEBHOOK_SECRET is required when TRADE_WEBHOOKS_ENABLED=true or PRICE_DROP_MONITOR_ENABLED=true"
                )
        if self.price_drop_monitor_enabled:
            if not self.web_app_webhook_url:
                errors.append(
                    "WEB_APP_WEBHOOK_URL is required when PRICE_DROP_MONITOR_ENABLED=true"
                )
        return errors


# Global config instance
config = Config()
