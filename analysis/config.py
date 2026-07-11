"""Centralised, environment-driven configuration.

All tunable knobs (cache TTLs, timeouts, retry policy, DB paths, feature flags)
live here instead of being scattered as magic constants across modules. Every
value can be overridden with an environment variable, which keeps the app
12-factor friendly and makes tests deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_dotenv() -> None:
    """Load ``KEY=VALUE`` pairs from a git-ignored ``.env`` at the repo root.

    Dependency-free and non-destructive: real environment variables always win,
    so this only fills values that aren't already set. Runs before ``Settings``
    is instantiated so all keys are picked up.
    """
    path = os.environ.get("DOTENV_PATH", os.path.join(_BASE_DIR, ".env"))
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # --- Paths --------------------------------------------------------------
    base_dir: str = _BASE_DIR
    data_dir: str = _DATA_DIR
    db_path: str = field(default_factory=lambda: os.environ.get(
        "DB_PATH", os.path.join(_DATA_DIR, "history.db")))
    model_dir: str = field(default_factory=lambda: os.environ.get(
        "MODEL_DIR", os.path.join(_DATA_DIR, "models")))

    # --- Flask / server -----------------------------------------------------
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 5000))
    secret_key: str | None = field(default_factory=lambda: os.environ.get("SECRET_KEY"))

    # --- Security -----------------------------------------------------------
    # Comma-separated list of allowed CORS origins (empty => same-origin only).
    cors_origins: tuple = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()))
    session_cookie_secure: bool = field(default_factory=lambda: _env_bool("SESSION_COOKIE_SECURE", False))
    login_max_attempts: int = field(default_factory=lambda: _env_int("LOGIN_MAX_ATTEMPTS", 8))
    login_window_seconds: int = field(default_factory=lambda: _env_int("LOGIN_WINDOW_SECONDS", 300))
    login_lockout_seconds: int = field(default_factory=lambda: _env_int("LOGIN_LOCKOUT_SECONDS", 300))
    api_rate_limit: int = field(default_factory=lambda: _env_int("API_RATE_LIMIT", 120))
    api_rate_window: int = field(default_factory=lambda: _env_int("API_RATE_WINDOW", 60))

    # --- Cache --------------------------------------------------------------
    cache_ttl_daily: int = field(default_factory=lambda: _env_int("CACHE_TTL_DAILY", 60 * 15))
    cache_ttl_intraday: int = field(default_factory=lambda: _env_int("CACHE_TTL_INTRADAY", 60 * 5))
    cache_ttl_info: int = field(default_factory=lambda: _env_int("CACHE_TTL_INFO", 60 * 60))
    redis_url: str | None = field(default_factory=lambda: os.environ.get("REDIS_URL"))

    # --- Data providers -----------------------------------------------------
    fetch_retries: int = field(default_factory=lambda: _env_int("FETCH_RETRIES", 3))
    fetch_backoff: float = field(default_factory=lambda: _env_float("FETCH_BACKOFF", 0.6))
    fetch_timeout: int = field(default_factory=lambda: _env_int("FETCH_TIMEOUT", 20))
    breaker_fail_threshold: int = field(default_factory=lambda: _env_int("BREAKER_FAIL_THRESHOLD", 5))
    breaker_reset_seconds: int = field(default_factory=lambda: _env_int("BREAKER_RESET_SECONDS", 60))
    stale_after_days: int = field(default_factory=lambda: _env_int("STALE_AFTER_DAYS", 3))
    # Optional extra price providers (used as fallbacks only when a key is set).
    twelvedata_api_key: str | None = field(default_factory=lambda: os.environ.get("TWELVEDATA_API_KEY"))
    alphavantage_api_key: str | None = field(default_factory=lambda: os.environ.get("ALPHAVANTAGE_API_KEY"))
    # Unofficial NSE India API (bennythadikaran/NseIndiaApi): direct-from-source
    # equity history + India VIX. Key-less but scrapes NSE, so it's opt-out and
    # degrades gracefully (NSE blocks many non-Indian / cloud IPs).
    use_nse_api: bool = field(default_factory=lambda: _env_bool("USE_NSE_API", True))
    nse_server_mode: bool = field(default_factory=lambda: _env_bool("NSE_SERVER_MODE", False))

    # --- ML -----------------------------------------------------------------
    ml_cache_ttl: int = field(default_factory=lambda: _env_int("ML_CACHE_TTL", 60 * 15))
    # Preferred amount of clean history for a full-confidence model.
    ml_min_rows: int = field(default_factory=lambda: _env_int("ML_MIN_ROWS", 300))
    # Hard floor: below this many usable rows we genuinely can't train. Between
    # the floor and ml_min_rows we still predict, but flag reduced data quality.
    ml_abs_min_rows: int = field(default_factory=lambda: _env_int("ML_ABS_MIN_ROWS", 90))
    ml_persist: bool = field(default_factory=lambda: _env_bool("ML_PERSIST", True))
    ml_model_ttl_hours: int = field(default_factory=lambda: _env_int("ML_MODEL_TTL_HOURS", 24))

    # --- Local data warehouse / auto-download -------------------------------
    # Keep the SQLite archive fresh so models train on the deepest local data.
    auto_download: bool = field(default_factory=lambda: _env_bool("AUTO_DOWNLOAD", True))
    # Hours between background incremental refreshes (0 = one-shot on startup).
    data_refresh_hours: int = field(default_factory=lambda: _env_int("DATA_REFRESH_HOURS", 24))
    # Bound the initial backfill (most-liquid symbols first) so first run isn't
    # a multi-hour download. Set 0 to backfill the entire universe.
    auto_download_limit: int = field(default_factory=lambda: _env_int("AUTO_DOWNLOAD_LIMIT", 300))
    # Years of daily history to keep per symbol.
    history_years: int = field(default_factory=lambda: _env_int("HISTORY_YEARS", 10))
    # Extend short local history for a queried NSE stock from the NSE API.
    nse_augment: bool = field(default_factory=lambda: _env_bool("NSE_AUGMENT", True))
    # Feed broad-market index context + global sentiment into every stock model.
    use_market_features: bool = field(default_factory=lambda: _env_bool("USE_MARKET_FEATURES", True))
    # Include India VIX (volatility / "fear" gauge) in the market context when
    # the NSE API is reachable. Falls back silently when unavailable.
    use_vix_feature: bool = field(default_factory=lambda: _env_bool("USE_VIX_FEATURE", True))

    # --- News / sentiment ---------------------------------------------------
    news_enabled: bool = field(default_factory=lambda: _env_bool("NEWS_ENABLED", True))
    # Comma-separated provider order; only those with a key are actually used.
    news_providers: tuple = field(default_factory=lambda: tuple(
        p.strip().lower() for p in os.environ.get(
            "NEWS_PROVIDERS", "finnhub,newsapi_ai,newsdata,gnews").split(",") if p.strip()))
    finnhub_api_key: str | None = field(default_factory=lambda: os.environ.get("FINNHUB_API_KEY"))
    gnews_api_key: str | None = field(default_factory=lambda: os.environ.get("GNEWS_API_KEY"))
    newsdata_api_key: str | None = field(default_factory=lambda: os.environ.get("NEWSDATA_API_KEY"))
    newsapi_ai_key: str | None = field(default_factory=lambda: os.environ.get("NEWSAPI_AI_KEY"))
    news_lookback_days: int = field(default_factory=lambda: _env_int("NEWS_LOOKBACK_DAYS", 30))
    news_max_articles: int = field(default_factory=lambda: _env_int("NEWS_MAX_ARTICLES", 50))
    news_cache_ttl: int = field(default_factory=lambda: _env_int("NEWS_CACHE_TTL", 60 * 30))
    use_news_features: bool = field(default_factory=lambda: _env_bool("USE_NEWS_FEATURES", True))

    # --- Logging ------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: _env_bool("LOG_JSON", False))
    sentry_dsn: str | None = field(default_factory=lambda: os.environ.get("SENTRY_DSN"))

    def news_keys(self) -> dict:
        return {
            "finnhub": self.finnhub_api_key,
            "gnews": self.gnews_api_key,
            "newsdata": self.newsdata_api_key,
            "newsapi_ai": self.newsapi_ai_key,
        }


settings = Settings()

# Apply persisted, GUI-editable overrides on top of env/.env (best-effort;
# never fatal if the file is missing or malformed).
try:
    from . import runtime_config as _runtime_config

    _runtime_config.apply_overrides_from_disk(settings)
except Exception:  # noqa: BLE001
    pass
