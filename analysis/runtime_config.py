"""Runtime-editable configuration (persisted overrides on top of env/.env).

The :class:`~analysis.config.Settings` singleton is a frozen dataclass built
once from environment variables. This module lets a subset of those settings be
changed **at runtime** — from the dashboard's settings panel — without a
restart, and persists the changes to a git-ignored JSON file so they survive
restarts too.

Precedence: dataclass defaults < ``.env`` / real env < persisted overrides.

Only a curated whitelist (:data:`FIELD_SPECS`) is editable. Everything is typed
and validated; secret values (API keys) are never sent to the browser — the API
reports only whether each is configured.

Most editable settings are read live by their consumers (e.g. ``news.is_enabled``
reads ``settings.news_enabled`` on every call), so changes take effect on the
next request. For the few that are wired at import time (the price-provider
chain, cached news-provider instances, the NSE client), :func:`apply` triggers
the relevant rebuild hooks and clears the result cache.
"""

from __future__ import annotations

import json
import os
import threading

from .logging_config import get_logger

log = get_logger(__name__)

_LOCK = threading.Lock()
_FILENAME = "runtime_config.json"


# Each spec: key, human label, group, type, and optional help / bounds.
#   type ∈ {"bool", "int", "text", "csv", "secret"}
FIELD_SPECS: list[dict] = [
    # --- Features -----------------------------------------------------------
    {"key": "news_enabled", "label": "Enable news & sentiment", "group": "Features",
     "type": "bool", "help": "Master switch for the news layer."},
    {"key": "use_news_features", "label": "Use news in ML model", "group": "Features",
     "type": "bool", "help": "Feed the sentiment archive into price-forecast features."},
    {"key": "use_market_features", "label": "Use broad-market context", "group": "Features",
     "type": "bool", "help": "Index dynamics, relative strength & global sentiment."},
    {"key": "use_vix_feature", "label": "Use India VIX", "group": "Features",
     "type": "bool", "help": "Volatility / fear gauge (needs the NSE API)."},
    {"key": "use_nse_api", "label": "Use NSE India API", "group": "Features",
     "type": "bool", "help": "Direct-from-source NSE data (unofficial; may be IP-blocked)."},

    # --- News tuning --------------------------------------------------------
    {"key": "news_providers", "label": "News provider order", "group": "News",
     "type": "csv", "help": "Comma list: finnhub, newsapi_ai, newsdata, gnews."},
    {"key": "news_lookback_days", "label": "News lookback (days)", "group": "News",
     "type": "int", "min": 1, "max": 365},
    {"key": "news_max_articles", "label": "Max articles per fetch", "group": "News",
     "type": "int", "min": 5, "max": 200},
    {"key": "news_cache_ttl", "label": "News cache TTL (sec)", "group": "News",
     "type": "int", "min": 30, "max": 86400},

    # --- API keys (secret) --------------------------------------------------
    {"key": "finnhub_api_key", "label": "Finnhub API key", "group": "API keys", "type": "secret"},
    {"key": "gnews_api_key", "label": "GNews API key", "group": "API keys", "type": "secret"},
    {"key": "newsdata_api_key", "label": "NewsData.io API key", "group": "API keys", "type": "secret"},
    {"key": "newsapi_ai_key", "label": "NewsAPI.ai key", "group": "API keys", "type": "secret"},
    {"key": "twelvedata_api_key", "label": "Twelve Data key", "group": "API keys", "type": "secret"},
    {"key": "alphavantage_api_key", "label": "Alpha Vantage key", "group": "API keys", "type": "secret"},

    # --- Machine learning ---------------------------------------------------
    {"key": "ml_persist", "label": "Persist trained models", "group": "Machine learning",
     "type": "bool", "help": "Cache model bundles to disk between runs."},
    {"key": "ml_min_rows", "label": "Min rows to train", "group": "Machine learning",
     "type": "int", "min": 60, "max": 5000},
    {"key": "ml_cache_ttl", "label": "Forecast cache TTL (sec)", "group": "Machine learning",
     "type": "int", "min": 30, "max": 86400},
    {"key": "ml_model_ttl_hours", "label": "Model bundle TTL (hours)", "group": "Machine learning",
     "type": "int", "min": 1, "max": 720},
]

SPEC_BY_KEY: dict[str, dict] = {s["key"]: s for s in FIELD_SPECS}


class ConfigError(ValueError):
    """Raised for an invalid runtime-config change."""


def _config_path(settings) -> str:
    return os.path.join(settings.data_dir, _FILENAME)


def _coerce(spec: dict, value):
    """Validate + coerce an incoming value for ``spec``. Raises ConfigError."""
    kind = spec["type"]
    key = spec["key"]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if kind == "int":
        try:
            iv = int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{key}: expected an integer")
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and iv < lo:
            raise ConfigError(f"{key}: must be >= {lo}")
        if hi is not None and iv > hi:
            raise ConfigError(f"{key}: must be <= {hi}")
        return iv
    if kind == "csv":
        if isinstance(value, (list, tuple)):
            parts = [str(p).strip().lower() for p in value]
        else:
            parts = [p.strip().lower() for p in str(value).split(",")]
        return tuple(p for p in parts if p)
    if kind == "secret":
        return str(value) if value is not None else ""
    # text
    return str(value).strip()


def _to_jsonable(spec: dict, value):
    """Convert a coerced settings value into a JSON-serialisable form."""
    if spec["type"] == "csv":
        return list(value or ())
    return value


def load_overrides(settings) -> dict:
    """Read the persisted override dict (or empty on any problem)."""
    path = _config_path(settings)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        log.warning("could not read %s; ignoring overrides", path, exc_info=True)
        return {}


def _save_overrides(settings, overrides: dict) -> None:
    path = _config_path(settings)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(overrides, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _set_on_settings(settings, key: str, value) -> None:
    # Settings is a frozen dataclass; bypass the freeze for controlled updates.
    object.__setattr__(settings, key, value)


def apply_overrides_from_disk(settings) -> None:
    """Apply any persisted overrides onto ``settings`` (called at startup)."""
    overrides = load_overrides(settings)
    for key, raw in overrides.items():
        spec = SPEC_BY_KEY.get(key)
        if not spec:
            continue
        try:
            _set_on_settings(settings, key, _coerce(spec, raw))
        except ConfigError:
            log.warning("ignoring invalid persisted override for %s", key)


def current_values(settings) -> list[dict]:
    """Field descriptors + current values for the settings UI (secrets masked)."""
    out = []
    for spec in FIELD_SPECS:
        item = {k: spec[k] for k in ("key", "label", "group", "type", "help",
                                     "min", "max") if k in spec}
        val = getattr(settings, spec["key"], None)
        if spec["type"] == "secret":
            item["configured"] = bool(val)
        elif spec["type"] == "csv":
            item["value"] = ",".join(val or ())
        else:
            item["value"] = val
        out.append(item)
    return out


def _apply_side_effects(changed: set[str]) -> None:
    """Rebuild anything wired at import time and drop stale cached results."""
    from . import cache

    if changed & {"twelvedata_api_key", "alphavantage_api_key", "use_nse_api"}:
        try:
            from . import providers
            providers.rebuild_daily_providers()
        except Exception:  # noqa: BLE001
            log.warning("provider rebuild failed", exc_info=True)

    if changed & {"finnhub_api_key", "gnews_api_key", "newsdata_api_key",
                  "newsapi_ai_key", "news_providers"}:
        try:
            from . import news
            news.reset_providers()
        except Exception:  # noqa: BLE001
            log.warning("news provider reset failed", exc_info=True)

    if "use_nse_api" in changed:
        try:
            from . import nse_client
            nse_client.reset()
        except Exception:  # noqa: BLE001
            log.warning("nse client reset failed", exc_info=True)

    # Any change can affect cached predictions / news / market context.
    cache.clear()


def update(settings, changes: dict) -> dict:
    """Validate + apply ``changes`` to ``settings`` and persist them.

    ``changes`` maps field keys to new values. Secret fields with an empty value
    are treated as "leave unchanged" (so the masked UI never wipes a key by
    accident); pass a value to update or the literal string ``"__clear__"`` to
    remove one.

    Returns ``{"applied": [...], "settings": current_values(...)}`` or raises
    :class:`ConfigError` on the first invalid field.
    """
    with _LOCK:
        overrides = load_overrides(settings)
        applied: list[str] = []
        for key, raw in (changes or {}).items():
            spec = SPEC_BY_KEY.get(key)
            if not spec:
                raise ConfigError(f"unknown setting: {key}")
            if spec["type"] == "secret":
                if raw is None or str(raw) == "":
                    continue  # unchanged
                coerced = None if str(raw) == "__clear__" else str(raw)
            else:
                coerced = _coerce(spec, raw)
            _set_on_settings(settings, key, coerced)
            overrides[key] = _to_jsonable(spec, coerced)
            applied.append(key)

        if applied:
            _save_overrides(settings, overrides)
            _apply_side_effects(set(applied))
    return {"applied": applied, "settings": current_values(settings)}
