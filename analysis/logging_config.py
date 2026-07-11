"""Structured logging setup shared across the app.

Replaces the ``except Exception: pass`` silent-failure pattern with real,
level-aware logging. Supports plain or JSON output (``LOG_JSON=1``) and an
optional Sentry hook when ``SENTRY_DSN`` is configured.
"""

from __future__ import annotations

import json
import logging
import sys
import time

from .config import settings

_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("symbol", "exchange", "provider", "request_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> None:
    """Configure the root logger exactly once (idempotent)."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if settings.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root.addHandler(handler)

    # yfinance/urllib3 are noisy at INFO; keep them at WARNING.
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    if settings.sentry_dsn:
        try:  # optional dependency
            import sentry_sdk  # type: ignore

            sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)
            logging.getLogger(__name__).info("Sentry error tracking enabled")
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning("SENTRY_DSN set but sentry_sdk unavailable")

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
