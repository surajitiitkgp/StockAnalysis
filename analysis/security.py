"""Lightweight, dependency-free security helpers.

Provides:
  - a sliding-window :class:`RateLimiter` (per-key) used for login throttling
    and API rate limiting,
  - CSRF token helpers backed by the Flask session.

These are intentionally in-memory and process-local — good enough for a
single-node deployment and robust against accidental hammering / brute force.
For multi-node you'd back the limiter with Redis (the cache module already
shows the pattern).
"""

from __future__ import annotations

import secrets
import threading
import time

from flask import session

from .config import settings


class RateLimiter:
    """Fixed-window rate limiter. Returns (allowed, retry_after_seconds)."""

    def __init__(self, max_events: int, window_seconds: int):
        self.max_events = max_events
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            events = [t for t in self._hits.get(key, []) if now - t < self.window]
            if len(events) >= self.max_events:
                retry = int(self.window - (now - events[0])) + 1
                self._hits[key] = events
                return False, max(retry, 1)
            events.append(now)
            self._hits[key] = events
            return True, 0

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


# Shared limiters configured from settings.
login_limiter = RateLimiter(settings.login_max_attempts, settings.login_window_seconds)
api_limiter = RateLimiter(settings.api_rate_limit, settings.api_rate_window)


# --------------------------------------------------------------------------- #
# CSRF
# --------------------------------------------------------------------------- #
_CSRF_KEY = "_csrf_token"


def csrf_token() -> str:
    token = session.get(_CSRF_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[_CSRF_KEY] = token
    return token


def validate_csrf(submitted: str | None) -> bool:
    expected = session.get(_CSRF_KEY)
    if not expected or not submitted:
        return False
    return secrets.compare_digest(expected, submitted)
