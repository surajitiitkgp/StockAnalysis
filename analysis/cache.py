"""Pluggable TTL cache with single-flight (stampede) protection.

Backend selection is automatic:
  - if ``REDIS_URL`` is set and ``redis`` importable -> shared Redis cache
  - otherwise -> process-local in-memory cache

The public API (``get_or_compute``) guarantees that concurrent callers asking
for the same missing key only trigger **one** underlying computation; the rest
wait for and reuse the result. This stops the screener's worker pool from
firing a dozen identical Yahoo requests at once.
"""

from __future__ import annotations

import pickle
import threading
import time
from typing import Any, Callable

from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)


class _InMemoryBackend:
    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if time.time() > expires_at:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: int):
        with self._lock:
            self._store[key] = (time.time() + ttl, value)

    def clear(self):
        with self._lock:
            self._store.clear()


class _RedisBackend:
    def __init__(self, client):
        self._r = client

    def get(self, key: str):
        raw = self._r.get(key)
        if raw is None:
            return None
        try:
            return pickle.loads(raw)
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, value: Any, ttl: int):
        try:
            self._r.setex(key, ttl, pickle.dumps(value))
        except Exception:  # noqa: BLE001
            log.warning("redis set failed for key=%s", key, exc_info=True)

    def clear(self):
        try:
            self._r.flushdb()
        except Exception:  # noqa: BLE001
            pass


def _make_backend():
    if settings.redis_url:
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(settings.redis_url)
            client.ping()
            log.info("Cache backend: Redis (%s)", settings.redis_url)
            return _RedisBackend(client)
        except Exception:  # noqa: BLE001
            log.warning("REDIS_URL set but Redis unavailable; using in-memory cache",
                        exc_info=True)
    log.info("Cache backend: in-memory")
    return _InMemoryBackend()


_backend = _make_backend()

# Per-key locks for single-flight. A lock guarding the lock dict itself.
_flight_locks: dict[str, threading.Lock] = {}
_flight_guard = threading.Lock()


def _key_lock(key: str) -> threading.Lock:
    with _flight_guard:
        lock = _flight_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _flight_locks[key] = lock
        return lock


def get(key: str):
    return _backend.get(key)


def set(key: str, value: Any, ttl: int):  # noqa: A001 - deliberate cache API
    _backend.set(key, value, ttl)


def clear():
    _backend.clear()


def get_or_compute(key: str, ttl: int, compute: Callable[[], Any]):
    """Return cached value for ``key`` or compute it exactly once.

    Uses double-checked locking so only one thread computes a missing value
    while others block and then read the freshly cached result.
    """
    cached = _backend.get(key)
    if cached is not None:
        return cached

    lock = _key_lock(key)
    with lock:
        cached = _backend.get(key)  # re-check after acquiring the lock
        if cached is not None:
            return cached
        value = compute()
        if value is not None:
            _backend.set(key, value, ttl)
        return value
