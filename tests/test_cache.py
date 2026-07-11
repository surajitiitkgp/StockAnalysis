"""Tests for the pluggable cache + single-flight behaviour."""

from __future__ import annotations

import threading
import time

from analysis import cache


def test_set_get_and_expiry():
    cache.clear()
    cache.set("k1", {"v": 1}, ttl=60)
    assert cache.get("k1") == {"v": 1}
    cache.set("k2", "x", ttl=1)
    time.sleep(1.1)
    assert cache.get("k2") is None


def test_get_or_compute_caches():
    cache.clear()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 42

    assert cache.get_or_compute("kc", 60, compute) == 42
    assert cache.get_or_compute("kc", 60, compute) == 42
    assert calls["n"] == 1  # computed once, then served from cache


def test_single_flight_under_concurrency():
    cache.clear()
    calls = {"n": 0}
    lock = threading.Lock()

    def compute():
        with lock:
            calls["n"] += 1
        time.sleep(0.2)  # simulate a slow fetch
        return "value"

    threads = [threading.Thread(target=lambda: cache.get_or_compute("sf", 60, compute))
               for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1  # only one thread computed the missing value
