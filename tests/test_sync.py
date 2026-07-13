"""Tests for the incremental sync engine (no network)."""

from __future__ import annotations

import pandas as pd

from analysis import sync, store
from tests.conftest import make_ohlcv


def _patch_provider(monkeypatch, df):
    monkeypatch.setattr(sync.providers, "get_daily",
                        lambda *a, **k: (df, {"provider": "fake"}))


def test_sync_symbol_writes_new_rows(monkeypatch):
    df = make_ohlcv(n=300)
    _patch_provider(monkeypatch, df)
    res = sync.sync_symbol("TESTSYNC", force=True)
    assert res["status"] == "updated"
    assert res["written"] > 0
    assert store.row_count("TESTSYNC") == res["written"]


def test_sync_is_idempotent(monkeypatch):
    df = make_ohlcv(n=300)
    _patch_provider(monkeypatch, df)
    sync.sync_symbol("IDEM", force=True)
    rows_after_first = store.row_count("IDEM")
    # Re-running with the same data must not duplicate rows (upsert).
    sync.sync_symbol("IDEM", force=True)
    assert store.row_count("IDEM") == rows_after_first


def test_sync_incremental_only_new(monkeypatch):
    df = make_ohlcv(n=300)
    # Seed the store with all but the last 5 bars.
    store.upsert_history("INCR", df.iloc[:-5])
    seeded = store.row_count("INCR")
    _patch_provider(monkeypatch, df)
    res = sync.sync_symbol("INCR", force=False)
    # Only the 5 missing bars should be written.
    assert res["status"] == "updated"
    assert res["written"] == 5
    assert store.row_count("INCR") == seeded + 5


def test_sync_fresh_symbol_skipped(monkeypatch):
    df = make_ohlcv(n=50)
    # Make the stored data end today so it's considered fresh.
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=50)
    df.index = idx
    store.upsert_history("FRESH", df)
    _patch_provider(monkeypatch, df)
    res = sync.sync_symbol("FRESH", force=False)
    assert res["status"] in ("fresh", "up_to_date")
    assert res["written"] == 0


def test_sync_run_records_health(monkeypatch):
    df = make_ohlcv(n=300)
    _patch_provider(monkeypatch, df)
    result = sync.sync(symbols=["A", "B", "C"], force=True)
    assert result["status"] in ("ok", "partial")
    assert result["updated"] == 3
    health = {h["source"]: h for h in store.sync_health()}
    assert "daily_sync" in health
    assert health["daily_sync"]["symbols_ok"] >= 3


def test_sync_isolates_failures(monkeypatch):
    df = make_ohlcv(n=300)

    def flaky(ticker, period):
        if ticker.startswith("BAD"):
            raise RuntimeError("provider down")
        return df, {"provider": "fake"}

    monkeypatch.setattr(sync.providers, "get_daily", flaky)
    result = sync.sync(symbols=["GOOD1", "BAD1", "GOOD2"], force=True)
    # One bad symbol must not abort the run.
    assert result["updated"] == 2
    assert result["failed"] == 1
    assert "BAD1" in result["failures"]


def test_incremental_period_scales_with_gap():
    # Brand-new symbol -> full history window.
    assert sync._incremental_period("NEVERSEEN").endswith("y")


# --------------------------------------------------------------------------- #
# Intraday market-hours gating for the price auto-refresh scheduler.
# --------------------------------------------------------------------------- #
def test_is_market_open_weekday_hours():
    from datetime import datetime
    from analysis import sync
    # Wednesday 2026-07-15 at 11:00 IST -> open.
    assert sync.is_market_open(datetime(2026, 7, 15, 11, 0)) is True
    # Before open (09:00) and after close (16:00) -> closed.
    assert sync.is_market_open(datetime(2026, 7, 15, 9, 0)) is False
    assert sync.is_market_open(datetime(2026, 7, 15, 16, 0)) is False


def test_is_market_open_weekend():
    from datetime import datetime
    from analysis import sync
    # Saturday / Sunday are always closed, even midday.
    assert sync.is_market_open(datetime(2026, 7, 18, 11, 0)) is False
    assert sync.is_market_open(datetime(2026, 7, 19, 11, 0)) is False


def test_is_market_open_boundaries():
    from datetime import datetime
    from analysis import sync
    # Exactly at open (09:15) and close (15:30) count as open (inclusive).
    assert sync.is_market_open(datetime(2026, 7, 15, 9, 15)) is True
    assert sync.is_market_open(datetime(2026, 7, 15, 15, 30)) is True
