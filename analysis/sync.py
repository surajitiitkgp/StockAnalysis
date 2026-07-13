"""Incremental market-data synchronisation (local-first, idempotent).

This is the daily-update engine (Sec. 2). It brings the local SQLite archive up
to date without re-downloading history it already has:

  - only fetches bars newer than each symbol's last stored date (incremental);
  - upserts, so re-running never duplicates rows (idempotent);
  - isolates per-symbol failures — one bad symbol never aborts the run;
  - records a per-run health summary via :func:`analysis.store.record_sync`, so
    the status page / readiness probe can show freshness and failures;
  - is resumable: interrupted runs simply skip already-fresh symbols next time.

It reuses the resilient provider chain in :mod:`analysis.providers` (retry +
circuit breaker + validation), so it inherits fallback behaviour for free.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

import pandas as pd

from . import providers, store, universe
from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)

_LOCK = threading.Lock()      # only one sync at a time (idempotent, cheap)
_running = False


def is_running() -> bool:
    return _running


def _needs_update(symbol: str, max_age_days: int) -> bool:
    """True if the local copy is missing or older than ``max_age_days``."""
    last = store.last_date(symbol)
    if not last:
        return True
    try:
        last_dt = datetime.strptime(last, "%Y-%m-%d")
    except ValueError:
        return True
    # Business-day staleness: weekends aren't "missing" sessions.
    return pd.Timestamp.now().normalize() - pd.Timestamp(last_dt) > timedelta(days=max_age_days)


def _incremental_period(symbol: str) -> str:
    """Choose the smallest fetch window that covers the local gap.

    Fresh-ish symbols only need a short top-up; brand-new symbols get the full
    configured history. This keeps daily syncs light and provider-friendly.
    """
    last = store.last_date(symbol)
    if not last:
        return f"{settings.history_years}y"
    try:
        gap_days = (datetime.now() - datetime.strptime(last, "%Y-%m-%d")).days
    except ValueError:
        return f"{settings.history_years}y"
    if gap_days <= 7:
        return "1mo"
    if gap_days <= 60:
        return "3mo"
    if gap_days <= 200:
        return "1y"
    return f"{settings.history_years}y"


def sync_symbol(symbol: str, exchange: str = "NSE", force: bool = False) -> dict:
    """Bring one symbol up to date. Returns a small per-symbol result dict."""
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    ticker = universe.to_yahoo(base, exchange)
    if not force and not _needs_update(base, settings.stale_after_days):
        return {"symbol": base, "status": "fresh", "written": 0}
    period = "max" if force else _incremental_period(base)
    try:
        df, meta = providers.get_daily(ticker, period)
    except Exception as exc:  # noqa: BLE001
        log.info("sync fetch failed for %s: %s", base, exc)
        return {"symbol": base, "status": "error", "written": 0, "detail": str(exc)}
    if df is None or df.empty:
        return {"symbol": base, "status": "no_data", "written": 0,
                "provider": meta.get("provider")}
    # Only persist bars newer than what we already have (incremental upsert).
    last = store.last_date(base)
    if last and not force:
        df = df[df.index > pd.Timestamp(last)]
        if df.empty:
            return {"symbol": base, "status": "up_to_date", "written": 0}
    try:
        written = store.upsert_history(base, df)
    except Exception as exc:  # noqa: BLE001
        return {"symbol": base, "status": "error", "written": 0, "detail": str(exc)}
    return {"symbol": base, "status": "updated", "written": written,
            "provider": meta.get("provider")}


def sync(symbols=None, limit: int | None = None, force: bool = False,
         exchange: str = "NSE") -> dict:
    """Synchronise a set of symbols, recording a health summary.

    ``symbols`` defaults to the most-liquid slice of the universe (bounded by
    ``limit`` or ``settings.auto_download_limit``). Never raises: individual
    failures are collected and reported.
    """
    global _running
    if not _LOCK.acquire(blocking=False):
        return {"status": "already_running"}
    _running = True
    started = time.time()
    try:
        store.init_db()
        if symbols is None:
            cap = limit or settings.auto_download_limit or None
            symbols = [d["symbol"] for d in universe.screener_symbols(cap)]
        elif limit:
            symbols = symbols[:limit]

        ok = failed = skipped = written = 0
        failures: list[str] = []
        for sym in symbols:
            res = sync_symbol(sym, exchange, force=force)
            st = res["status"]
            written += res.get("written", 0)
            if st in ("updated",):
                ok += 1
            elif st in ("fresh", "up_to_date"):
                skipped += 1
            else:
                failed += 1
                failures.append(sym)
        elapsed = round(time.time() - started, 1)
        detail = (f"updated={ok} skipped={skipped} failed={failed} "
                  f"rows={written} in {elapsed}s")
        status = "ok" if failed == 0 else ("partial" if ok or skipped else "error")
        store.record_sync("daily_sync", "prices", status, detail,
                          symbols_ok=ok + skipped, symbols_failed=failed)
        log.info("daily sync complete: %s", detail)
        return {
            "status": status, "updated": ok, "skipped": skipped, "failed": failed,
            "rows_written": written, "elapsed_s": elapsed,
            "failures": failures[:50],
        }
    finally:
        _running = False
        _LOCK.release()


def _seconds_until_next_run(hour: int, minute: int) -> float:
    """Seconds until the next daily HH:MM (local time)."""
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def is_market_open(now: datetime | None = None) -> bool:
    """True if the NSE cash market is open right now (Mon-Fri, 09:15-15:30 IST).

    Assumes the host clock is IST (the app targets Indian markets). The window
    is configurable via ``MARKET_OPEN_*`` / ``MARKET_CLOSE_*`` settings.
    """
    now = now or datetime.now()
    if now.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    open_t = now.replace(hour=settings.market_open_hour,
                         minute=settings.market_open_minute, second=0, microsecond=0)
    close_t = now.replace(hour=settings.market_close_hour,
                          minute=settings.market_close_minute, second=0, microsecond=0)
    return open_t <= now <= close_t


def run_intraday_scheduler(stop_event: threading.Event | None = None) -> None:
    """Keep prices near-live during market hours (background thread).

    While the market is open it runs a light incremental sync of the most-liquid
    slice every ``intraday_sync_minutes`` minutes. Outside market hours it sleeps
    efficiently until the next open, so it costs nothing overnight/weekends and
    stays provider-friendly. Honours a stop event for clean shutdown.
    """
    every = max(1, settings.intraday_sync_minutes)
    limit = settings.intraday_sync_limit or settings.sync_limit
    log.info("intraday-sync scheduler armed: every %d min during market hours "
             "(%02d:%02d-%02d:%02d, top %d symbols)", every,
             settings.market_open_hour, settings.market_open_minute,
             settings.market_close_hour, settings.market_close_minute, limit)
    while not (stop_event and stop_event.is_set()):
        if is_market_open():
            try:
                sync(limit=limit)
            except Exception:  # noqa: BLE001
                log.warning("intraday sync failed", exc_info=True)
            wait = every * 60
        else:
            # Closed: nap in short chunks so shutdown/open is picked up promptly.
            wait = 300
        while wait > 0 and not (stop_event and stop_event.is_set()):
            chunk = min(wait, 60)
            time.sleep(chunk)
            wait -= chunk


def run_scheduler(stop_event: threading.Event | None = None) -> None:
    """Daily post-close scheduler loop (Sec. 2). Runs in a background thread.

    Sleeps until the configured market-close-plus time, syncs, then repeats.
    Weekends still run (cheap: everything is already fresh). Honours a stop
    event for clean shutdown.
    """
    hour = settings.sync_hour
    minute = settings.sync_minute
    log.info("daily-sync scheduler armed for %02d:%02d local", hour, minute)
    while not (stop_event and stop_event.is_set()):
        wait = _seconds_until_next_run(hour, minute)
        # Wake periodically so a stop event is honoured promptly.
        while wait > 0 and not (stop_event and stop_event.is_set()):
            chunk = min(wait, 300)
            time.sleep(chunk)
            wait -= chunk
        if stop_event and stop_event.is_set():
            break
        try:
            sync()
        except Exception:  # noqa: BLE001
            log.warning("scheduled sync failed", exc_info=True)
