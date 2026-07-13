"""Historical price data fetching for Indian stocks (local-first, resilient).

Data flow:
  1. Serve daily history from the local SQLite archive when available.
  2. Top up recent bars from a provider (Yahoo, then Stooq fallback) when the
     local copy is stale.
  3. For symbols not in the store (BSE / brand-new), fetch from a provider and
     persist.

All fetches go through :mod:`analysis.providers` (retry + backoff + circuit
breaker + validation) and results are cached via :mod:`analysis.cache`
(shared/Redis-capable, single-flight). Every load carries freshness metadata so
the API/UI can flag stale or degraded data instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from . import cache, providers, store, universe
from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)


@dataclass
class StockData:
    symbol: str            # base symbol, e.g. RELIANCE
    exchange: str          # NSE or BSE
    yahoo_ticker: str      # e.g. RELIANCE.NS
    name: str
    history: pd.DataFrame  # daily OHLCV, DatetimeIndex
    intraday: pd.DataFrame # 5-minute bars for the latest sessions (may be empty)
    info: dict             # fundamentals / metadata from the provider
    meta: dict = field(default_factory=dict)  # freshness / provider / quality


def _period_to_start(period: str):
    period = (period or "").strip().lower()
    if not period or period == "max":
        return None
    try:
        if period.endswith("y"):
            days = int(float(period[:-1]) * 365.25)
        elif period.endswith("mo"):
            days = int(float(period[:-2]) * 30.4)
        elif period.endswith("d"):
            days = int(period[:-1])
        else:
            return None
    except ValueError:
        return None
    return (datetime.now() - timedelta(days=days + 5)).strftime("%Y-%m-%d")


def _freshness(df: pd.DataFrame) -> dict:
    """Describe how current a daily frame is."""
    if df is None or df.empty:
        return {"rows": 0, "last_date": None, "age_days": None, "stale": True}
    last = df.index.max()
    age = (datetime.now() - last.to_pydatetime()).days
    return {
        "rows": int(len(df)),
        "last_date": last.strftime("%Y-%m-%d"),
        "age_days": int(age),
        "stale": age > settings.stale_after_days,
    }


def _load_daily(symbol: str, exchange: str, period: str) -> tuple[pd.DataFrame, dict]:
    ticker = universe.to_yahoo(symbol, exchange)
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    start = _period_to_start(period)
    use_store = exchange.upper() == "NSE" and not symbol.upper().endswith(".BO")

    meta: dict = {"provider": None, "source": None, "quality": None}
    df = pd.DataFrame()

    if use_store:
        try:
            stored = store.get_history(base)
        except Exception:  # noqa: BLE001
            log.warning("store read failed for %s", base, exc_info=True)
            stored = pd.DataFrame()
        if not stored.empty:
            meta["source"] = "store"
            last = stored.index.max()
            if (datetime.now() - last.to_pydatetime()) > timedelta(days=settings.stale_after_days):
                fresh, fmeta = providers.get_daily(ticker, "1mo")
                if not fresh.empty:
                    try:
                        store.upsert_history(base, fresh)
                        stored = store.get_history(base)
                        meta["source"] = "store+topup"
                        meta["provider"] = fmeta.get("provider")
                    except Exception:  # noqa: BLE001
                        log.warning("store top-up failed for %s", base, exc_info=True)
            df = stored if start is None else stored[stored.index >= start]

    if df.empty:
        df, fmeta = providers.get_daily(ticker, period)
        meta["provider"] = fmeta.get("provider")
        meta["quality"] = fmeta.get("quality")
        meta["source"] = "provider"
        if use_store and not df.empty:
            try:
                store.upsert_history(base, df)
            except Exception:  # noqa: BLE001
                log.warning("store persist failed for %s", base, exc_info=True)

    # Last-resort fallback: if every provider failed but we DO have something in
    # the local archive (possibly stale), serve that instead of returning
    # nothing. Stale-but-real data beats a hard "no data" error for the user.
    if df.empty and use_store:
        try:
            stored = store.get_history(base)
        except Exception:  # noqa: BLE001
            stored = pd.DataFrame()
        if not stored.empty:
            log.info("serving stale local data for %s (providers unavailable)", base)
            df = stored if start is None else stored[stored.index >= start]
            if df.empty:
                df = stored  # ignore the period window rather than serve nothing
            meta["source"] = "store_stale_fallback"
            meta["degraded"] = True

    meta["freshness"] = _freshness(df)
    return df, meta


def get_daily_history(symbol: str, exchange: str = "NSE", period: str = "2y") -> pd.DataFrame:
    """Daily OHLCV history (local store first, providers as top-up/fallback)."""
    df, _ = get_daily_history_with_meta(symbol, exchange, period)
    return df


def get_daily_history_with_meta(symbol: str, exchange: str = "NSE",
                                period: str = "2y") -> tuple[pd.DataFrame, dict]:
    ticker = universe.to_yahoo(symbol, exchange)
    key = f"daily:{ticker}:{period}"
    cached = cache.get(key)
    if cached is not None and not getattr(cached[0], "empty", True):
        return cached

    result = _load_daily(symbol, exchange, period)
    # Only cache non-empty results. Caching an empty frame after a transient
    # provider outage would make the failure sticky for the whole TTL.
    if result is not None and not getattr(result[0], "empty", True):
        cache.set(key, result, settings.cache_ttl_daily)
    if result is None:
        return pd.DataFrame(), {"provider": None, "freshness": _freshness(pd.DataFrame())}
    return result


def get_intraday(symbol: str, exchange: str = "NSE", period: str = "5d",
                 interval: str = "5m") -> pd.DataFrame:
    """Intraday OHLCV bars. Cached per (ticker, period, interval)."""
    ticker = universe.to_yahoo(symbol, exchange)
    key = f"intraday:{ticker}:{period}:{interval}"
    result = cache.get_or_compute(
        key, settings.cache_ttl_intraday,
        lambda: providers.get_intraday(ticker, period, interval),
    )
    return result if result is not None else pd.DataFrame()


def get_info(symbol: str, exchange: str = "NSE") -> dict:
    """Fundamental metadata (PE, market cap, 52w range, etc.). Best-effort."""
    ticker = universe.to_yahoo(symbol, exchange)
    key = f"info:{ticker}"
    result = cache.get_or_compute(
        key, settings.cache_ttl_info,
        lambda: providers.get_info(ticker),
    )
    return result if result is not None else {}


def load_stock(symbol: str, exchange: str = "NSE", with_intraday: bool = True,
               with_info: bool = True, period: str = "2y") -> StockData:
    """Load everything needed to analyse a single stock."""
    ticker = universe.to_yahoo(symbol, exchange)
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    name = universe.SYMBOL_TO_NAME.get(base, base)

    history, meta = get_daily_history_with_meta(symbol, exchange, period=period)
    intraday = get_intraday(symbol, exchange) if with_intraday else pd.DataFrame()
    info = get_info(symbol, exchange) if with_info else {}

    return StockData(
        symbol=base,
        exchange=exchange.upper(),
        yahoo_ticker=ticker,
        name=name,
        history=history,
        intraday=intraday,
        info=info,
        meta=meta,
    )
