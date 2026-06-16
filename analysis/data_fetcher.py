"""Historical price data fetching for Indian stocks via yfinance.

NSE/BSE do not offer a reliable public bulk API (their site actively blocks
automated requests). yfinance proxies Yahoo Finance, which carries clean
historical OHLCV data for both NSE (``.NS``) and BSE (``.BO``) listings and is
the most dependable free source. A simple in-memory TTL cache avoids hammering
the upstream service when the screener scans many symbols.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from . import store, universe

_CACHE_TTL_SECONDS = 60 * 15  # 15 minutes
_cache: dict[tuple, tuple[float, object]] = {}
_lock = threading.Lock()


def _cache_get(key):
    with _lock:
        item = _cache.get(key)
        if item is None:
            return None
        ts, value = item
        if time.time() - ts > _CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        return value


def _cache_set(key, value):
    with _lock:
        _cache[key] = (time.time(), value)


@dataclass
class StockData:
    symbol: str            # base symbol, e.g. RELIANCE
    exchange: str          # NSE or BSE
    yahoo_ticker: str      # e.g. RELIANCE.NS
    name: str
    history: pd.DataFrame  # daily OHLCV, DatetimeIndex
    intraday: pd.DataFrame # 5-minute bars for the latest sessions (may be empty)
    info: dict             # fundamentals / metadata from yfinance


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["Close"])
    # Drop timezone for clean JSON serialisation downstream.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _period_to_start(period: str) -> str | None:
    """Translate a yfinance-style period (e.g. '2y', '6mo', 'max') to a start date."""
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


def _fetch_remote_daily(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    return _clean_ohlcv(df)


def get_daily_history(symbol: str, exchange: str = "NSE", period: str = "2y") -> pd.DataFrame:
    """Daily OHLCV history — local SQLite store first, Yahoo as top-up/fallback.

    For NSE we keep a local 10-year archive (see ``scripts/download_history.py``).
    We serve from there for speed/offline use, and top up the latest bars from
    Yahoo when the stored copy is a few days stale. BSE always goes to Yahoo.
    """
    ticker = universe.to_yahoo(symbol, exchange)
    key = ("daily", ticker, period)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    start = _period_to_start(period)

    # BSE (or any pre-suffixed BSE ticker) is not in the local store -> remote.
    use_store = exchange.upper() == "NSE" and not symbol.upper().endswith(".BO")

    df = pd.DataFrame()
    if use_store:
        stored = store.get_history(base)
        if not stored.empty:
            last = stored.index.max()
            # Top up recent bars if the local copy is more than ~3 days stale.
            if (datetime.now() - last.to_pydatetime()) > timedelta(days=3):
                try:
                    fresh = _fetch_remote_daily(ticker, "1mo")
                    if not fresh.empty:
                        store.upsert_history(base, fresh)
                        stored = store.get_history(base)
                except Exception:  # noqa: BLE001 - offline: serve what we have
                    pass
            df = stored if start is None else stored[stored.index >= start]

    if df.empty:
        # No local data (new symbol / BSE / empty store) -> fetch & persist.
        df = _fetch_remote_daily(ticker, period)
        if use_store and not df.empty:
            try:
                store.upsert_history(base, df)
            except Exception:  # noqa: BLE001
                pass

    _cache_set(key, df)
    return df


def get_intraday(symbol: str, exchange: str = "NSE", period: str = "5d",
                 interval: str = "5m") -> pd.DataFrame:
    """Intraday OHLCV bars. Cached per (ticker, period, interval)."""
    ticker = universe.to_yahoo(symbol, exchange)
    key = ("intraday", ticker, period, interval)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        df = _clean_ohlcv(df)
    except Exception:
        df = pd.DataFrame()
    _cache_set(key, df)
    return df


def get_info(symbol: str, exchange: str = "NSE") -> dict:
    """Fundamental metadata (PE, market cap, 52w range, etc.). Best-effort."""
    ticker = universe.to_yahoo(symbol, exchange)
    key = ("info", ticker)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    info = {}
    try:
        raw = yf.Ticker(ticker).info or {}
        wanted = [
            "longName", "shortName", "sector", "industry", "currency",
            "marketCap", "trailingPE", "forwardPE", "priceToBook",
            "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "beta", "returnOnEquity", "profitMargins", "debtToEquity",
            "earningsGrowth", "revenueGrowth", "recommendationKey",
            "targetMeanPrice", "averageVolume",
        ]
        info = {k: raw.get(k) for k in wanted if raw.get(k) is not None}
    except Exception:
        info = {}
    _cache_set(key, info)
    return info


def load_stock(symbol: str, exchange: str = "NSE", with_intraday: bool = True,
               with_info: bool = True, period: str = "2y") -> StockData:
    """Load everything needed to analyse a single stock."""
    ticker = universe.to_yahoo(symbol, exchange)
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    name = universe.SYMBOL_TO_NAME.get(base, base)

    history = get_daily_history(symbol, exchange, period=period)
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
    )
