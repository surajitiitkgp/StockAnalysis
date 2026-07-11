"""Thin, optional wrapper around the unofficial NSE India API.

Uses `bennythadikaran/NseIndiaApi <https://bennythadikaran.github.io/NseIndiaApi/>`_
(``pip install nse[local]``) to pull data **directly from NSE**:

  - :func:`equity_history` — daily OHLCV for an NSE equity (price redundancy)
  - :func:`vix_history`    — India VIX close (a volatility / "fear" feature)

Everything here is best-effort and dependency-optional:

  - if the ``nse`` package isn't installed, or the config toggle is off, or NSE
    is unreachable (it blocks many non-Indian / cloud IPs), every function
    returns an empty frame/series and callers degrade gracefully;
  - the underlying library self-throttles to 3 req/s, and long date ranges are
    chunked into <=1-year windows (NSE caps a single request).

A single :class:`NSE` client is reused (it manages cookies in a download
folder) behind a lock, so repeated calls don't re-do the handshake.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timedelta

import pandas as pd

from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)

_LOCK = threading.Lock()
_CLIENT = None            # cached nse.NSE instance
_UNAVAILABLE = False      # set once if import/handshake permanently fails
_CHUNK_DAYS = 300         # NSE caps a single historical request (~1 year)


def is_enabled() -> bool:
    return bool(settings.use_nse_api) and not _UNAVAILABLE


def reset() -> None:
    """Forget the cached client / availability flag (after a config change)."""
    global _CLIENT, _UNAVAILABLE
    with _LOCK:
        _CLIENT = None
        _UNAVAILABLE = False


def _get_client():
    """Return a cached NSE client, or ``None`` if unavailable."""
    global _CLIENT, _UNAVAILABLE
    if _UNAVAILABLE or not settings.use_nse_api:
        return None
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is not None:
            return _CLIENT
        try:
            from nse import NSE  # optional dependency
        except Exception:  # noqa: BLE001
            log.info("nse package not installed; NSE API disabled "
                     "(pip install \"nse[local]\")")
            _UNAVAILABLE = True
            return None
        try:
            import os
            os.makedirs(settings.data_dir, exist_ok=True)
            _CLIENT = NSE(download_folder=settings.data_dir,
                          server=settings.nse_server_mode)
        except Exception:  # noqa: BLE001
            log.info("NSE client init failed; disabling NSE API", exc_info=True)
            _UNAVAILABLE = True
            return None
    return _CLIENT


def _period_to_start(period: str) -> date:
    today = date.today()
    p = (period or "10y").strip().lower()
    if p in ("max", "all"):
        return today - timedelta(days=365 * 15)
    try:
        if p.endswith("y"):
            return today - timedelta(days=int(float(p[:-1]) * 365) + 5)
        if p.endswith("mo"):
            return today - timedelta(days=int(float(p[:-2]) * 31) + 5)
        if p.endswith("d"):
            return today - timedelta(days=int(float(p[:-1])) + 1)
    except ValueError:
        pass
    return today - timedelta(days=365 * 10)


def _chunks(start: date, end: date):
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=_CHUNK_DAYS), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def _parse_day(value) -> datetime | None:
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(value).strip().title(), fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:  # noqa: BLE001
        return None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def equity_history(symbol: str, period: str = "10y") -> pd.DataFrame:
    """Daily OHLCV for an NSE equity as a DatetimeIndex frame (may be empty)."""
    client = _get_client()
    if client is None:
        return pd.DataFrame()
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    start = _period_to_start(period)
    end = date.today()
    rows: list[dict] = []
    try:
        for frm, to in _chunks(start, end):
            data = client.fetch_equity_historical_data(base, from_date=frm, to_date=to)
            for r in data or []:
                if (r.get("chSeries") or "EQ") != "EQ":
                    continue
                dt = _parse_day(r.get("mtimestamp"))
                close = _to_float(r.get("chClosingPrice"))
                if dt is None or close is None:
                    continue
                rows.append({
                    "Date": dt,
                    "Open": _to_float(r.get("chOpeningPrice")) or close,
                    "High": _to_float(r.get("chTradeHighPrice")) or close,
                    "Low": _to_float(r.get("chTradeLowPrice")) or close,
                    "Close": close,
                    "Volume": _to_float(r.get("chTotTradedQty")) or 0.0,
                })
    except Exception:  # noqa: BLE001
        log.info("NSE equity history failed for %s", base, exc_info=True)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset="Date").set_index("Date").sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def vix_history(period: str = "10y") -> pd.Series:
    """India VIX daily close as a DatetimeIndex Series (may be empty)."""
    client = _get_client()
    if client is None:
        return pd.Series(dtype=float)
    start = _period_to_start(period)
    end = date.today()
    points: dict[datetime, float] = {}
    try:
        for frm, to in _chunks(start, end):
            data = client.fetch_historical_vix_data(from_date=frm, to_date=to)
            for r in data or []:
                dt = _parse_day(r.get("EOD_TIMESTAMP"))
                close = _to_float(r.get("EOD_CLOSE_INDEX_VAL"))
                if dt is not None and close is not None:
                    points[dt] = close
    except Exception:  # noqa: BLE001
        log.info("NSE India VIX history failed", exc_info=True)
        return pd.Series(dtype=float)

    if not points:
        return pd.Series(dtype=float)
    s = pd.Series(points).sort_index()
    s.index = pd.to_datetime(s.index)
    s.name = "vix"
    return s
