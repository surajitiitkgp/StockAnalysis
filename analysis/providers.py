"""Market-data providers with retry, backoff and circuit-breaking.

The app previously fetched from Yahoo (``yfinance``) with a single attempt and
swallowed every error, so any transient blip returned empty data and looked
like "no data for this stock". This module fixes that:

  - a small ``DataProvider`` interface (daily / intraday / info),
  - a ``YahooProvider`` (primary) plus optional India fallbacks (NSE India,
    Twelve Data, Alpha Vantage; Stooq only when ``USE_STOOQ=true``),
  - ``retry_with_backoff`` for transient failures,
  - a per-provider ``CircuitBreaker`` so a dead upstream fails fast instead of
    blocking every request for the full timeout.

``get_daily``/``get_intraday``/``get_info`` try providers in order and return
the first good result, logging what happened along the way.
"""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

from .config import settings
from .logging_config import get_logger
from .validation import clean_ohlcv

log = get_logger(__name__)


class ProviderError(Exception):
    """Raised when a provider cannot return data (after retries)."""


# --------------------------------------------------------------------------- #
# Retry + circuit breaker
# --------------------------------------------------------------------------- #
# HTTP statuses that will never succeed on retry (auth / not-found / bad request).
_PERMANENT_HTTP = {400, 401, 403, 404, 422}


def _is_permanent(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and exc.code in _PERMANENT_HTTP


def retry_with_backoff(fn, *, retries: int | None = None, backoff: float | None = None,
                       label: str = "call"):
    """Call ``fn`` with exponential backoff. Re-raises the last error.

    Permanent HTTP errors (401/403/404/…) are not retried — they won't
    succeed on a second attempt and only add latency and log noise.
    """
    retries = settings.fetch_retries if retries is None else retries
    backoff = settings.fetch_backoff if backoff is None else backoff
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_permanent(exc):
                raise ProviderError(f"{label} failed (permanent): {exc}") from None
            if attempt >= retries:
                break
            sleep = backoff * (2 ** (attempt - 1))
            log.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        label, attempt, retries, exc, sleep)
            time.sleep(sleep)
    raise ProviderError(f"{label} failed after {retries} attempts: {last_exc}")


class CircuitBreaker:
    """Trips open after N consecutive failures; resets after a cool-off."""

    def __init__(self, name: str, fail_threshold: int, reset_seconds: int):
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_seconds = reset_seconds
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._failures < self.fail_threshold:
                return False
            if time.time() - self._opened_at >= self.reset_seconds:
                # Half-open: allow a trial call.
                self._failures = self.fail_threshold - 1
                return False
            return True

    def record_success(self):
        with self._lock:
            self._failures = 0
            self._opened_at = 0.0

    def record_failure(self):
        with self._lock:
            self._failures += 1
            if self._failures >= self.fail_threshold:
                self._opened_at = time.time()
                log.error("circuit breaker OPEN for %s (%d consecutive failures)",
                          self.name, self._failures)


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class DataProvider:
    name = "base"
    supports_intraday = False
    supports_info = False
    # Optional providers are fallbacks: their failure is non-critical because a
    # healthy primary already satisfies every request. Used by provider_health()
    # to report a soft "limited" state instead of an alarming "degraded" one.
    optional = False

    def __init__(self):
        self.breaker = CircuitBreaker(
            self.name, settings.breaker_fail_threshold, settings.breaker_reset_seconds)

    def daily(self, ticker: str, period: str) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame()

    def info(self, ticker: str) -> dict:
        return {}


class YahooProvider(DataProvider):
    name = "yahoo"
    supports_intraday = True
    supports_info = True

    def daily(self, ticker: str, period: str) -> pd.DataFrame:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        return df

    def intraday(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        import yfinance as yf
        return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)

    def info(self, ticker: str) -> dict:
        import yfinance as yf
        return yf.Ticker(ticker).info or {}


class StooqProvider(DataProvider):
    """Daily-only fallback using Stooq's free CSV endpoint.

    Stooq uses ``<symbol>.<market>`` tickers; Indian NSE symbols map to
    ``<symbol>.ns``. No API key required.
    """

    name = "stooq"
    optional = True
    _URL = "https://stooq.com/q/d/l/?s={sym}&i=d"

    def _stooq_symbol(self, ticker: str) -> str:
        t = ticker.upper()
        if t.endswith(".NS"):
            return t[:-3].lower() + ".ns"
        if t.endswith(".BO"):
            return t[:-3].lower() + ".bo"
        return t.lower()

    def daily(self, ticker: str, period: str) -> pd.DataFrame:
        sym = self._stooq_symbol(ticker)
        url = self._URL.format(sym=sym)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=settings.fetch_timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        first = text.splitlines()[0] if text else ""
        # Stooq's free CSV endpoint sometimes returns an anti-bot HTML challenge
        # ("This site requires JavaScript to verify your browser") instead of
        # data. A no-JS client can't pass it, so surface a precise message
        # rather than a vague "no data" so the Status page is actionable.
        low = text.lower()
        if ("<html" in low or "<!doctype" in low
                or "requires javascript" in low or "enable javascript" in low):
            raise ProviderError("stooq blocked request (bot/JS challenge)")
        if not text or "Date" not in first:
            raise ProviderError("stooq returned no usable data")
        df = pd.read_csv(io.StringIO(text))
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        return df


class TwelveDataProvider(DataProvider):
    """Daily OHLCV via Twelve Data (good NSE/BSE coverage). Key-gated.

    Free tier: ~8 requests/min, 800/day. Used only as a fallback so it doesn't
    burn quota when Yahoo is healthy.
    """

    name = "twelvedata"
    optional = True

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _symbol_exchange(self, ticker: str) -> tuple[str, str | None]:
        t = ticker.upper()
        if t.endswith(".NS"):
            return t[:-3], "NSE"
        if t.endswith(".BO"):
            return t[:-3], "BSE"
        return t, None

    def daily(self, ticker: str, period: str) -> pd.DataFrame:
        sym, exchange = self._symbol_exchange(ticker)
        params = {"symbol": sym, "interval": "1day", "outputsize": "5000",
                  "apikey": self.api_key, "format": "JSON"}
        if exchange:
            params["exchange"] = exchange
        url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=settings.fetch_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            # Read the JSON error body so plan-gating (common for Indian
            # exchanges on the free tier) becomes a clear message.
            try:
                data = json.loads(exc.read().decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                raise ProviderError(f"twelvedata: HTTP {exc.code}") from None
        msg = str(data.get("message", ""))
        if "Grow or Venture plan" in msg or "upgrading" in msg.lower():
            raise ProviderError(
                "twelvedata: Indian (NSE/BSE) symbols need a paid Grow/Venture "
                "plan; not available on the free tier")
        if data.get("status") == "error" or "values" not in data:
            raise ProviderError(f"twelvedata: {data.get('message', 'no values')}")
        rows = data["values"]
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class AlphaVantageProvider(DataProvider):
    """Daily OHLCV via Alpha Vantage. Key-gated, best-effort.

    India coverage is limited (mostly BSE via a ``.BSE`` suffix) and the free
    tier allows only ~25 requests/day, so this sits last in the fallback chain.
    """

    name = "alphavantage"
    optional = True

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _av_symbol(self, ticker: str) -> str:
        t = ticker.upper()
        if t.endswith(".NS"):
            return t[:-3] + ".BSE"
        if t.endswith(".BO"):
            return t[:-3] + ".BSE"
        return t

    def daily(self, ticker: str, period: str) -> pd.DataFrame:
        params = {"function": "TIME_SERIES_DAILY", "symbol": self._av_symbol(ticker),
                  "outputsize": "full", "apikey": self.api_key}
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=settings.fetch_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        series = data.get("Time Series (Daily)")
        if not series:
            note = data.get("Note") or data.get("Information") or data.get("Error Message")
            raise ProviderError(f"alphavantage: {note or 'no data'}")
        recs = []
        for day, vals in series.items():
            recs.append({
                "datetime": day, "open": vals.get("1. open"), "high": vals.get("2. high"),
                "low": vals.get("3. low"), "close": vals.get("4. close"),
                "volume": vals.get("5. volume"),
            })
        df = pd.DataFrame(recs)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df


class NseProvider(DataProvider):
    """Daily OHLCV straight from NSE via the unofficial NseIndiaApi.

    Key-less but scrapes NSE, so it sits after Yahoo/Stooq as a
    direct-from-source fallback for NSE equities only (``.NS``). Delegates to
    :mod:`analysis.nse_client`, which handles the optional dependency, cookie
    handshake, request chunking and graceful degradation.
    """

    name = "nse"
    optional = True

    def daily(self, ticker: str, period: str) -> pd.DataFrame:
        t = ticker.upper()
        if not t.endswith(".NS"):
            raise ProviderError("nse provider only serves NSE (.NS) equities")
        from . import nse_client
        df = nse_client.equity_history(t, period)
        if df is None or df.empty:
            raise ProviderError("nse returned no data")
        return df


def _build_daily_providers() -> list:
    """Build the daily-provider chain for the Indian market.

    Order: Yahoo (primary) -> NSE India (direct-from-source) -> Twelve Data ->
    Alpha Vantage. Stooq is **excluded by default** because its NSE/BSE coverage
    is poor/unreliable (and its free endpoint now serves a bot challenge); set
    ``USE_STOOQ=true`` to re-enable it as a last-resort fallback.
    """
    chain: list[DataProvider] = [_YAHOO]
    if settings.use_nse_api:
        chain.append(NseProvider())
    if settings.twelvedata_api_key:
        chain.append(TwelveDataProvider(settings.twelvedata_api_key))
    if settings.alphavantage_api_key:
        chain.append(AlphaVantageProvider(settings.alphavantage_api_key))
    if getattr(settings, "use_stooq", False):
        chain.append(_STOOQ)
    return chain


_YAHOO = YahooProvider()
_STOOQ = StooqProvider()
_DAILY_PROVIDERS: list[DataProvider] = _build_daily_providers()


def rebuild_daily_providers() -> list:
    """Rebuild the daily-provider chain (after a runtime config change)."""
    global _DAILY_PROVIDERS
    _DAILY_PROVIDERS = _build_daily_providers()
    return _DAILY_PROVIDERS


def _period_start(period: str):
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
    return datetime.now() - timedelta(days=days + 5)


def get_daily(ticker: str, period: str = "2y") -> tuple[pd.DataFrame, dict]:
    """Fetch cleaned daily OHLCV, trying each provider in order.

    Returns ``(df, meta)`` where meta records the provider used and the
    data-quality report. ``df`` may be empty if every provider fails.
    """
    start = _period_start(period)
    for provider in _DAILY_PROVIDERS:
        if provider.breaker.is_open:
            log.warning("skipping %s for %s (breaker open)", provider.name, ticker)
            continue
        try:
            raw = retry_with_backoff(
                lambda p=provider: p.daily(ticker, period),
                label=f"{provider.name}.daily({ticker})",
            )
            provider.breaker.record_success()
            df = _standardise(raw)
            df, report = clean_ohlcv(df)
            if df.empty:
                log.info("%s returned empty/clean-empty for %s", provider.name, ticker)
                continue
            if start is not None:
                df = df[df.index >= start]
            return df, {"provider": provider.name, "quality": report.to_dict()}
        except ProviderError as exc:
            provider.breaker.record_failure()
            log.warning("%s daily failed for %s: %s", provider.name, ticker, exc)
        except Exception as exc:  # noqa: BLE001
            provider.breaker.record_failure()
            log.warning("%s daily error for %s: %s", provider.name, ticker, exc)
    return pd.DataFrame(), {"provider": None, "quality": {"ok": False, "issues": ["all_providers_failed"]}}


def get_intraday(ticker: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    if _YAHOO.breaker.is_open:
        return pd.DataFrame()
    try:
        raw = retry_with_backoff(
            lambda: _YAHOO.intraday(ticker, period, interval),
            retries=2, label=f"yahoo.intraday({ticker})",
        )
        _YAHOO.breaker.record_success()
        df, _ = clean_ohlcv(_standardise(raw))
        return df
    except Exception as exc:  # noqa: BLE001
        _YAHOO.breaker.record_failure()
        log.info("intraday unavailable for %s: %s", ticker, exc)
        return pd.DataFrame()


def get_info(ticker: str) -> dict:
    if _YAHOO.breaker.is_open:
        return {}
    wanted = [
        "longName", "shortName", "sector", "industry", "currency",
        "marketCap", "trailingPE", "forwardPE", "priceToBook",
        "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "beta", "returnOnEquity", "profitMargins", "debtToEquity",
        "earningsGrowth", "revenueGrowth", "recommendationKey",
        "targetMeanPrice", "averageVolume",
    ]
    try:
        raw = retry_with_backoff(lambda: _YAHOO.info(ticker), retries=2,
                                 label=f"yahoo.info({ticker})")
        _YAHOO.breaker.record_success()
        return {k: raw.get(k) for k in wanted if raw.get(k) is not None}
    except Exception as exc:  # noqa: BLE001
        _YAHOO.breaker.record_failure()
        log.info("info unavailable for %s: %s", ticker, exc)
        return {}


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and index tz to the app's convention."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def breaker_status() -> dict:
    """Expose breaker state for health checks."""
    return {p.name: ("open" if p.breaker.is_open else "closed") for p in _DAILY_PROVIDERS}


def provider_health(probe_ticker: str = "RELIANCE.NS") -> list[dict]:
    """Lightweight per-provider health check (Sec. 5).

    Returns one row per provider in the daily chain: its name, breaker state,
    and — for providers whose breaker is closed — a live reachability probe. The
    probe is best-effort and time-boxed by the provider timeout; a failure marks
    the provider ``degraded`` without affecting the request path.
    """
    out = []
    for p in _DAILY_PROVIDERS:
        row = {"name": p.name,
               "breaker": "open" if p.breaker.is_open else "closed",
               "optional": bool(p.optional),
               "supports_intraday": p.supports_intraday,
               "supports_info": p.supports_info}
        if p.breaker.is_open:
            row["status"] = "breaker_open"
        else:
            try:
                raw = p.daily(probe_ticker, "5d")
                df = _standardise(raw)
                row["status"] = "ok" if df is not None and not df.empty else "empty"
                row["rows"] = 0 if df is None else int(len(df))
            except Exception as exc:  # noqa: BLE001
                # A failing *optional* fallback is non-critical when a primary
                # still works, so report the softer "limited" instead of the
                # alarming "degraded" (which is reserved for the primary chain).
                row["status"] = "limited" if p.optional else "degraded"
                row["detail"] = str(exc)[:200]
        out.append(row)
    return out
