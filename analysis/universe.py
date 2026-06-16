"""Universe of Indian stocks used for search and screening.

Symbols use the yfinance convention:
  - NSE: ``<SYMBOL>.NS``
  - BSE: ``<SYMBOL>.BO``

The full universe is the **complete list of NSE-listed equities** (~2,000+ names),
loaded from the official NSE equity master (``EQUITY_L.csv``). Loading strategy:

  1. Try to refresh from the live NSE archive URL (cached on disk).
  2. Fall back to the bundled snapshot shipped in ``analysis/data/EQUITY_L.csv``.
  3. Fall back to a small hardcoded Nifty 50 list if everything else fails.

This guarantees the app always has the *entire* Indian (NSE) stock list available
for search, while still working fully offline.
"""

from __future__ import annotations

import csv
import io
import os
import time
import urllib.request

# --------------------------------------------------------------------------- #
# Hardcoded fallback (Nifty 50 + a few popular names). Used only if both the
# live download and the bundled CSV are unavailable.
# --------------------------------------------------------------------------- #
NIFTY_50 = {
    "Reliance Industries": "RELIANCE",
    "Tata Consultancy Services": "TCS",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "Infosys": "INFY",
    "Hindustan Unilever": "HINDUNILVR",
    "ITC": "ITC",
    "State Bank of India": "SBIN",
    "Bharti Airtel": "BHARTIARTL",
    "Bajaj Finance": "BAJFINANCE",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Larsen & Toubro": "LT",
    "Axis Bank": "AXISBANK",
    "Asian Paints": "ASIANPAINT",
    "Maruti Suzuki": "MARUTI",
    "HCL Technologies": "HCLTECH",
    "Sun Pharma": "SUNPHARMA",
    "Titan Company": "TITAN",
    "Wipro": "WIPRO",
    "UltraTech Cement": "ULTRACEMCO",
    "Nestle India": "NESTLEIND",
    "Power Grid Corp": "POWERGRID",
    "NTPC": "NTPC",
    "Tata Motors": "TATAMOTORS",
    "Tata Steel": "TATASTEEL",
    "Mahindra & Mahindra": "M&M",
    "JSW Steel": "JSWSTEEL",
    "Bajaj Finserv": "BAJAJFINSV",
    "Adani Enterprises": "ADANIENT",
    "Adani Ports": "ADANIPORTS",
    "Coal India": "COALINDIA",
    "Hindalco Industries": "HINDALCO",
    "Oil & Natural Gas Corp": "ONGC",
    "Grasim Industries": "GRASIM",
    "Tech Mahindra": "TECHM",
    "Dr Reddy's Labs": "DRREDDY",
    "Cipla": "CIPLA",
    "Eicher Motors": "EICHERMOT",
    "Britannia Industries": "BRITANNIA",
    "Apollo Hospitals": "APOLLOHOSP",
    "Divi's Laboratories": "DIVISLAB",
    "Hero MotoCorp": "HEROMOTOCO",
    "Bajaj Auto": "BAJAJ-AUTO",
    "SBI Life Insurance": "SBILIFE",
    "HDFC Life Insurance": "HDFCLIFE",
    "Tata Consumer Products": "TATACONSUM",
    "Indusind Bank": "INDUSINDBK",
    "Shriram Finance": "SHRIRAMFIN",
    "LTIMindtree": "LTIM",
    "Bharat Petroleum": "BPCL",
}

EXTRA = {
    "Zomato": "ZOMATO",
    "Paytm (One97)": "PAYTM",
    "IRCTC": "IRCTC",
    "Vedanta": "VEDL",
    "DLF": "DLF",
    "Trent": "TRENT",
    "Bank of Baroda": "BANKBARODA",
    "Punjab National Bank": "PNB",
    "GAIL": "GAIL",
    "Indian Oil Corp": "IOC",
}

# Symbols (without suffix) that are most liquid / popular. They are scanned
# first by the screener so a default scan covers the headline names.
PRIORITY_SYMBOLS = list(NIFTY_50.values()) + list(EXTRA.values())

# --------------------------------------------------------------------------- #
# Full-universe loading
# --------------------------------------------------------------------------- #
_NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CSV_PATH = os.path.join(_DATA_DIR, "EQUITY_L.csv")
# Refresh the on-disk snapshot at most once every 7 days.
_MAX_AGE_SECONDS = 7 * 24 * 3600
_DOWNLOAD_TIMEOUT = 20

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _titlecase_company(name: str) -> str:
    """Tidy NSE company names a little for display."""
    name = name.strip()
    # Drop the trailing " Limited" / " Ltd" / " Ltd." noise for compactness.
    for suffix in (" Limited", " Ltd.", " Ltd"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.strip()


def _parse_equity_csv(text: str) -> dict:
    """Parse an NSE EQUITY_L.csv payload into {display_name: symbol}."""
    stocks: dict[str, str] = {}
    reader = csv.DictReader(io.StringIO(text))
    # NSE headers have leading spaces, e.g. " SERIES". Normalise keys.
    for row in reader:
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        symbol = clean.get("SYMBOL", "")
        name = clean.get("NAME OF COMPANY", "")
        series = clean.get("SERIES", "")
        if not symbol or not name:
            continue
        # Keep regular equity series only (EQ, BE, BZ, etc. are tradable equities).
        if series and series not in ("EQ", "BE", "BZ", "SM", "ST"):
            continue
        display = _titlecase_company(name)
        if display in stocks:
            display = f"{display} ({symbol})"
        stocks[display] = symbol
    return stocks


def _download_nse_list() -> str | None:
    """Fetch the live NSE equity master CSV. Returns text or None on failure."""
    try:
        req = urllib.request.Request(_NSE_URL, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read().decode("utf-8", errors="replace")
        if "SYMBOL" in data and "NAME OF COMPANY" in data:
            return data
    except Exception:  # noqa: BLE001 - any network/parse error -> fall back
        return None
    return None


def _refresh_snapshot_if_stale() -> None:
    """Download a fresh CSV to disk if the cached copy is missing or old."""
    try:
        fresh_enough = (
            os.path.exists(_CSV_PATH)
            and (time.time() - os.path.getmtime(_CSV_PATH)) < _MAX_AGE_SECONDS
        )
        if fresh_enough:
            return
        data = _download_nse_list()
        if data:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_CSV_PATH, "w", encoding="utf-8") as fh:
                fh.write(data)
    except Exception:  # noqa: BLE001
        pass


def _load_universe() -> dict:
    """Build the full {display_name: symbol} universe with graceful fallbacks."""
    _refresh_snapshot_if_stale()
    # 1) Bundled / cached snapshot on disk.
    if os.path.exists(_CSV_PATH):
        try:
            with open(_CSV_PATH, "r", encoding="utf-8") as fh:
                parsed = _parse_equity_csv(fh.read())
            if parsed:
                return parsed
        except Exception:  # noqa: BLE001
            pass
    # 2) Last resort: hardcoded names.
    return {**NIFTY_50, **EXTRA}


# Master mapping: display name -> base symbol (entire NSE universe).
STOCKS = _load_universe()

# Convenience reverse lookup: base symbol -> display name.
SYMBOL_TO_NAME = {v: k for k, v in STOCKS.items()}


def reload_universe() -> int:
    """Force a reload of the universe from disk/live. Returns the stock count."""
    global STOCKS, SYMBOL_TO_NAME
    STOCKS = _load_universe()
    SYMBOL_TO_NAME = {v: k for k, v in STOCKS.items()}
    return len(STOCKS)


def to_yahoo(symbol: str, exchange: str = "NSE") -> str:
    """Return the yfinance ticker for a base symbol on a given exchange."""
    symbol = symbol.upper().strip()
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    suffix = ".BO" if exchange.upper() == "BSE" else ".NS"
    return f"{symbol}{suffix}"


def search(query: str, limit: int = 20):
    """Search the universe by display name or symbol (case-insensitive).

    Results are ranked so the most relevant matches surface first:
      1. exact symbol match
      2. symbol starts with the query
      3. name starts with the query
      4. symbol/name contains the query anywhere
    """
    q = query.lower().strip()
    if not q:
        return all_symbols()[:limit]

    exact, sym_prefix, name_prefix, contains = [], [], [], []
    for name, sym in STOCKS.items():
        sym_l = sym.lower()
        name_l = name.lower()
        item = {"name": name, "symbol": sym}
        if sym_l == q:
            exact.append(item)
        elif sym_l.startswith(q):
            sym_prefix.append(item)
        elif name_l.startswith(q):
            name_prefix.append(item)
        elif q in sym_l or q in name_l:
            contains.append(item)
    return (exact + sym_prefix + name_prefix + contains)[:limit]


def all_symbols():
    """Return the full screening universe as a list of dicts."""
    return [{"name": name, "symbol": sym} for name, sym in STOCKS.items()]


def screener_symbols(limit: int | None = None):
    """Symbols for the screener, most-liquid/popular names first.

    Scanning the entire NSE universe live would be extremely slow and prone to
    rate limiting, so the screener scans a prioritised slice. ``limit`` caps how
    many symbols are scanned (``None`` = the full universe).
    """
    ordered: list[dict] = []
    seen: set[str] = set()
    for sym in PRIORITY_SYMBOLS:
        name = SYMBOL_TO_NAME.get(sym)
        if name and sym not in seen:
            ordered.append({"name": name, "symbol": sym})
            seen.add(sym)
    for name, sym in STOCKS.items():
        if sym not in seen:
            ordered.append({"name": name, "symbol": sym})
            seen.add(sym)
    return ordered[:limit] if limit else ordered


def count() -> int:
    """Total number of stocks in the universe."""
    return len(STOCKS)
