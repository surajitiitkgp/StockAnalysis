"""Universe of Indian stocks (NSE) used for screening.

Symbols use the yfinance convention:
  - NSE: ``<SYMBOL>.NS``
  - BSE: ``<SYMBOL>.BO``

The default screening universe is the Nifty 50 plus a handful of other
liquid, popular names. You can extend ``STOCKS`` freely.
"""

from __future__ import annotations

# name -> NSE trading symbol (without the .NS suffix)
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

# A few additional popular / high-beta names useful for intraday screening.
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

# Master mapping: display name -> base symbol.
STOCKS = {**NIFTY_50, **EXTRA}

# Convenience reverse lookup: base symbol -> display name.
SYMBOL_TO_NAME = {v: k for k, v in STOCKS.items()}


def to_yahoo(symbol: str, exchange: str = "NSE") -> str:
    """Return the yfinance ticker for a base symbol on a given exchange."""
    symbol = symbol.upper().strip()
    # Already suffixed.
    if symbol.endswith(".NS") or symbol.endswith(".BO"):
        return symbol
    suffix = ".BO" if exchange.upper() == "BSE" else ".NS"
    return f"{symbol}{suffix}"


def search(query: str, limit: int = 15):
    """Search the universe by display name or symbol (case-insensitive)."""
    q = query.lower().strip()
    results = []
    for name, sym in STOCKS.items():
        if q in name.lower() or q in sym.lower():
            results.append({"name": name, "symbol": sym})
        if len(results) >= limit:
            break
    return results


def all_symbols():
    """Return the full screening universe as a list of dicts."""
    return [{"name": name, "symbol": sym} for name, sym in STOCKS.items()]
