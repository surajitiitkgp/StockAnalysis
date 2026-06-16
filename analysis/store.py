"""Local historical price store backed by SQLite.

A single file (``analysis/data/history.db``) holds daily OHLCV bars for the whole
NSE universe. This lets the app serve history instantly and work offline, and
gives the prediction model a stable training set without hammering Yahoo.

Schema
------
prices(symbol, date, open, high, low, close, volume)  PRIMARY KEY(symbol, date)
"""

from __future__ import annotations

import os
import sqlite3
import threading

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(_DATA_DIR, "history.db")
_local = threading.local()


def _conn() -> sqlite3.Connection:
    """Thread-local connection (SQLite connections aren't shareable across threads)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        _local.conn = conn
        _init(conn)
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol);")
    conn.commit()


def init_db() -> None:
    _conn()


def upsert_history(symbol: str, df: pd.DataFrame) -> int:
    """Insert/replace OHLCV rows for a symbol. Returns number of rows written."""
    if df is None or df.empty:
        return 0
    symbol = symbol.upper().strip()
    rows = []
    for ts, r in df.iterrows():
        date = pd.Timestamp(ts).strftime("%Y-%m-%d")
        rows.append((
            symbol, date,
            _num(r.get("Open")), _num(r.get("High")), _num(r.get("Low")),
            _num(r.get("Close")), _num(r.get("Volume")),
        ))
    if not rows:
        return 0
    conn = _conn()
    conn.executemany(
        "INSERT OR REPLACE INTO prices (symbol, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def _num(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def get_history(symbol: str, start: str | None = None) -> pd.DataFrame:
    """Return stored daily OHLCV for a symbol as a DatetimeIndex frame."""
    symbol = symbol.upper().strip()
    conn = _conn()
    q = "SELECT date, open, high, low, close, volume FROM prices WHERE symbol = ?"
    params: list = [symbol]
    if start:
        q += " AND date >= ?"
        params.append(start)
    q += " ORDER BY date ASC"
    df = pd.read_sql_query(q, conn, params=params)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.dropna(subset=["Close"])
    return df


def last_date(symbol: str) -> str | None:
    symbol = symbol.upper().strip()
    cur = _conn().execute("SELECT MAX(date) FROM prices WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def symbols_present() -> set[str]:
    cur = _conn().execute("SELECT DISTINCT symbol FROM prices")
    return {r[0] for r in cur.fetchall()}


def row_count(symbol: str | None = None) -> int:
    conn = _conn()
    if symbol:
        cur = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", (symbol.upper().strip(),))
    else:
        cur = conn.execute("SELECT COUNT(*) FROM prices")
    return int(cur.fetchone()[0])


def stats() -> dict:
    conn = _conn()
    nsym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
    nrows = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    rng = conn.execute("SELECT MIN(date), MAX(date) FROM prices").fetchone()
    return {"symbols": int(nsym), "rows": int(nrows),
            "min_date": rng[0], "max_date": rng[1]}


def exists() -> bool:
    """True if the DB file exists and has data."""
    return os.path.exists(DB_PATH) and row_count() > 0
