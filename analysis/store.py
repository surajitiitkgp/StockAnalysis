"""Local historical price store backed by SQLite.

A single file (``analysis/data/history.db``) holds daily OHLCV bars for the whole
NSE universe. This lets the app serve history instantly and work offline, and
gives the prediction model a stable training set without hammering Yahoo.

Schema
------
prices(symbol, date, open, high, low, close, volume)  PRIMARY KEY(symbol, date)
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading

import pandas as pd

from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)

_DATA_DIR = settings.data_dir
DB_PATH = settings.db_path
_local = threading.local()

# Bump when the schema changes; migrations run in ``_migrate``.
SCHEMA_VERSION = 2


def _conn() -> sqlite3.Connection:
    """Thread-local connection (SQLite connections aren't shareable across threads)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        _local.conn = conn
        _init(conn)
        _migrate(conn)
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
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    # Self-accumulating daily news-sentiment archive. ``symbol`` may be a real
    # ticker or the pseudo-symbol ``__MARKET__`` for market/geopolitical news.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_daily (
            symbol    TEXT NOT NULL,
            date      TEXT NOT NULL,
            sentiment REAL,
            article_count INTEGER,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sent_symbol ON sentiment_daily(symbol);")
    _init_v2(conn)
    conn.commit()


def _init_v2(conn: sqlite3.Connection) -> None:
    """Prediction audit trail + data-source health/sync tables (schema v2)."""
    # Every served prediction is logged with the data, model, feature and
    # timing metadata behind it, so results are auditable and reproducible.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT NOT NULL,
            exchange      TEXT NOT NULL,
            horizon_days  INTEGER NOT NULL,
            created_at    TEXT NOT NULL,
            last_data_date TEXT,
            prediction_mode TEXT,
            model_key     TEXT,
            verdict       TEXT,
            forecast_price REAL,
            expected_return_pct REAL,
            confidence    REAL,
            directional_accuracy_pct REAL,
            raw_observations INTEGER,
            feature_observations INTEGER,
            data_quality  TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol, horizon_days);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);")
    # Per-provider / per-job synchronisation health so a status page can show
    # last success, last error and staleness at a glance.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_health (
            source        TEXT NOT NULL,
            kind          TEXT NOT NULL,
            status        TEXT,
            detail        TEXT,
            symbols_ok    INTEGER,
            symbols_failed INTEGER,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (source, kind)
        )
        """
    )


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply forward-only schema migrations based on the stored version."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    current = int(row[0]) if row and row[0] else 0
    if current == 0:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        current = SCHEMA_VERSION
    if current < 2:
        # Additive migration: create the v2 tables on pre-existing databases.
        _init_v2(conn)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        current = 2
    if current != SCHEMA_VERSION:
        log.warning("store schema version %s != expected %s", current, SCHEMA_VERSION)


def reset_connection() -> None:
    """Close and forget the thread-local connection (e.g. after DB_PATH changes)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        _local.conn = None


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


# --------------------------------------------------------------------------- #
# News-sentiment archive
# --------------------------------------------------------------------------- #
MARKET_SYMBOL = "__MARKET__"


def upsert_sentiment(symbol: str, series: list) -> int:
    """Insert/replace daily sentiment rows.

    ``series`` is a list of ``{"date", "sentiment", "count"}`` dicts. Existing
    days are overwritten (the freshest fetch wins), so re-running the refresh
    job keeps the archive current and progressively deeper.
    """
    if not series:
        return 0
    symbol = symbol.upper().strip()
    rows = [(symbol, s["date"], _num(s.get("sentiment")), int(s.get("count") or 0))
            for s in series if s.get("date")]
    if not rows:
        return 0
    conn = _conn()
    conn.executemany(
        "INSERT OR REPLACE INTO sentiment_daily (symbol, date, sentiment, article_count) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def get_sentiment(symbol: str, start: str | None = None) -> pd.DataFrame:
    """Return stored daily sentiment for a symbol as a DatetimeIndex frame."""
    symbol = symbol.upper().strip()
    conn = _conn()
    q = "SELECT date, sentiment, article_count FROM sentiment_daily WHERE symbol = ?"
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
    df.columns = ["sentiment", "article_count"]
    return df


def sentiment_stats() -> dict:
    conn = _conn()
    try:
        nsym = conn.execute("SELECT COUNT(DISTINCT symbol) FROM sentiment_daily").fetchone()[0]
        nrows = conn.execute("SELECT COUNT(*) FROM sentiment_daily").fetchone()[0]
        rng = conn.execute("SELECT MIN(date), MAX(date) FROM sentiment_daily").fetchone()
    except Exception:  # noqa: BLE001
        return {"symbols": 0, "rows": 0, "min_date": None, "max_date": None}
    return {"symbols": int(nsym), "rows": int(nrows),
            "min_date": rng[0], "max_date": rng[1]}


# --------------------------------------------------------------------------- #
# Prediction audit trail (schema v2)
# --------------------------------------------------------------------------- #
def log_prediction(result: dict) -> int:
    """Persist a served prediction with its full provenance (best-effort).

    Accepts the dict returned by :func:`analysis.predictor.predict` and writes
    one row per horizon. Never raises — a logging failure must not break the
    prediction response. Returns the number of rows written.
    """
    if not result or not result.get("available"):
        return 0
    symbol = str(result.get("symbol", "")).upper().strip()
    if not symbol:
        return 0
    exchange = str(result.get("exchange", "NSE")).upper()
    created_at = result.get("trained_at") or __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
    last_date = result.get("last_date")
    mode = result.get("prediction_mode")
    model_key = (result.get("model") or {}).get("key")
    dq = result.get("data_quality") or {}
    raw_obs = dq.get("raw_observations") or result.get("history_days")
    feat_obs = dq.get("feature_ready_observations") or result.get("train_samples")
    dq_json = json.dumps(dq) if dq else None

    rows = []
    for h in result.get("horizons", []):
        metrics = h.get("metrics") or {}
        rows.append((
            symbol, exchange, int(h.get("days", 0)), created_at, last_date, mode,
            model_key, h.get("verdict"), _num(h.get("forecast_price")),
            _num(h.get("expected_return_pct")), _num(h.get("confidence")),
            _num(metrics.get("directional_accuracy_pct")),
            int(raw_obs) if raw_obs is not None else None,
            int(feat_obs) if feat_obs is not None else None,
            dq_json,
        ))
    if not rows:
        return 0
    try:
        conn = _conn()
        conn.executemany(
            "INSERT INTO predictions (symbol, exchange, horizon_days, created_at, "
            "last_data_date, prediction_mode, model_key, verdict, forecast_price, "
            "expected_return_pct, confidence, directional_accuracy_pct, "
            "raw_observations, feature_observations, data_quality) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return len(rows)
    except Exception:  # noqa: BLE001
        log.warning("failed to log prediction for %s", symbol, exc_info=True)
        return 0


def recent_predictions(symbol: str | None = None, limit: int = 50) -> list[dict]:
    """Return recent prediction audit rows (newest first)."""
    conn = _conn()
    q = ("SELECT symbol, exchange, horizon_days, created_at, last_data_date, "
         "prediction_mode, model_key, verdict, forecast_price, expected_return_pct, "
         "confidence, directional_accuracy_pct, raw_observations, feature_observations "
         "FROM predictions")
    params: list = []
    if symbol:
        q += " WHERE symbol = ?"
        params.append(symbol.upper().strip())
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    cols = ["symbol", "exchange", "horizon_days", "created_at", "last_data_date",
            "prediction_mode", "model_key", "verdict", "forecast_price",
            "expected_return_pct", "confidence", "directional_accuracy_pct",
            "raw_observations", "feature_observations"]
    try:
        cur = conn.execute(q, params)
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:  # noqa: BLE001
        return []


def prediction_stats() -> dict:
    conn = _conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        last = conn.execute("SELECT MAX(created_at) FROM predictions").fetchone()[0]
        by_mode = {(r[0] or "unknown"): int(r[1]) for r in conn.execute(
            "SELECT prediction_mode, COUNT(*) FROM predictions GROUP BY prediction_mode").fetchall()}
    except Exception:  # noqa: BLE001
        return {"total": 0, "last_created": None, "by_mode": {}}
    return {"total": int(total), "last_created": last, "by_mode": by_mode}


# --------------------------------------------------------------------------- #
# Data-source health / synchronisation status (schema v2)
# --------------------------------------------------------------------------- #
def record_sync(source: str, kind: str, status: str, detail: str | None = None,
                symbols_ok: int = 0, symbols_failed: int = 0) -> None:
    """Upsert a sync-health row for a provider/job (best-effort)."""
    ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO sync_health "
            "(source, kind, status, detail, symbols_ok, symbols_failed, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source, kind, status, detail, int(symbols_ok), int(symbols_failed), ts),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        log.warning("failed to record sync health for %s/%s", source, kind, exc_info=True)


def sync_health() -> list[dict]:
    """Return all sync-health rows (for the status page / readiness probe)."""
    conn = _conn()
    cols = ["source", "kind", "status", "detail", "symbols_ok", "symbols_failed", "updated_at"]
    try:
        cur = conn.execute(
            "SELECT source, kind, status, detail, symbols_ok, symbols_failed, updated_at "
            "FROM sync_health ORDER BY updated_at DESC")
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:  # noqa: BLE001
        return []
