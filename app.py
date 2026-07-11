"""Flask backend for the Indian Stock Market Analysis web app.

Endpoints
---------
GET  /                      -> dashboard UI
GET  /api/stocks            -> screening universe (name + symbol)
GET  /api/search?q=         -> search the universe
GET  /api/analyze           -> full analysis (chart + signals + multi-horizon ML)
GET  /api/predict           -> multi-model / multi-horizon ML forecast
GET  /api/models            -> available ML models for the UI
GET  /api/screener          -> rank the universe for a given horizon
GET  /healthz, /readyz      -> liveness / readiness probes

Disclaimer: outputs are algorithmic, technical decision-support only and are
NOT investment advice.
"""

from __future__ import annotations

import math
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import pandas as pd
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_cors import CORS

from analysis import (
    auth,
    data_fetcher,
    indicators,
    models,
    news,
    predictor,
    runtime_config,
    security,
    store,
    strategy,
    universe,
)
from analysis.config import settings
from analysis.logging_config import get_logger
from analysis.providers import breaker_status

log = get_logger(__name__)

app = Flask(__name__)

# CORS: scope to explicit origins when provided; otherwise same-origin only.
if settings.cors_origins:
    CORS(app, origins=list(settings.cors_origins), supports_credentials=True)

_VALID_EXCHANGES = {"NSE", "BSE"}
_VALID_HORIZONS = {"intraday", "short_term", "long_term"}
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9&.\-]{1,20}$")
CHART_DAYS = 200


# --------------------------------------------------------------------------- #
# Session / auth setup
# --------------------------------------------------------------------------- #
_SECRET_PATH = os.path.join(settings.data_dir, ".secret_key")


def _load_secret_key() -> str:
    if settings.secret_key:
        return settings.secret_key
    try:
        if os.path.exists(_SECRET_PATH):
            with open(_SECRET_PATH, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        key = secrets.token_hex(32)
        os.makedirs(os.path.dirname(_SECRET_PATH), exist_ok=True)
        with open(_SECRET_PATH, "w", encoding="utf-8") as fh:
            fh.write(key)
        return key
    except Exception:  # noqa: BLE001
        log.warning("could not persist secret key; using ephemeral key", exc_info=True)
        return secrets.token_hex(32)


app.secret_key = _load_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=settings.session_cookie_secure,
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,
)
auth.ensure_default_user()
app.jinja_env.globals["csrf_token"] = security.csrf_token


# --------------------------------------------------------------------------- #
# Decorators & error handling
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def api_login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "authentication required"}), 401
        allowed, retry = security.api_limiter.check(_client_ip())
        if not allowed:
            return jsonify({"error": "rate limit exceeded", "retry_after": retry}), 429
        return view(*args, **kwargs)

    return wrapped


@app.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    return jsonify({"error": exc.message}), exc.status


@app.errorhandler(404)
def _handle_404(_):
    if request.path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404
    return redirect(url_for("index"))


@app.errorhandler(Exception)
def _handle_unexpected(exc: Exception):
    log.exception("unhandled error on %s: %s", request.path, exc)
    if request.path.startswith("/api/"):
        return jsonify({"error": "internal server error"}), 500
    return ("Internal server error", 500)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() if fwd else request.remote_addr) or "unknown"


def _req_symbol() -> str:
    symbol = request.args.get("symbol", "").strip().upper()
    if not symbol:
        raise ApiError("symbol is required", 400)
    if not _SYMBOL_RE.match(symbol):
        raise ApiError("invalid symbol", 400)
    return symbol


def _req_exchange() -> str:
    ex = request.args.get("exchange", "NSE").strip().upper()
    if ex not in _VALID_EXCHANGES:
        raise ApiError(f"invalid exchange (expected one of {sorted(_VALID_EXCHANGES)})", 400)
    return ex


def _req_model() -> str:
    m = request.args.get("model", "auto").strip().lower()
    if m != "auto" and m not in models.model_keys():
        raise ApiError("invalid model", 400)
    return m


def _req_horizons():
    raw = request.args.get("horizons", "").strip()
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and int(part) in predictor.HORIZONS:
            out.append(int(part))
    return out or None


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def _clean_float(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def _chart_payload(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    tail = df.tail(CHART_DAYS)
    dates = [d.strftime("%Y-%m-%d") for d in tail.index]

    def col(name):
        if name not in tail.columns:
            return [None] * len(tail)
        return [_clean_float(v) for v in tail[name].tolist()]

    return {
        "dates": dates,
        "open": col("Open"), "high": col("High"), "low": col("Low"),
        "close": col("Close"), "volume": col("Volume"),
        "sma20": col("SMA20"), "sma50": col("SMA50"), "sma200": col("SMA200"),
        "ema9": col("EMA9"), "ema21": col("EMA21"),
        "bb_upper": col("BB_UPPER"), "bb_lower": col("BB_LOWER"),
        "rsi": col("RSI14"), "macd": col("MACD"),
        "macd_signal": col("MACD_SIGNAL"), "macd_hist": col("MACD_HIST"),
    }


def _info_payload(stock: data_fetcher.StockData, df: pd.DataFrame) -> dict:
    info = dict(stock.info)
    last = df.iloc[-1] if df is not None and not df.empty else None
    prev = df.iloc[-2] if df is not None and len(df) > 1 else None
    price = _clean_float(last["Close"]) if last is not None else None
    change = change_pct = None
    if last is not None and prev is not None:
        change = _clean_float(last["Close"] - prev["Close"])
        if prev["Close"]:
            change_pct = _clean_float((last["Close"] - prev["Close"]) / prev["Close"] * 100)
    return {
        "name": stock.name, "symbol": stock.symbol, "exchange": stock.exchange,
        "ticker": stock.yahoo_ticker, "price": price,
        "change": change, "change_pct": change_pct,
        "sector": info.get("sector"), "industry": info.get("industry"),
        "market_cap": _clean_float(info.get("marketCap")),
        "pe": _clean_float(info.get("trailingPE")),
        "pb": _clean_float(info.get("priceToBook")),
        "dividend_yield": _clean_float(info.get("dividendYield")),
        "week52_high": _clean_float(info.get("fiftyTwoWeekHigh")),
        "week52_low": _clean_float(info.get("fiftyTwoWeekLow")),
        "beta": _clean_float(info.get("beta")),
        "roe": _clean_float(info.get("returnOnEquity")),
    }


# --------------------------------------------------------------------------- #
# Auth routes
# --------------------------------------------------------------------------- #
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("index"))

    error = None
    mode = request.form.get("mode", "login")
    if request.method == "POST":
        if not security.validate_csrf(request.form.get("csrf_token")):
            error = "Session expired, please try again."
            return render_template("login.html", error=error, mode=mode), 400

        allowed, retry = security.login_limiter.check(_client_ip())
        if not allowed:
            error = f"Too many attempts. Try again in {retry}s."
            return render_template("login.html", error=error, mode=mode), 429

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if mode == "register":
            confirm = request.form.get("confirm", "")
            if password != confirm:
                error = "Passwords do not match."
            else:
                ok, msg = auth.create_user(username, password)
                if ok:
                    session["user"] = username.lower()
                    security.login_limiter.reset(_client_ip())
                    return redirect(request.args.get("next") or url_for("index"))
                error = msg
        else:
            if auth.verify(username, password):
                session["user"] = username.lower()
                security.login_limiter.reset(_client_ip())
                return redirect(request.args.get("next") or url_for("index"))
            error = "Invalid username or password."

    return render_template("login.html", error=error, mode=mode)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=session.get("user"),
                           stock_count=universe.count())


# --------------------------------------------------------------------------- #
# Health probes (unauthenticated)
# --------------------------------------------------------------------------- #
@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "time": int(time.time())})


@app.route("/readyz")
def readyz():
    try:
        db_ok = True
        db_stats = store.stats()
    except Exception:  # noqa: BLE001
        db_ok = False
        db_stats = {}
    ready = db_ok
    return (jsonify({
        "ready": ready,
        "universe": universe.count(),
        "store": db_stats,
        "providers": breaker_status(),
        "news": news.status(),
    }), 200 if ready else 503)


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #
@app.route("/api/models")
@api_login_required
def api_models():
    return jsonify({"models": models.available_models(), "horizons": list(predictor.HORIZONS)})


@app.route("/api/config", methods=["GET"])
@api_login_required
def api_config_get():
    return jsonify({"settings": runtime_config.current_values(settings)})


@app.route("/api/config", methods=["POST"])
@api_login_required
def api_config_post():
    # State-changing endpoint: require a valid CSRF token from the session.
    token = request.headers.get("X-CSRF-Token") or (request.get_json(silent=True) or {}).get("csrf_token")
    if not security.validate_csrf(token):
        raise ApiError("invalid or missing CSRF token", 403)
    payload = request.get_json(silent=True) or {}
    changes = payload.get("changes", payload)
    if not isinstance(changes, dict):
        raise ApiError("invalid payload", 400)
    changes = {k: v for k, v in changes.items() if k != "csrf_token"}
    try:
        result = runtime_config.update(settings, changes)
    except runtime_config.ConfigError as exc:
        raise ApiError(str(exc), 400)
    log.info("runtime config updated by %s: %s", session.get("user"), result["applied"])
    return jsonify(result)


@app.route("/api/stocks")
@api_login_required
def api_stocks():
    return jsonify(universe.all_symbols())


@app.route("/api/search")
@api_login_required
def api_search():
    q = request.args.get("q", "")
    if not q:
        return jsonify(universe.all_symbols()[:15])
    return jsonify(universe.search(q))


@app.route("/api/analyze")
@api_login_required
def api_analyze():
    symbol = _req_symbol()
    exchange = _req_exchange()
    model = _req_model()
    try:
        stock = data_fetcher.load_stock(symbol, exchange, with_intraday=True, with_info=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("load_stock failed for %s/%s: %s", symbol, exchange, exc)
        raise ApiError(f"failed to fetch data: {exc}", 502)

    if stock.history is None or stock.history.empty:
        raise ApiError(f"no data found for {symbol} on {exchange}", 404)

    df = indicators.add_daily_indicators(stock.history)
    recommendations = strategy.analyze(stock)
    try:
        prediction = predictor.predict(symbol, exchange, model=model)
    except Exception as exc:  # noqa: BLE001 - ML must never break the analysis
        log.warning("prediction failed for %s: %s", symbol, exc)
        prediction = {"available": False, "reason": f"model error: {exc}", "verdict": "HOLD"}

    try:
        news_summary = news.get_sentiment_summary(symbol, exchange, stock.name)
    except Exception as exc:  # noqa: BLE001
        log.info("news summary failed for %s: %s", symbol, exc)
        news_summary = {"available": False, "reason": "news lookup failed"}

    return jsonify({
        "info": _info_payload(stock, df),
        "chart": _chart_payload(df),
        "recommendations": recommendations,
        "prediction": prediction,
        "news": news_summary,
        "data": stock.meta,
    })


@app.route("/api/news")
@api_login_required
def api_news():
    scope = request.args.get("scope", "company").strip().lower()
    if scope == "market":
        return jsonify(news.get_market_news())
    symbol = _req_symbol()
    exchange = _req_exchange()
    company = universe.SYMBOL_TO_NAME.get(symbol, symbol)
    return jsonify(news.get_sentiment_summary(symbol, exchange, company))


@app.route("/api/predict")
@api_login_required
def api_predict():
    symbol = _req_symbol()
    exchange = _req_exchange()
    model = _req_model()
    horizons = _req_horizons()
    try:
        return jsonify(predictor.predict(symbol, exchange, model=model, horizons=horizons))
    except Exception as exc:  # noqa: BLE001
        log.warning("predict failed for %s: %s", symbol, exc)
        raise ApiError(f"prediction failed: {exc}", 502)


def _screen_one(item: dict, exchange: str, horizon: str):
    try:
        stock = data_fetcher.load_stock(
            item["symbol"], exchange,
            with_intraday=(horizon == "intraday"),
            with_info=(horizon == "long_term"),
        )
        if stock.history is None or stock.history.empty:
            return None
        rec_fn = {
            "intraday": strategy.intraday,
            "short_term": strategy.short_term,
            "long_term": strategy.long_term,
        }[horizon]
        rec = rec_fn(stock)
        last = stock.history["Close"].iloc[-1]
        prev = stock.history["Close"].iloc[-2] if len(stock.history) > 1 else last
        return {
            "name": item["name"], "symbol": item["symbol"], "exchange": exchange,
            "price": _clean_float(last),
            "change_pct": _clean_float((last - prev) / prev * 100) if prev else None,
            "verdict": rec.verdict, "score": rec.score, "confidence": rec.confidence,
            "entry": rec.entry, "target": rec.target, "stop_loss": rec.stop_loss,
            "risk_reward": rec.risk_reward,
            "top_signals": [
                {"label": s.label, "direction": s.direction, "detail": s.detail}
                for s in rec.signals[:4]
            ],
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("screen failed for %s: %s", item.get("symbol"), exc)
        return None


@app.route("/api/screener")
@api_login_required
def api_screener():
    horizon = request.args.get("horizon", "short_term").strip()
    exchange = _req_exchange()
    if horizon not in _VALID_HORIZONS:
        raise ApiError("invalid horizon", 400)

    try:
        limit = int(request.args.get("limit", 150))
    except (TypeError, ValueError):
        limit = 150
    limit = max(10, min(limit, universe.count()))

    stocks = universe.screener_symbols(limit)
    results = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_screen_one, item, exchange, horizon) for item in stocks]
        for fut in futures:
            r = fut.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda r: (r["score"] if r["score"] is not None else -999), reverse=True)
    buys = [r for r in results if r["verdict"] in ("BUY", "STRONG BUY")]
    sells = [r for r in results if r["verdict"] in ("SELL", "STRONG SELL")]
    return jsonify({
        "horizon": horizon, "exchange": exchange,
        "scanned": len(stocks), "count": len(results),
        "top_buys": buys[:12],
        "top_sells": sells[-12:][::-1] if sells else [],
        "all": results,
    })


if __name__ == "__main__":
    log.info("Starting Bharat Stocks on %s:%s (debug=%s)",
             settings.host, settings.port, settings.debug)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
