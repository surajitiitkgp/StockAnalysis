"""Flask backend for the Indian Stock Market Analysis web app.

Endpoints
---------
GET  /                      -> dashboard UI
GET  /api/stocks            -> screening universe (name + symbol)
GET  /api/search?q=         -> search the universe
GET  /api/analyze           -> full analysis for one stock (chart + signals)
GET  /api/screener          -> rank the universe for a given horizon

Disclaimer: outputs are algorithmic, technical decision-support only and are
NOT investment advice.
"""

from __future__ import annotations

import math
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import numpy as np
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

from analysis import auth, data_fetcher, indicators, predictor, strategy, universe

app = Flask(__name__)
CORS(app)

# --- Session / auth setup ------------------------------------------------- #
# A stable secret key keeps users logged in across restarts. Override in
# production with the SECRET_KEY env var.
_SECRET_PATH = os.path.join(os.path.dirname(__file__), "analysis", "data", ".secret_key")


def _load_secret_key() -> str:
    env = os.environ.get("SECRET_KEY")
    if env:
        return env
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
        return secrets.token_hex(32)


app.secret_key = _load_secret_key()
auth.ensure_default_user()


def login_required(view):
    """Protect page routes — redirect to the login screen when logged out."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def api_login_required(view):
    """Protect API routes — return 401 JSON when logged out."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return jsonify({"error": "authentication required"}), 401
        return view(*args, **kwargs)

    return wrapped


CHART_DAYS = 200  # how many recent daily bars to send to the chart


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
    """Serialise the tail of an indicator-augmented daily frame for plotting."""
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
        "open": col("Open"),
        "high": col("High"),
        "low": col("Low"),
        "close": col("Close"),
        "volume": col("Volume"),
        "sma20": col("SMA20"),
        "sma50": col("SMA50"),
        "sma200": col("SMA200"),
        "ema9": col("EMA9"),
        "ema21": col("EMA21"),
        "bb_upper": col("BB_UPPER"),
        "bb_lower": col("BB_LOWER"),
        "rsi": col("RSI14"),
        "macd": col("MACD"),
        "macd_signal": col("MACD_SIGNAL"),
        "macd_hist": col("MACD_HIST"),
    }


def _info_payload(stock: data_fetcher.StockData, df: pd.DataFrame) -> dict:
    info = dict(stock.info)
    last = df.iloc[-1] if df is not None and not df.empty else None
    prev = df.iloc[-2] if df is not None and len(df) > 1 else None
    price = _clean_float(last["Close"]) if last is not None else None
    change = None
    change_pct = None
    if last is not None and prev is not None:
        change = _clean_float(last["Close"] - prev["Close"])
        if prev["Close"]:
            change_pct = _clean_float((last["Close"] - prev["Close"]) / prev["Close"] * 100)
    return {
        "name": stock.name,
        "symbol": stock.symbol,
        "exchange": stock.exchange,
        "ticker": stock.yahoo_ticker,
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": _clean_float(info.get("marketCap")),
        "pe": _clean_float(info.get("trailingPE")),
        "pb": _clean_float(info.get("priceToBook")),
        "dividend_yield": _clean_float(info.get("dividendYield")),
        "week52_high": _clean_float(info.get("fiftyTwoWeekHigh")),
        "week52_low": _clean_float(info.get("fiftyTwoWeekLow")),
        "beta": _clean_float(info.get("beta")),
        "roe": _clean_float(info.get("returnOnEquity")),
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("index"))

    error = None
    mode = request.form.get("mode", "login")
    if request.method == "POST":
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
                    return redirect(request.args.get("next") or url_for("index"))
                error = msg
        else:
            if auth.verify(username, password):
                session["user"] = username.lower()
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
    return render_template("index.html", user=session.get("user"), stock_count=universe.count())


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
    symbol = request.args.get("symbol", "").strip()
    exchange = request.args.get("exchange", "NSE").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    try:
        stock = data_fetcher.load_stock(symbol, exchange, with_intraday=True, with_info=True)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"failed to fetch data: {exc}"}), 502

    if stock.history is None or stock.history.empty:
        return jsonify({"error": f"no data found for {symbol} on {exchange}"}), 404

    df = indicators.add_daily_indicators(stock.history)
    recommendations = strategy.analyze(stock)
    try:
        prediction = predictor.predict(symbol, exchange)
    except Exception as exc:  # noqa: BLE001 - never fail the whole analysis on ML errors
        prediction = {"available": False, "reason": f"model error: {exc}", "verdict": "HOLD"}
    return jsonify({
        "info": _info_payload(stock, df),
        "chart": _chart_payload(df),
        "recommendations": recommendations,
        "prediction": prediction,
    })


@app.route("/api/predict")
@api_login_required
def api_predict():
    symbol = request.args.get("symbol", "").strip()
    exchange = request.args.get("exchange", "NSE").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    try:
        return jsonify(predictor.predict(symbol, exchange))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"prediction failed: {exc}"}), 502


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
            "name": item["name"],
            "symbol": item["symbol"],
            "exchange": exchange,
            "price": _clean_float(last),
            "change_pct": _clean_float((last - prev) / prev * 100) if prev else None,
            "verdict": rec.verdict,
            "score": rec.score,
            "confidence": rec.confidence,
            "entry": rec.entry,
            "target": rec.target,
            "stop_loss": rec.stop_loss,
            "risk_reward": rec.risk_reward,
            "top_signals": [
                {"label": s.label, "direction": s.direction, "detail": s.detail}
                for s in rec.signals[:4]
            ],
        }
    except Exception:  # noqa: BLE001
        return None


@app.route("/api/screener")
@api_login_required
def api_screener():
    horizon = request.args.get("horizon", "short_term").strip()
    exchange = request.args.get("exchange", "NSE").strip().upper()
    if horizon not in ("intraday", "short_term", "long_term"):
        return jsonify({"error": "invalid horizon"}), 400

    # Scanning every NSE stock live would be very slow / rate-limited, so the
    # screener scans the most liquid names first, capped by ``limit``.
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
        "horizon": horizon,
        "exchange": exchange,
        "scanned": len(stocks),
        "count": len(results),
        "top_buys": buys[:12],
        "top_sells": sells[-12:][::-1] if sells else [],
        "all": results,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
