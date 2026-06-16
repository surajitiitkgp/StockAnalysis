"""Machine-learning price prediction + verdict.

A Random Forest **regression** model is trained per stock on up to 10 years of
daily history (served from the local SQLite store). It predicts the stock's
**7-trading-day-ahead return**, from which we derive:

  - a 7-day forward **price target**,
  - a **backtest** of the last 7 days (predicted vs. actual, out-of-sample),
  - a **verdict**: STRONG BUY / BUY / HOLD / SELL / STRONG SELL.

The model is causal (every feature uses only information available on the
prediction date) and the backtest is strictly out-of-sample (training data ends
before the backtested window), so the reported accuracy is honest.

Disclaimer: statistical estimate for education only — not investment advice.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import data_fetcher

HORIZON = 7          # predict this many trading days ahead
BACKTEST_DAYS = 7    # out-of-sample days shown as predicted-vs-actual
_MIN_ROWS = 300      # need a reasonable history to train

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 60 * 15


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high, low, vol = df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["Close"] = close

    # Past returns over several lookbacks.
    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = close.pct_change(n)

    # Price relative to moving averages.
    for n in (5, 10, 20, 50, 200):
        sma = close.rolling(n).mean()
        f[f"px_sma_{n}"] = close / sma - 1.0
    f["sma_10_50"] = close.rolling(10).mean() / close.rolling(50).mean() - 1.0

    # Momentum / oscillators.
    f["rsi_14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - macd_sig) / close

    # Volatility & range.
    ret1 = close.pct_change()
    f["vol_10"] = ret1.rolling(10).std()
    f["vol_20"] = ret1.rolling(20).std()
    f["range_14"] = ((high - low) / close).rolling(14).mean()

    # Volume behaviour.
    f["vol_ratio"] = vol / vol.rolling(20).mean()

    # Position within the 52-week range.
    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    f["pos_52w"] = (close - roll_min) / (roll_max - roll_min)

    f = f.replace([np.inf, -np.inf], np.nan)
    return f


_FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "px_sma_5", "px_sma_10", "px_sma_20", "px_sma_50", "px_sma_200",
    "sma_10_50", "rsi_14", "macd_hist", "vol_10", "vol_20",
    "range_14", "vol_ratio", "pos_52w",
]


def _verdict(expected_return: float) -> str:
    """Map a predicted 7-day return to a verdict."""
    pct = expected_return * 100
    if pct >= 6:
        return "STRONG BUY"
    if pct >= 2:
        return "BUY"
    if pct > -2:
        return "HOLD"
    if pct > -6:
        return "SELL"
    return "STRONG SELL"


def _empty(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "verdict": "HOLD",
        "horizon_days": HORIZON,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def predict(symbol: str, exchange: str = "NSE", use_cache: bool = True) -> dict:
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    ckey = f"{base}:{exchange.upper()}"
    if use_cache:
        with _cache_lock:
            hit = _cache.get(ckey)
            if hit and time.time() - hit[0] < _CACHE_TTL:
                return hit[1]

    # Pull the deepest history we have (local 10y store, else remote).
    df = data_fetcher.get_daily_history(symbol, exchange, period="10y")
    if df is None or df.empty or len(df) < _MIN_ROWS:
        return _empty("Not enough history to train a model.")

    feats = _build_features(df)
    feats = feats.dropna(subset=_FEATURE_COLS + ["Close"])
    if len(feats) < _MIN_ROWS:
        return _empty("Not enough clean history after feature construction.")

    X = feats[_FEATURE_COLS].to_numpy(dtype=float)
    close = feats["Close"].to_numpy(dtype=float)
    dates = feats.index
    n = len(feats)

    H = HORIZON
    # Forward return target; only defined where a +H bar exists.
    last_valid = n - H  # rows [0, last_valid) have a known H-ahead outcome
    if last_valid < _MIN_ROWS // 2:
        return _empty("Not enough samples with a known forward outcome.")

    y = np.full(n, np.nan)
    for i in range(last_valid):
        y[i] = close[i + H] / close[i] - 1.0

    valid_idx = np.arange(last_valid)
    bt_idx = valid_idx[-BACKTEST_DAYS:]
    # Leave a gap of H rows so training never overlaps the backtest window.
    train_end = max(0, len(valid_idx) - BACKTEST_DAYS - H)
    train_idx = valid_idx[:train_end]
    if len(train_idx) < _MIN_ROWS // 2:
        return _empty("Not enough training data after the leakage gap.")

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42,
    )
    model.fit(X[train_idx], y[train_idx])

    # --- Out-of-sample backtest over the last BACKTEST_DAYS outcomes --------- #
    backtest = []
    actuals, preds = [], []
    for i in bt_idx:
        pr = float(model.predict(X[i].reshape(1, -1))[0])
        pred_price = close[i] * (1.0 + pr)
        actual_price = close[i + H]
        target_date = dates[i + H]
        backtest.append({
            "date": target_date.strftime("%Y-%m-%d"),
            "actual": round(float(actual_price), 2),
            "predicted": round(float(pred_price), 2),
        })
        actuals.append(actual_price)
        preds.append(pred_price)

    actuals = np.array(actuals)
    preds = np.array(preds)
    rmse_pct = float(np.sqrt(np.mean(((preds - actuals) / actuals) ** 2)) * 100)
    mae_pct = float(np.mean(np.abs((preds - actuals) / actuals)) * 100)
    # Directional accuracy: did we get the up/down direction right?
    base_prices = close[bt_idx]
    dir_actual = np.sign(actuals - base_prices)
    dir_pred = np.sign(preds - base_prices)
    directional_acc = float(np.mean(dir_actual == dir_pred) * 100)

    # --- Forward forecast: retrain on all valid rows, predict from today ----- #
    model.fit(X[valid_idx], y[valid_idx])
    last_price = float(close[-1])
    fwd_ret = float(model.predict(X[-1].reshape(1, -1))[0])
    forecast_price = last_price * (1.0 + fwd_ret)
    last_date = dates[-1]
    target_date = last_date + pd.tseries.offsets.BDay(H)

    verdict = _verdict(fwd_ret)
    # Confidence blends directional hit-rate with the size of the expected move.
    confidence = min(95.0, 35.0 + directional_acc * 0.4
                     + min(abs(fwd_ret) * 100, 12) * 1.5)

    # Top model features for transparency.
    importances = sorted(
        zip(_FEATURE_COLS, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )[:5]

    result = {
        "available": True,
        "verdict": verdict,
        "horizon_days": H,
        "last_price": round(last_price, 2),
        "last_date": last_date.strftime("%Y-%m-%d"),
        "forecast_price": round(forecast_price, 2),
        "forecast_date": target_date.strftime("%Y-%m-%d"),
        "expected_return_pct": round(fwd_ret * 100, 2),
        "confidence": round(confidence, 0),
        "model": "RandomForestRegressor",
        "train_samples": int(len(train_idx)),
        "history_days": int(n),
        "backtest": backtest,
        "metrics": {
            "rmse_pct": round(rmse_pct, 2),
            "mae_pct": round(mae_pct, 2),
            "directional_accuracy_pct": round(directional_acc, 0),
        },
        "top_features": [{"name": k, "importance": round(float(v), 3)} for k, v in importances],
    }

    with _cache_lock:
        _cache[ckey] = (time.time(), result)
    return result
