"""Technical indicators implemented with pandas/numpy (no TA-Lib dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min = df["Low"].rolling(k, min_periods=k).min()
    high_max = df["High"].rolling(k, min_periods=k).max()
    percent_k = 100 * (df["Close"] - low_min) / (high_max - low_min)
    percent_d = percent_k.rolling(d, min_periods=d).mean()
    return percent_k, percent_d


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean().fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price (cumulative within the provided frame)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (typical * df["Volume"]).cumsum() / cum_vol


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def add_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Augment a daily OHLCV frame with the indicators used by the strategies."""
    if df is None or df.empty:
        return df
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["EMA9"] = ema(close, 9)
    out["EMA21"] = ema(close, 21)
    out["RSI14"] = rsi(close, 14)
    macd_line, signal_line, hist = macd(close)
    out["MACD"] = macd_line
    out["MACD_SIGNAL"] = signal_line
    out["MACD_HIST"] = hist
    upper, mid, lower = bollinger(close, 20)
    out["BB_UPPER"] = upper
    out["BB_MID"] = mid
    out["BB_LOWER"] = lower
    out["ATR14"] = atr(out, 14)
    out["ADX14"] = adx(out, 14)
    k, d = stochastic(out)
    out["STOCH_K"] = k
    out["STOCH_D"] = d
    out["OBV"] = obv(out)
    out["VOL_SMA20"] = sma(out["Volume"], 20)
    return out
