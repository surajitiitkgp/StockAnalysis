"""Unit tests for technical indicators (known-value / invariant checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import indicators


def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = indicators.sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == 2.0
    assert out.iloc[4] == 4.0


def test_rsi_bounds_and_uptrend():
    # Persistent uptrend with regular small pullbacks: gains dominate losses,
    # so RSI should sit well above 70.
    steps = np.array([1.0, 1.0, 1.0, 1.0, -0.4] * 40)
    s = pd.Series(100 + np.cumsum(steps))
    r = indicators.rsi(s, 14)
    assert (r.dropna() >= 0).all() and (r.dropna() <= 100).all()
    assert r.iloc[-1] > 70


def test_macd_hist_is_diff_of_lines():
    s = pd.Series(np.linspace(10, 50, 100))
    macd_line, signal, hist = indicators.macd(s)
    assert np.allclose((macd_line - signal).to_numpy(), hist.to_numpy(), atol=1e-9)


def test_bollinger_ordering(ohlcv):
    upper, mid, lower = indicators.bollinger(ohlcv["Close"], 20)
    valid = upper.dropna().index
    assert (upper.loc[valid] >= mid.loc[valid]).all()
    assert (mid.loc[valid] >= lower.loc[valid]).all()


def test_atr_non_negative(ohlcv):
    a = indicators.atr(ohlcv, 14).dropna()
    assert (a >= 0).all()


def test_add_daily_indicators_columns(ohlcv):
    out = indicators.add_daily_indicators(ohlcv)
    for col in ["SMA20", "SMA50", "SMA200", "EMA9", "RSI14", "MACD", "ATR14", "ADX14"]:
        assert col in out.columns
