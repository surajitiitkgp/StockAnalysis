"""Tests for the recommendation engine (verdict thresholds + scoring)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import strategy
from analysis.data_fetcher import StockData
from tests.conftest import make_ohlcv


def _stock(df, intraday=None):
    return StockData(
        symbol="TEST", exchange="NSE", yahoo_ticker="TEST.NS", name="Test",
        history=df, intraday=intraday if intraday is not None else pd.DataFrame(),
        info={},
    )


def test_verdict_thresholds():
    assert strategy._verdict(60) == "STRONG BUY"
    assert strategy._verdict(20) == "BUY"
    assert strategy._verdict(0) == "HOLD"
    assert strategy._verdict(-20) == "SELL"
    assert strategy._verdict(-60) == "STRONG SELL"


def test_scores_within_range():
    df = make_ohlcv()
    stock = _stock(df)
    for rec in (strategy.long_term(stock), strategy.short_term(stock), strategy.intraday(stock)):
        assert -100 <= rec.score <= 100
        assert 0 <= rec.confidence <= 100
        assert rec.verdict in {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}


def test_uptrend_is_not_bearish():
    # A persistent uptrend should not produce a SELL long-term verdict.
    n = 400
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = pd.Series(np.linspace(100, 300, n), index=idx)
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 500000.0,
    }, index=idx)
    rec = strategy.long_term(_stock(df))
    assert rec.score > 0
    assert rec.verdict in {"BUY", "STRONG BUY", "HOLD"}


def test_insufficient_data_returns_hold():
    idx = pd.bdate_range("2023-01-01", periods=5)
    df = pd.DataFrame({
        "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0,
    }, index=idx)
    rec = strategy.short_term(_stock(df))
    assert rec.verdict == "HOLD"
