"""Tests for the broad-market context module (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import market
from tests.conftest import make_ohlcv


def test_index_features_are_causal():
    df = make_ohlcv(n=300)
    feats = market._index_features(df["Close"])
    assert set(feats.columns) == {"mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20"}
    # First row of a diff-based feature is NaN (no look-ahead / no fill here).
    assert np.isnan(feats["mkt_ret_1"].iloc[0])


def test_compute_context_with_index_and_sentiment(monkeypatch):
    idx_df = make_ohlcv(n=300)
    monkeypatch.setattr(market.providers, "get_daily", lambda *a, **k: (idx_df, {}))
    sent = pd.DataFrame(
        {"sentiment": np.linspace(-0.2, 0.2, 30), "article_count": 5},
        index=idx_df.index[-30:],
    )
    monkeypatch.setattr(market.store, "get_sentiment", lambda *a, **k: sent)
    ctx = market._compute_context("NSE", "max")
    for col in market.MARKET_FEATURE_COLS:
        assert col in ctx.columns
    # Recent rows carry the merged sentiment; older rows are neutral-filled.
    assert ctx["mkt_sent_1d"].iloc[-1] != 0.0
    assert ctx["mkt_sent_1d"].iloc[0] == 0.0


def test_compute_context_without_index(monkeypatch):
    monkeypatch.setattr(market.providers, "get_daily", lambda *a, **k: (pd.DataFrame(), {}))
    monkeypatch.setattr(market.store, "get_sentiment", lambda *a, **k: pd.DataFrame())
    ctx = market._compute_context("NSE", "max")
    # Degrades gracefully: sentiment columns present and neutral.
    assert "mkt_sent_1d" in ctx.columns


def test_context_signature():
    assert market.context_signature(pd.DataFrame()) == "none"
    df = make_ohlcv(n=10)
    sig = market.context_signature(df)
    assert ":" in sig and sig.startswith("10:")
