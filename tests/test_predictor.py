"""Tests for the multi-model / multi-horizon predictor (no network)."""

from __future__ import annotations

import numpy as np

from analysis import models, predictor
from tests.conftest import make_ohlcv


def _patch_history(monkeypatch, df):
    monkeypatch.setattr(predictor.data_fetcher, "get_daily_history", lambda *a, **k: df)


def test_available_models_includes_auto():
    keys = [m["key"] for m in models.available_models()]
    assert "auto" in keys
    assert "random_forest" in keys
    assert "ensemble" in keys


def test_verdict_scales_with_horizon():
    # sqrt-time scaling: thresholds shrink for short horizons, grow for long ones.
    # A +3% forecast is a BUY over 7d, a very strong signal over 1d, and only a
    # HOLD over 30d (where a 3% move is unremarkable).
    assert predictor._verdict(0.03, 7) == "BUY"
    assert predictor._verdict(0.03, 1) == "STRONG BUY"
    assert predictor._verdict(0.03, 30) == "HOLD"


def test_predict_multi_horizon(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    r = predictor.predict("TEST", model="random_forest", use_cache=False)
    assert r["available"]
    days = [h["days"] for h in r["horizons"]]
    assert days == sorted(days)
    assert set(days).issubset(set(predictor.HORIZONS))
    for h in r["horizons"]:
        assert "forecast_price" in h and h["forecast_price"] > 0
        assert -100 <= h["expected_return_pct"] <= 1000
        assert 0 <= h["confidence"] <= 95
        assert set(h["metrics"]).issuperset({"rmse_pct", "mae_pct", "directional_accuracy_pct"})


def test_predict_specific_horizons(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    r = predictor.predict("TEST", model="ridge", horizons=[1, 7], use_cache=False)
    assert [h["days"] for h in r["horizons"]] == [1, 7]


def test_auto_selects_a_model(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    r = predictor.predict("TEST", model="auto", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["model"]["selected_from"] == "auto"
    assert r["model"]["key"] in models.selectable_keys()
    assert len(r["model"]["scoreboard"]) >= 1


def test_insufficient_history(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv(n=50))
    r = predictor.predict("TEST", model="random_forest", use_cache=False)
    assert not r["available"]
    assert "history" in r["reason"].lower()


def test_predict_uses_sentiment_archive(monkeypatch):
    import pandas as pd

    df = make_ohlcv()
    _patch_history(monkeypatch, df)
    idx = df.index[-400:]
    sent = pd.DataFrame(
        {"sentiment": np.linspace(-0.3, 0.3, len(idx)), "article_count": 4}, index=idx)
    monkeypatch.setattr(predictor.store, "get_sentiment", lambda *a, **k: sent)
    r = predictor.predict("TESTNEWS", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["news_features_used"] is True
    assert any("news" in f["name"] for f in r["top_features"])


def test_predict_no_news_by_default(monkeypatch):
    import pandas as pd

    _patch_history(monkeypatch, make_ohlcv())
    monkeypatch.setattr(predictor.store, "get_sentiment", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(predictor.news, "is_enabled", lambda: False)
    r = predictor.predict("TESTNONE", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["news_features_used"] is False
    assert r["news"] is None


def test_predict_uses_market_context(monkeypatch):
    import pandas as pd

    df = make_ohlcv()
    _patch_history(monkeypatch, df)
    ctx = pd.DataFrame({
        "mkt_ret_1": np.random.default_rng(1).normal(0, 0.01, len(df)),
        "mkt_ret_5": 0.0, "mkt_ret_20": 0.01, "mkt_vol_20": 0.012,
        "mkt_sent_1d": 0.1, "mkt_sent_7d": 0.05,
    }, index=df.index)
    monkeypatch.setattr(predictor.market, "get_market_context", lambda *a, **k: ctx)
    r = predictor.predict("TESTMKT", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["market_features_used"] is True


def test_news_overlay_adjusts_confidence(monkeypatch):
    per_horizon = [{"expected_return_pct": 5.0, "confidence": 50.0}]
    monkeypatch.setattr(predictor.news, "is_enabled", lambda: True)
    monkeypatch.setattr(predictor.news, "get_sentiment_summary",
                        lambda *a, **k: {"available": True, "provider": "fake",
                                         "aggregate": {"score": 0.5, "label": "positive",
                                                       "count": 3, "positive": 3, "negative": 0},
                                         "daily": [], "headlines": []})
    block = predictor._news_overlay("TEST", "NSE", per_horizon)
    assert block["available"] is True
    # Positive news agreeing with a positive forecast raises confidence.
    assert per_horizon[0]["confidence"] > 50.0
    assert per_horizon[0]["news_adjusted"] is True
