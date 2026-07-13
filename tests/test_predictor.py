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


# --------------------------------------------------------------------------- #
# Graduated insufficient-history workflow (Sec. 7) + provenance (Sec. 8)
# --------------------------------------------------------------------------- #
def test_full_history_mode_and_freshness(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    r = predictor.predict("TESTFULL", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["prediction_mode"] == predictor.MODE_FULL
    assert r["limited_data"] is False
    dq = r["data_quality"]
    for key in ("first_observation", "last_observation", "raw_observations",
                "feature_ready_observations", "interval", "trading_sessions_old",
                "quality", "generated_at"):
        assert key in dq
    assert dq["raw_observations"] == r["history_days"]


def test_baseline_models_registered():
    assert set(models.baseline_keys()) == {"naive", "drift"}
    assert models.is_baseline("naive")
    assert not models.is_baseline("random_forest")
    assert "naive" not in models.selectable_keys()
    assert "drift" not in models.selectable_keys()


def test_naive_and_drift_predict():
    X = np.zeros((5, 3))
    naive = models.build("naive").fit(X, np.array([0.1, -0.2, 0.3, 0.0, 0.05]))
    assert list(naive.predict(X)) == [0.0] * 5
    drift = models.build("drift").fit(X, np.array([0.1, 0.3]))
    assert abs(drift.predict(X)[0] - 0.2) < 1e-9


def test_reduced_feature_fallback(monkeypatch):
    # ~140 rows: below ml_min_rows (300) so the full tier can't validate, but
    # well above the hard floor, so a limited-data tier should engage.
    _patch_history(monkeypatch, make_ohlcv(n=140))
    r = predictor.predict("TESTLOW", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["prediction_mode"] in (predictor.MODE_REDUCED, predictor.MODE_BASELINE)
    assert r["limited_data"] is True
    assert r["confidence"] <= predictor._MODE_CONF_CAP[r["prediction_mode"]]


def test_refuses_below_hard_floor(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv(n=40))
    r = predictor.predict("TESTTINY", model="random_forest", use_cache=False)
    assert not r["available"]
    assert r["prediction_mode"] == predictor.MODE_REFUSED
    assert r["horizons"] == []


def test_baseline_comparison_present(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    r = predictor.predict("TESTBM", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    h = r["horizons"][0]
    assert "baseline_comparison" in h
    assert "beats_baseline" in h["baseline_comparison"]


def test_prediction_is_audited(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    from analysis import store
    before = store.prediction_stats()["total"]
    predictor.predict("AUDIT", model="random_forest", horizons=[1, 7], use_cache=False)
    rows = store.recent_predictions("AUDIT")
    assert len(rows) == 2  # one row per horizon
    assert {r["horizon_days"] for r in rows} == {1, 7}
    assert store.prediction_stats()["total"] == before + 2
    assert all(r["prediction_mode"] == predictor.MODE_FULL for r in rows)


def test_data_quality_includes_coverage(monkeypatch):
    _patch_history(monkeypatch, make_ohlcv())
    r = predictor.predict("COVTEST", model="random_forest", horizons=[7], use_cache=False)
    assert r["available"]
    cov = r["data_quality"].get("coverage")
    assert cov is not None
    assert "coverage_pct" in cov and "missing" in cov


# --------------------------------------------------------------------------- #
# Multi-Source Fusion model: specialists + leak-free blend + attribution + bands
# --------------------------------------------------------------------------- #
def test_fusion_model_registered():
    keys = [m["key"] for m in models.available_models()]
    assert "fusion" in keys
    assert "fusion" in models.selectable_keys()
    assert not models.is_baseline("fusion")


def test_fusion_regressor_fits_and_attributes():
    import numpy as np
    from analysis import features

    rng = np.random.default_rng(0)
    n = 260
    # 3 price cols carry the real signal; news/geo cols are noise.
    price = rng.normal(0, 1, (n, 3))
    news = rng.normal(0, 1, (n, 2))
    geo = rng.normal(0, 1, (n, 2))
    X = np.column_stack([price, news, geo])
    y = 0.6 * price[:, 0] + 0.3 * price[:, 1] + rng.normal(0, 0.1, n)
    group_map = {"price": [0, 1, 2], "news": [3, 4], "geopolitics": [5, 6]}

    fr = models.build_fusion(group_map, gap=1).fit(X, y)
    assert set(fr.group_weights_) == {"price", "news", "geopolitics"}
    # Weights are a valid non-negative distribution.
    assert abs(sum(fr.group_weights_.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in fr.group_weights_.values())
    # Price should dominate since it carries the true signal.
    assert fr.group_weights_["price"] >= fr.group_weights_["news"]
    assert fr.group_weights_["price"] >= fr.group_weights_["geopolitics"]
    # Attribution keys match the groups and are finite.
    attr = fr.attribution(X)
    assert set(attr) == {"price", "news", "geopolitics"}
    assert all(np.isfinite(v) for v in attr.values())


def test_fusion_single_group_degrades_gracefully():
    import numpy as np
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (120, 4))
    y = X[:, 0] + rng.normal(0, 0.1, 120)
    # No group map => treated as one "price" group; must still fit & predict.
    fr = models.FusionRegressor().fit(X, y)
    preds = fr.predict(X[:5])
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_quantile_bands_are_monotone():
    import numpy as np
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (200, 3))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.2, 200)
    qb = models.QuantileBands().fit(X, y)
    b = qb.predict_bands(X[:10])
    assert np.all(b["p10"] <= b["p50"] + 1e-9)
    assert np.all(b["p50"] <= b["p90"] + 1e-9)


def test_predict_with_fusion_emits_band_and_attribution(monkeypatch):
    import numpy as np
    import pandas as pd

    df = make_ohlcv()
    _patch_history(monkeypatch, df)
    idx = df.index
    sent = pd.DataFrame({"sentiment": np.linspace(-0.2, 0.2, len(idx)),
                         "article_count": 5}, index=idx)
    monkeypatch.setattr(predictor.store, "get_sentiment", lambda *a, **k: sent)
    ctx = pd.DataFrame({
        "mkt_ret_1": np.random.default_rng(3).normal(0, 0.01, len(df)),
        "mkt_ret_5": 0.0, "mkt_ret_20": 0.01, "mkt_vol_20": 0.012,
        "mkt_sent_1d": 0.1, "mkt_sent_7d": 0.05,
    }, index=df.index)
    monkeypatch.setattr(predictor.market, "get_market_context", lambda *a, **k: ctx)
    monkeypatch.setattr(predictor.news, "is_enabled", lambda: False)

    r = predictor.predict("FUSE", model="fusion", horizons=[7], use_cache=False)
    assert r["available"]
    assert r["model"]["key"] == "fusion"
    h = r["horizons"][0]
    band = h.get("forecast_band")
    assert band is not None
    assert band["p10_price"] <= band["p50_price"] <= band["p90_price"]
    sources = h["signal_attribution"]["sources"]
    assert {s["source"] for s in sources} == {"price", "news", "geopolitics"}
    assert abs(sum(s["share_pct"] for s in sources) - 100.0) < 1.5
