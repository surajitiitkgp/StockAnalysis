"""API integration tests using Flask's test client (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

import app as app_module
from analysis.data_fetcher import StockData
from tests.conftest import make_ohlcv


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.fixture
def logged_in(client):
    with client.session_transaction() as sess:
        sess["user"] = "tester"
    return client


def _fake_stock():
    df = make_ohlcv()
    return StockData(
        symbol="RELIANCE", exchange="NSE", yahoo_ticker="RELIANCE.NS",
        name="Reliance", history=df, intraday=pd.DataFrame(), info={},
        meta={"provider": "yahoo", "freshness": {"rows": len(df), "last_date": "2024-01-01",
                                                 "age_days": 1, "stale": False}},
    )


def test_health(client):
    assert client.get("/healthz").status_code == 200


def test_ready(client):
    r = client.get("/readyz")
    assert r.status_code in (200, 503)
    assert "universe" in r.get_json()


def test_api_requires_auth(client):
    assert client.get("/api/analyze?symbol=RELIANCE").status_code == 401


def test_models_endpoint(logged_in):
    data = logged_in.get("/api/models").get_json()
    assert any(m["key"] == "auto" for m in data["models"])
    assert 7 in data["horizons"]


def test_analyze_validation(logged_in):
    # Missing symbol -> 400.
    assert logged_in.get("/api/analyze").status_code == 400
    # Bad exchange -> 400.
    assert logged_in.get("/api/analyze?symbol=RELIANCE&exchange=XYZ").status_code == 400
    # Bad symbol characters -> 400.
    assert logged_in.get("/api/analyze?symbol=../etc").status_code == 400


def test_analyze_happy_path(logged_in, monkeypatch):
    monkeypatch.setattr(app_module.data_fetcher, "load_stock", lambda *a, **k: _fake_stock())
    monkeypatch.setattr(app_module.predictor, "predict",
                        lambda *a, **k: {"available": True, "verdict": "BUY", "horizons": []})
    r = logged_in.get("/api/analyze?symbol=RELIANCE&exchange=NSE&model=auto")
    assert r.status_code == 200
    data = r.get_json()
    assert set(data).issuperset({"info", "chart", "recommendations", "prediction", "data"})
    assert data["recommendations"]["short_term"]["verdict"] in {
        "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}


def test_analyze_includes_news(logged_in, monkeypatch):
    monkeypatch.setattr(app_module.data_fetcher, "load_stock", lambda *a, **k: _fake_stock())
    monkeypatch.setattr(app_module.predictor, "predict",
                        lambda *a, **k: {"available": True, "verdict": "BUY", "horizons": []})
    monkeypatch.setattr(app_module.news, "get_sentiment_summary",
                        lambda *a, **k: {"available": True, "provider": "fake",
                                         "aggregate": {"score": 0.3, "label": "positive",
                                                       "count": 5}, "headlines": []})
    data = logged_in.get("/api/analyze?symbol=RELIANCE&exchange=NSE").get_json()
    assert data["news"]["available"] is True
    assert data["news"]["aggregate"]["label"] == "positive"


def test_api_news_company(logged_in, monkeypatch):
    monkeypatch.setattr(app_module.news, "get_sentiment_summary",
                        lambda *a, **k: {"available": True, "aggregate": {"score": 0.1}})
    r = logged_in.get("/api/news?symbol=RELIANCE&exchange=NSE")
    assert r.status_code == 200
    assert r.get_json()["available"] is True


def test_api_news_market(logged_in, monkeypatch):
    monkeypatch.setattr(app_module.news, "get_market_news",
                        lambda: {"available": True, "headlines": []})
    r = logged_in.get("/api/news?scope=market")
    assert r.status_code == 200
    assert r.get_json()["available"] is True


def test_api_news_requires_auth(client):
    assert client.get("/api/news?symbol=RELIANCE").status_code == 401


def test_predict_invalid_model(logged_in):
    assert logged_in.get("/api/predict?symbol=RELIANCE&model=bogus").status_code == 400


def test_screener_invalid_horizon(logged_in):
    assert logged_in.get("/api/screener?horizon=bogus").status_code == 400
