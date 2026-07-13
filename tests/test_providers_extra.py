"""Tests for the optional extra price providers (mocked HTTP, no network)."""

from __future__ import annotations

import json

from analysis import providers


class _FakeResp:
    def __init__(self, data: bytes):
        self._d = data

    def read(self):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_urlopen(monkeypatch, payload: dict):
    raw = json.dumps(payload).encode()
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeResp(raw))


def test_twelvedata_symbol_exchange_mapping():
    p = providers.TwelveDataProvider("KEY")
    assert p._symbol_exchange("RELIANCE.NS") == ("RELIANCE", "NSE")
    assert p._symbol_exchange("TCS.BO") == ("TCS", "BSE")
    assert p._symbol_exchange("AAPL") == ("AAPL", None)


def test_twelvedata_daily_parsing(monkeypatch):
    payload = {"status": "ok", "values": [
        {"datetime": "2026-07-10", "open": "100", "high": "102", "low": "99",
         "close": "101", "volume": "1000"},
        {"datetime": "2026-07-09", "open": "98", "high": "101", "low": "97",
         "close": "100", "volume": "1200"},
    ]}
    _patch_urlopen(monkeypatch, payload)
    p = providers.TwelveDataProvider("KEY")
    df = p.daily("RELIANCE.NS", "2y")
    assert len(df) == 2
    assert df.index[0] < df.index[1]  # sorted ascending
    assert df["close"].iloc[-1] == 101


def test_twelvedata_error_raises(monkeypatch):
    _patch_urlopen(monkeypatch, {"status": "error", "message": "bad symbol"})
    p = providers.TwelveDataProvider("KEY")
    try:
        p.daily("BOGUS.NS", "2y")
        assert False, "expected ProviderError"
    except providers.ProviderError:
        pass


def test_alphavantage_symbol_mapping():
    p = providers.AlphaVantageProvider("KEY")
    assert p._av_symbol("RELIANCE.NS") == "RELIANCE.BSE"
    assert p._av_symbol("AAPL") == "AAPL"


def test_alphavantage_daily_parsing(monkeypatch):
    payload = {"Time Series (Daily)": {
        "2026-07-10": {"1. open": "100", "2. high": "102", "3. low": "99",
                       "4. close": "101", "5. volume": "1000"},
        "2026-07-09": {"1. open": "98", "2. high": "101", "3. low": "97",
                       "4. close": "100", "5. volume": "1200"},
    }}
    _patch_urlopen(monkeypatch, payload)
    p = providers.AlphaVantageProvider("KEY")
    df = p.daily("RELIANCE.NS", "2y")
    assert len(df) == 2
    assert df["close"].iloc[-1] == 101


def test_alphavantage_rate_limit_note_raises(monkeypatch):
    _patch_urlopen(monkeypatch, {"Note": "rate limit reached"})
    p = providers.AlphaVantageProvider("KEY")
    try:
        p.daily("RELIANCE.NS", "2y")
        assert False, "expected ProviderError"
    except providers.ProviderError:
        pass


# --------------------------------------------------------------------------- #
# Stooq bot-challenge detection + fallback health severity
# --------------------------------------------------------------------------- #
class _FakeTextResp:
    def __init__(self, text: str):
        self._d = text.encode()

    def read(self):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch_text(monkeypatch, text: str):
    monkeypatch.setattr(providers.urllib.request, "urlopen",
                        lambda req, timeout=None: _FakeTextResp(text))


def test_stooq_valid_csv(monkeypatch):
    csv = "Date,Open,High,Low,Close,Volume\n2026-07-10,100,102,99,101,1000\n"
    _patch_text(monkeypatch, csv)
    df = providers.StooqProvider().daily("RELIANCE.NS", "2y")
    assert len(df) == 1
    assert df["Close"].iloc[-1] == 101


def test_stooq_bot_challenge_detected(monkeypatch):
    html = ('<!DOCTYPE html><html><head></head><body><noscript>'
            'This site requires JavaScript to verify your browser.</noscript></body></html>')
    _patch_text(monkeypatch, html)
    try:
        providers.StooqProvider().daily("RELIANCE.NS", "2y")
        assert False, "expected ProviderError"
    except providers.ProviderError as exc:
        assert "challenge" in str(exc).lower()


def test_stooq_empty_body_raises(monkeypatch):
    _patch_text(monkeypatch, "")
    try:
        providers.StooqProvider().daily("RELIANCE.NS", "2y")
        assert False, "expected ProviderError"
    except providers.ProviderError as exc:
        assert "no usable data" in str(exc).lower()


def test_stooq_is_optional_fallback():
    assert providers.StooqProvider().optional is True
    assert providers.YahooProvider().optional is False


def test_provider_health_optional_failure_is_limited(monkeypatch):
    # An optional provider that raises should be reported "limited", not "degraded".
    def boom(self, ticker, period):
        raise providers.ProviderError("stooq blocked request (bot/JS challenge)")
    monkeypatch.setattr(providers.StooqProvider, "daily", boom)
    monkeypatch.setattr(providers.YahooProvider, "daily",
                        lambda self, t, p: __import__("pandas").DataFrame(
                            {"Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1]},
                            index=__import__("pandas").to_datetime(["2026-07-10"])))
    rows = {r["name"]: r for r in providers.provider_health("RELIANCE.NS")}
    assert rows["yahoo"]["status"] == "ok"
    if "stooq" in rows:
        assert rows["stooq"]["status"] == "limited"
        assert rows["stooq"]["optional"] is True
