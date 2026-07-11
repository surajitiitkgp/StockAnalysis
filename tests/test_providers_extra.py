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
