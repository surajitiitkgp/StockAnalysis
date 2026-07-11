"""Tests for the optional NSE India API client, provider and VIX feature.

All network access is faked, so these run offline and deterministically.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from analysis import features, nse_client, providers


class _FakeNSE:
    """Stand-in for ``nse.NSE`` returning canned NSE-shaped payloads."""

    def __init__(self, equity=None, vix=None):
        self._equity = equity or []
        self._vix = vix or []

    def fetch_equity_historical_data(self, symbol, from_date=None, to_date=None):
        return self._equity

    def fetch_historical_vix_data(self, from_date=None, to_date=None):
        return self._vix

    def exit(self):
        pass


# --------------------------------------------------------------------------- #
# nse_client parsing helpers
# --------------------------------------------------------------------------- #
def test_parse_day_handles_nse_formats():
    assert nse_client._parse_day("29-Jun-2026").strftime("%Y-%m-%d") == "2026-06-29"
    # VIX uses upper-case month.
    assert nse_client._parse_day("29-JUN-2026").strftime("%Y-%m-%d") == "2026-06-29"
    assert nse_client._parse_day("") is None


def test_period_to_start_years():
    start = nse_client._period_to_start("2y")
    assert (date.today() - start).days >= 365 * 2


def test_chunks_cover_range_without_overlap():
    start, end = date(2020, 1, 1), date(2022, 1, 1)
    chunks = list(nse_client._chunks(start, end))
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # Each chunk starts the day after the previous ended (no gap / overlap).
    for (a_s, a_e), (b_s, _b_e) in zip(chunks, chunks[1:]):
        assert (b_s - a_e).days == 1


def test_equity_history_parses_and_filters(monkeypatch):
    rows = [
        {"chSeries": "EQ", "mtimestamp": "10-Jul-2026", "chOpeningPrice": 100,
         "chTradeHighPrice": 105, "chTradeLowPrice": 99, "chClosingPrice": 101,
         "chTotTradedQty": 1000},
        {"chSeries": "BE", "mtimestamp": "09-Jul-2026", "chClosingPrice": 50},  # dropped
        {"chSeries": "EQ", "mtimestamp": "09-Jul-2026", "chOpeningPrice": 98,
         "chTradeHighPrice": 101, "chTradeLowPrice": 97, "chClosingPrice": 100,
         "chTotTradedQty": 1200},
    ]
    monkeypatch.setattr(nse_client, "_get_client", lambda: _FakeNSE(equity=rows))
    df = nse_client.equity_history("RELIANCE.NS", "1y")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2  # BE series filtered out
    assert df.index[0] < df.index[1]
    assert df["Close"].iloc[-1] == 101


def test_equity_history_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(nse_client, "_get_client", lambda: None)
    assert nse_client.equity_history("RELIANCE.NS", "1y").empty


def test_vix_history_parses(monkeypatch):
    rows = [
        {"EOD_TIMESTAMP": "10-JUL-2026", "EOD_CLOSE_INDEX_VAL": 12.25},
        {"EOD_TIMESTAMP": "09-JUL-2026", "EOD_CLOSE_INDEX_VAL": 13.36},
    ]
    monkeypatch.setattr(nse_client, "_get_client", lambda: _FakeNSE(vix=rows))
    s = nse_client.vix_history("1y")
    assert len(s) == 2
    assert s.name == "vix"
    assert s.iloc[-1] == 12.25


# --------------------------------------------------------------------------- #
# NseProvider
# --------------------------------------------------------------------------- #
def test_nse_provider_rejects_non_nse():
    p = providers.NseProvider()
    with pytest.raises(providers.ProviderError):
        p.daily("AAPL", "1y")


def test_nse_provider_uses_client(monkeypatch):
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10.0]},
        index=pd.to_datetime(["2026-07-10"]),
    )
    monkeypatch.setattr(nse_client, "equity_history", lambda *a, **k: frame)
    df = providers.NseProvider().daily("RELIANCE.NS", "1y")
    assert df["Close"].iloc[0] == 1.5


def test_nse_provider_empty_raises(monkeypatch):
    monkeypatch.setattr(nse_client, "equity_history", lambda *a, **k: pd.DataFrame())
    with pytest.raises(providers.ProviderError):
        providers.NseProvider().daily("RELIANCE.NS", "1y")


# --------------------------------------------------------------------------- #
# VIX flows into the feature set dynamically
# --------------------------------------------------------------------------- #
def _synthetic_prices(n=260):
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = pd.Series(100 + np.arange(n) * 0.1, index=idx)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1e5,
    }, index=idx)


def _market_frame(index, with_vix):
    m = pd.DataFrame(index=index)
    m["mkt_ret_1"] = 0.001
    m["mkt_ret_5"] = 0.002
    m["mkt_ret_20"] = 0.01
    m["mkt_vol_20"] = 0.02
    m["mkt_sent_1d"] = 0.1
    m["mkt_sent_7d"] = 0.05
    if with_vix:
        m["mkt_vix"] = 14.0
        m["mkt_vix_chg_5"] = 0.03
    return m


def test_market_columns_include_vix_when_present():
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = features._market_columns(_market_frame(idx, with_vix=True))
    assert "mkt_vix" in cols and "mkt_vix_chg_5" in cols
    assert cols[-1] == "rel_strength_20"


def test_market_columns_exclude_vix_when_absent():
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = features._market_columns(_market_frame(idx, with_vix=False))
    assert "mkt_vix" not in cols
    assert "rel_strength_20" in cols


def test_supervised_includes_vix_features():
    df = _synthetic_prices()
    market_df = _market_frame(df.index, with_vix=True)
    X, y, close, dates, last_valid, cols = features.make_supervised(df, 7, None, market_df)
    assert "mkt_vix" in cols and "mkt_vix_chg_5" in cols
    assert X.shape[1] == len(cols)
    # No NaNs leaked into the training matrix.
    assert not np.isnan(X).any()
