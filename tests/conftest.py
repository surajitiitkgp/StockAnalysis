"""Shared test fixtures.

Runs everything in a temporary data directory and against synthetic price
data so the suite never touches the network or the real SQLite archive.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Point data/model/db paths at a temp dir and disable persistence noise.

    Also hard-disable live news so a developer's local ``.env`` (with real API
    keys) never causes the suite to hit the network. Tests that exercise the
    news path monkeypatch these explicitly.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "history.db"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ML_PERSIST", "0")

    from analysis import market, news, nse_client
    monkeypatch.setattr(news, "_active_providers", lambda: [])
    monkeypatch.setattr(news, "is_enabled", lambda: False)
    # Prevent live index fetches; tests exercising market features patch this.
    monkeypatch.setattr(market, "get_market_context", lambda *a, **k: pd.DataFrame())
    # Hard-disable the unofficial NSE API so the suite never scrapes NSE. We
    # stub the client factory (not the public helpers) so unit tests can still
    # exercise the real parsing by overriding ``_get_client`` themselves.
    monkeypatch.setattr(nse_client, "is_enabled", lambda: False)
    monkeypatch.setattr(nse_client, "_get_client", lambda: None)
    yield


def make_ohlcv(n: int = 900, seed: int = 7, start: str = "2019-01-01") -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV with a mild drift."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    rets = rng.normal(0.0005, 0.015, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    vol = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()
