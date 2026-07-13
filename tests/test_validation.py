"""Tests for the data-quality validation layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import validation
from tests.conftest import make_ohlcv


def test_clean_good_data_passes(ohlcv):
    out, report = validation.clean_ohlcv(ohlcv)
    assert report.ok
    assert len(out) == len(ohlcv)
    assert report.rows_out == report.rows_in


def test_empty_frame_flagged():
    out, report = validation.clean_ohlcv(pd.DataFrame())
    assert not report.ok
    assert "empty" in report.issues
    assert out.empty


def test_nonpositive_prices_dropped():
    df = make_ohlcv(n=60)
    df.iloc[10, df.columns.get_loc("Close")] = -5.0
    out, report = validation.clean_ohlcv(df)
    assert len(out) == len(df) - 1
    assert any("nonpositive" in i for i in report.issues)


def test_duplicate_dates_deduped():
    df = make_ohlcv(n=30)
    dup = pd.concat([df, df.iloc[[5]]]).sort_index()
    out, report = validation.clean_ohlcv(dup)
    assert not out.index.duplicated().any()
    assert any("deduped" in i for i in report.issues)


def test_price_spike_dropped():
    df = make_ohlcv(n=60)
    df.iloc[30, df.columns.get_loc("Close")] *= 3  # +200% single-day spike
    out, report = validation.clean_ohlcv(df)
    assert any("spike" in i for i in report.issues)


def test_high_low_envelope_repaired():
    df = make_ohlcv(n=40)
    # Force an inconsistent bar: High below Close.
    i = df.columns.get_loc("High")
    df.iloc[15, i] = df.iloc[15, df.columns.get_loc("Close")] * 0.5
    out, report = validation.clean_ohlcv(df)
    row = out.iloc[15]
    assert row["High"] >= max(row["Open"], row["Close"])


def test_expected_sessions_counts_business_days():
    import pandas as pd
    from analysis import validation
    # Mon 2024-01-01 .. Fri 2024-01-05 inclusive = 5 business days.
    assert validation.expected_sessions(pd.Timestamp("2024-01-01"),
                                        pd.Timestamp("2024-01-05")) == 5


def test_missing_sessions_flags_gap():
    import numpy as np
    import pandas as pd
    from analysis import validation
    # Two continuous business-day weeks with a one-week hole in the middle.
    idx = pd.bdate_range("2024-01-01", periods=5).append(
        pd.bdate_range("2024-01-15", periods=5))
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                       "Volume": 1.0}, index=idx)
    rep = validation.missing_sessions(df)
    assert rep["present"] == 10
    assert rep["missing"] >= 5
    assert rep["largest_gaps"]
    assert rep["coverage_pct"] is not None


def test_missing_sessions_empty():
    import pandas as pd
    from analysis import validation
    rep = validation.missing_sessions(pd.DataFrame())
    assert rep["present"] == 0 and rep["largest_gaps"] == []
