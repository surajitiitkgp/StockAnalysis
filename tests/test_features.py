"""Tests for feature engineering and the supervised-target builder."""

from __future__ import annotations

import numpy as np

from analysis import features


def test_build_features_has_all_columns(ohlcv):
    f = features.build_features(ohlcv)
    for col in features.FEATURE_COLS:
        assert col in f.columns
    assert "Close" in f.columns


def test_no_inf_values(ohlcv):
    f = features.build_features(ohlcv)
    assert not np.isinf(f.select_dtypes("number").to_numpy()).any()


def test_make_supervised_target_is_forward_return(ohlcv):
    X, y, close, dates, last_valid, cols = features.make_supervised(ohlcv, horizon=7)
    assert last_valid == len(close) - 7
    # y[i] must equal close[i+7]/close[i]-1 for valid rows.
    i = 10
    expected = close[i + 7] / close[i] - 1.0
    assert abs(y[i] - expected) < 1e-9
    # Rows beyond last_valid have no known outcome.
    assert np.isnan(y[last_valid:]).all()


def test_feature_matrix_shape_matches(ohlcv):
    X, y, close, dates, last_valid, cols = features.make_supervised(ohlcv, horizon=7)
    assert X.shape[0] == len(y) == len(close) == len(dates)
    assert X.shape[1] == len(features.FEATURE_COLS)
    assert cols == features.FEATURE_COLS


def test_sentiment_features_appended(ohlcv):
    import pandas as pd
    # Build a sparse sentiment archive covering part of the price range.
    idx = ohlcv.index[-50:]
    sent = pd.DataFrame(
        {"sentiment": np.linspace(-0.5, 0.5, len(idx)), "article_count": 3},
        index=idx,
    )
    X, y, close, dates, last_valid, cols = features.make_supervised(ohlcv, 7, sent)
    assert cols == features.FEATURE_COLS + features.NEWS_FEATURE_COLS
    assert X.shape[1] == len(cols)
    # No NaNs introduced by the news merge (neutral-filled where absent).
    assert not np.isnan(X).any()
