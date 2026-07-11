"""Causal feature engineering for the price-forecast models.

Every feature is computed using only information available on the row's date
(no look-ahead), so models trained on these features and validated with a
leakage gap produce honest out-of-sample estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "px_sma_5", "px_sma_10", "px_sma_20", "px_sma_50", "px_sma_200",
    "sma_10_50", "rsi_14", "macd_hist", "vol_10", "vol_20",
    "range_14", "vol_ratio", "pos_52w", "dow", "month",
]

# Causal news-sentiment features, appended only when a sentiment series is
# supplied. All are trailing (use only information up to the row's date).
NEWS_FEATURE_COLS = [
    "news_sent_1d", "news_sent_3d", "news_sent_7d", "news_vol_7d", "news_flow_7d",
]

# Broad-market context features (index dynamics + global sentiment). India VIX
# columns are appended dynamically only when the market frame provides them.
_BASE_MARKET_COLS = [
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20",
    "mkt_sent_1d", "mkt_sent_7d",
]
_VIX_COLS = ["mkt_vix", "mkt_vix_chg_5"]
_SENT_MARKET_COLS = ("mkt_sent_1d", "mkt_sent_7d")

# Default full set (base + relative strength) for reference / back-compat.
MARKET_FEATURE_COLS = _BASE_MARKET_COLS + ["rel_strength_20"]


def _market_columns(market_df: pd.DataFrame) -> list:
    """Market feature columns actually present in ``market_df`` (deterministic)."""
    present = [c for c in _BASE_MARKET_COLS + _VIX_COLS if c in market_df.columns]
    return present + ["rel_strength_20"]


def feature_columns(with_news: bool, market_cols: list | None = None) -> list:
    cols = list(FEATURE_COLS)
    if with_news:
        cols += NEWS_FEATURE_COLS
    if market_cols:
        cols += list(market_cols)
    return cols


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _merge_sentiment(index: pd.Index, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Align a daily sentiment archive onto the price index, causally.

    Days without news map to neutral (0) sentiment and zero article volume.
    All emitted features are trailing rolling aggregates.
    """
    out = pd.DataFrame(index=index)
    s = sentiment_df.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    daily = s["sentiment"].reindex(index).fillna(0.0)
    vol = s["article_count"].reindex(index).fillna(0.0) if "article_count" in s else pd.Series(0.0, index=index)

    out["news_sent_1d"] = daily
    out["news_sent_3d"] = daily.rolling(3, min_periods=1).mean()
    out["news_sent_7d"] = daily.rolling(7, min_periods=1).mean()
    out["news_vol_7d"] = np.log1p(vol.rolling(7, min_periods=1).sum())
    out["news_flow_7d"] = (daily * vol).rolling(7, min_periods=1).sum() / (
        vol.rolling(7, min_periods=1).sum().replace(0, np.nan))
    out["news_flow_7d"] = out["news_flow_7d"].fillna(0.0)
    return out


def _merge_market(f: pd.DataFrame, market_df: pd.DataFrame) -> None:
    """Merge broad-market context onto the feature frame ``f`` in place.

    Index returns/vol are forward-filled onto the stock's trading days; global
    sentiment is neutral-filled. ``rel_strength_20`` compares the stock's 20-day
    return against the market's (a classic outperformance signal).
    """
    m = market_df.copy()
    if getattr(m.index, "tz", None) is not None:
        m.index = m.index.tz_localize(None)
    aligned = m.reindex(f.index.union(m.index)).sort_index().ffill().reindex(f.index)
    mkt_cols = [c for c in market_df.columns if c.startswith("mkt_")]
    for col in mkt_cols:
        series = aligned[col]
        if col in _VIX_COLS:
            # VIX is a level: carry the last known value across market holidays
            # (ffill) and backfill only the leading pre-history gap.
            series = series.ffill().bfill()
        f[col] = series
    mkt_ret_20 = f["mkt_ret_20"] if "mkt_ret_20" in f else 0.0
    f["rel_strength_20"] = f["ret_20"] - mkt_ret_20
    for col in mkt_cols + ["rel_strength_20"]:
        if col in f:
            f[col] = f[col].fillna(0.0)


def build_features(df: pd.DataFrame, sentiment_df: pd.DataFrame | None = None,
                   market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a feature frame (indexed like ``df``) including a ``Close`` column.

    When ``sentiment_df`` is provided, causal news-sentiment features are
    appended (see :data:`NEWS_FEATURE_COLS`). When ``market_df`` is provided,
    broad-market context features are appended (see :data:`MARKET_FEATURE_COLS`).
    """
    close = df["Close"]
    high, low, vol = df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["Close"] = close

    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = close.pct_change(n)

    for n in (5, 10, 20, 50, 200):
        sma = close.rolling(n).mean()
        f[f"px_sma_{n}"] = close / sma - 1.0
    f["sma_10_50"] = close.rolling(10).mean() / close.rolling(50).mean() - 1.0

    f["rsi_14"] = _rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd_hist"] = (macd - macd_sig) / close

    ret1 = close.pct_change()
    f["vol_10"] = ret1.rolling(10).std()
    f["vol_20"] = ret1.rolling(20).std()
    f["range_14"] = ((high - low) / close).rolling(14).mean()
    f["vol_ratio"] = vol / vol.rolling(20).mean()

    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    f["pos_52w"] = (close - roll_min) / (roll_max - roll_min)

    # Mild seasonality signals.
    f["dow"] = f.index.dayofweek
    f["month"] = f.index.month

    if sentiment_df is not None and not sentiment_df.empty:
        news = _merge_sentiment(f.index, sentiment_df)
        for col in NEWS_FEATURE_COLS:
            f[col] = news[col]

    if market_df is not None and not market_df.empty:
        _merge_market(f, market_df)

    return f.replace([np.inf, -np.inf], np.nan)


def make_supervised(df: pd.DataFrame, horizon: int, sentiment_df: pd.DataFrame | None = None,
                    market_df: pd.DataFrame | None = None):
    """Build (X, y, close, dates, last_valid, feature_cols) for H-day-ahead returns.

    ``y[i]`` is the fractional return from ``close[i]`` to ``close[i + H]`` and
    is only defined for ``i < len - H`` (NaN elsewhere). News and market-context
    features are included in ``X``/``feature_cols`` when their frames are given.
    """
    with_news = sentiment_df is not None and not sentiment_df.empty
    with_market = market_df is not None and not market_df.empty
    market_cols = _market_columns(market_df) if with_market else None
    cols = feature_columns(with_news, market_cols)
    feats = build_features(df, sentiment_df, market_df).dropna(subset=cols + ["Close"])
    X = feats[cols].to_numpy(dtype=float)
    close = feats["Close"].to_numpy(dtype=float)
    dates = feats.index
    n = len(feats)
    y = np.full(n, np.nan)
    last_valid = n - horizon
    for i in range(max(0, last_valid)):
        y[i] = close[i + horizon] / close[i] - 1.0
    return X, y, close, dates, last_valid, cols
