"""Broad-market context for the price models.

"The market moves the stock, and news moves the market." This module builds a
per-date context frame combining:

  - **Index dynamics** — returns and volatility of the broad market index
    (NIFTY 50 ``^NSEI`` for NSE, SENSEX ``^BSESN`` for BSE), fetched via the
    resilient provider chain.
  - **Global sentiment** — the market/geopolitical sentiment archive
    (``__MARKET__`` in :mod:`analysis.store`) that the news layer accumulates.

The result is merged into every stock's feature matrix (see
:mod:`analysis.features`), so a single macro/news signal informs the whole
universe — even for stocks that have no per-company news coverage.

Everything is causal (trailing only) and degrades gracefully: if the index or
sentiment is unavailable, the missing columns are simply neutral-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import cache, nse_client, providers, store
from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)

INDEX_TICKERS = {"NSE": "^NSEI", "BSE": "^BSESN"}

MARKET_FEATURE_COLS = [
    "mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20",
    "mkt_sent_1d", "mkt_sent_7d",
]

# India VIX columns, appended to the context only when NSE data is reachable.
VIX_FEATURE_COLS = ["mkt_vix", "mkt_vix_chg_5"]


def _index_features(close: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    out["mkt_ret_1"] = close.pct_change(1)
    out["mkt_ret_5"] = close.pct_change(5)
    out["mkt_ret_20"] = close.pct_change(20)
    out["mkt_vol_20"] = close.pct_change().rolling(20).std()
    return out


def _compute_context(exchange: str, period: str) -> pd.DataFrame:
    ticker = INDEX_TICKERS.get(exchange.upper(), "^NSEI")
    try:
        idx_df, _ = providers.get_daily(ticker, period)
    except Exception:  # noqa: BLE001
        log.info("market index fetch failed for %s", ticker, exc_info=True)
        idx_df = pd.DataFrame()

    if idx_df is not None and not idx_df.empty:
        ctx = _index_features(idx_df["Close"])
    else:
        ctx = pd.DataFrame(columns=["mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_vol_20"])

    # Global market/geopolitical sentiment archive.
    try:
        sent = store.get_sentiment(store.MARKET_SYMBOL)
    except Exception:  # noqa: BLE001
        sent = pd.DataFrame()

    if sent is not None and not sent.empty:
        s = sent["sentiment"]
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        # Align onto the index calendar (or the sentiment's own dates if no index).
        base_index = ctx.index if not ctx.empty else s.index
        daily = s.reindex(base_index).ffill().fillna(0.0)
        ctx = ctx.reindex(base_index) if not ctx.empty else pd.DataFrame(index=base_index)
        ctx["mkt_sent_1d"] = daily
        ctx["mkt_sent_7d"] = daily.rolling(7, min_periods=1).mean()
    else:
        ctx["mkt_sent_1d"] = 0.0
        ctx["mkt_sent_7d"] = 0.0

    _merge_vix(ctx, period)

    return ctx.replace([np.inf, -np.inf], np.nan)


def _merge_vix(ctx: pd.DataFrame, period: str) -> None:
    """Append India VIX level + 5-day change onto ``ctx`` in place (best-effort).

    VIX is a market-wide volatility / "fear" gauge, sourced directly from NSE.
    Aligned causally onto the index calendar; skipped silently when the NSE API
    is disabled or unreachable so the base context still works everywhere.
    """
    if not (settings.use_vix_feature and nse_client.is_enabled()) or ctx.empty:
        return
    try:
        vix = nse_client.vix_history(period)
    except Exception:  # noqa: BLE001
        log.info("India VIX fetch failed", exc_info=True)
        return
    if vix is None or vix.empty:
        return
    if getattr(vix.index, "tz", None) is not None:
        vix.index = vix.index.tz_localize(None)
    aligned = vix.reindex(ctx.index.union(vix.index)).sort_index().ffill().reindex(ctx.index)
    ctx["mkt_vix"] = aligned
    ctx["mkt_vix_chg_5"] = aligned.pct_change(5)


def get_market_context(exchange: str = "NSE", period: str = "max") -> pd.DataFrame:
    """Cached per-date market context frame (index dynamics + global sentiment)."""
    if not settings.use_market_features:
        return pd.DataFrame()
    key = f"mktctx:{exchange.upper()}:{period}"
    result = cache.get_or_compute(
        key, settings.ml_cache_ttl, lambda: _compute_context(exchange, period))
    return result if result is not None else pd.DataFrame()


def context_signature(ctx: pd.DataFrame) -> str:
    """Compact signature for cache/bundle invalidation.

    Includes the column set so bundles are rebuilt when a feature group (e.g.
    India VIX) becomes available or drops out.
    """
    if ctx is None or ctx.empty:
        return "none"
    last = ctx.index.max()
    cols = ",".join(sorted(ctx.columns))
    return f"{len(ctx)}:{last.strftime('%Y-%m-%d')}:{cols}"
