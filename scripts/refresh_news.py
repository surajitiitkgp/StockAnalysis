"""Populate the local news-sentiment archive (``sentiment_daily`` in SQLite).

Fetches recent company news for each symbol, scores it into a daily sentiment
series and upserts it into the store. Run it regularly (e.g. daily) so the
archive **deepens over time** — turning short free-tier news windows into a
growing historical dataset the ML models can train on.

Usage
-----
    python scripts/refresh_news.py                 # priority names (fast)
    python scripts/refresh_news.py --limit 200     # first 200 universe symbols
    python scripts/refresh_news.py --symbols RELIANCE TCS INFY
    python scripts/refresh_news.py --all           # entire universe (slow!)
    python scripts/refresh_news.py --market         # also refresh market/geo news
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import news, store, universe  # noqa: E402
from analysis.logging_config import get_logger  # noqa: E402

log = get_logger("refresh_news")


def refresh_symbols(symbols, pause=1.0):
    store.init_db()
    if not news.is_enabled():
        log.error("No news provider configured. Set an API key "
                  "(FINNHUB_API_KEY / GNEWS_API_KEY / NEWSDATA_API_KEY / NEWSAPI_AI_KEY).")
        return 0, len(symbols)
    ok = fail = 0
    total_rows = 0
    for i, sym in enumerate(symbols, 1):
        company = universe.SYMBOL_TO_NAME.get(sym, sym)
        try:
            summary = news.get_sentiment_summary(sym, "NSE", company)
            if summary.get("available") and summary.get("daily"):
                total_rows += store.upsert_sentiment(sym, summary["daily"])
                ok += 1
            else:
                fail += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("news refresh failed for %s: %s", sym, exc)
            fail += 1
        if i % 10 == 0:
            log.info("progress %d/%d ok=%d fail=%d rows=%d", i, len(symbols), ok, fail, total_rows)
        time.sleep(pause)
    log.info("Finished. ok=%d fail=%d rows_written=%d", ok, fail, total_rows)
    log.info("Sentiment archive stats: %s", store.sentiment_stats())
    return ok, fail


def refresh_market():
    market = news.get_market_news()
    if market.get("available"):
        agg = market.get("aggregate", {})
        # Store the per-day sentiment series so it aligns with trading days.
        series = market.get("daily") or [{
            "date": time.strftime("%Y-%m-%d"),
            "sentiment": agg.get("score", 0.0), "count": agg.get("count", 0)}]
        rows = store.upsert_sentiment(store.MARKET_SYMBOL, series)
        log.info("Market news refreshed: score=%s count=%s (%d days written)",
                 agg.get("score"), agg.get("count"), rows)
    else:
        log.warning("Market news unavailable: %s", market.get("reason"))


def main():
    ap = argparse.ArgumentParser(description="Refresh the news-sentiment archive.")
    ap.add_argument("--symbols", nargs="*", default=None, help="specific base symbols")
    ap.add_argument("--limit", type=int, default=None, help="limit number of symbols")
    ap.add_argument("--all", action="store_true", help="entire universe (slow)")
    ap.add_argument("--market", action="store_true", help="also refresh market/geo news")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between symbols")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.all:
        symbols = [d["symbol"] for d in universe.all_symbols()]
    else:
        symbols = [d["symbol"] for d in universe.screener_symbols(args.limit or 60)]
    if args.limit:
        symbols = symbols[: args.limit]

    t0 = time.time()
    if args.market:
        refresh_market()
    refresh_symbols(symbols, pause=args.pause)
    log.info("Elapsed: %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
