"""Single entry point — configure, initialise, warm, and serve.

Everything is wired automatically:

1. Loads config + secrets from ``.env`` (via ``analysis.config``).
2. Initialises the SQLite store (creates tables/schema if missing).
3. Warms the news/sentiment archive in the background (non-blocking) so the ML
   features and the News card have data, without delaying server start.
4. Starts the Flask web app.

Usage
-----
    python run.py                     # configure + serve (recommended)
    python run.py --refresh-news 25   # also fetch company news for 25 symbols now
    python run.py --no-warm           # skip the background news warm-up
    python run.py --port 8080         # override port

Everything degrades gracefully: with no API keys the app still runs, just
without news/sentiment.
"""

from __future__ import annotations

import argparse
import threading
import time

# Importing config first ensures the .env file is loaded before anything else.
from analysis import diagnostics, news, store, sync as sync_engine, universe
from analysis.config import settings
from analysis.logging_config import get_logger

log = get_logger("run")

# How often to refresh cheap market/geopolitical sentiment (hours). 0 = once.
_MARKET_REFRESH_HOURS = 6


def _warm_news(refresh_symbols: int) -> None:
    """Background warm-up: market sentiment now (+ periodically), and optionally
    a slice of company news. Kept off the request path so startup stays fast."""
    if not news.is_enabled():
        log.info("News disabled (no provider key) — skipping warm-up.")
        return
    log.info("News providers active: %s", news.status()["providers"])

    # Optional one-off company refresh (respects provider quotas — keep small).
    if refresh_symbols > 0:
        symbols = [d["symbol"] for d in universe.screener_symbols(refresh_symbols)]
        log.info("Warming company news for %d symbols...", len(symbols))
        written = 0
        for sym in symbols:
            company = universe.SYMBOL_TO_NAME.get(sym, sym)
            try:
                summary = news.get_sentiment_summary(sym, "NSE", company)
                if summary.get("available") and summary.get("daily"):
                    written += store.upsert_sentiment(sym, summary["daily"])
            except Exception:  # noqa: BLE001
                log.debug("warm company news failed for %s", sym, exc_info=True)
            time.sleep(0.3)
        log.info("Company news warm-up complete (%d rows).", written)

    # Periodic market/geopolitical refresh (cheap; one request per cycle).
    while True:
        try:
            market = news.get_market_news()
            if market.get("available"):
                agg = market.get("aggregate", {})
                # Persist the per-day series (aligns with trading days); fall
                # back to a single dated row if the series is empty.
                series = market.get("daily") or [{
                    "date": time.strftime("%Y-%m-%d"),
                    "sentiment": agg.get("score", 0.0), "count": agg.get("count", 0)}]
                store.upsert_sentiment(store.MARKET_SYMBOL, series)
                log.info("Market sentiment refreshed: score=%s (%s articles, %d days).",
                         agg.get("score"), agg.get("count"), len(series))
        except Exception:  # noqa: BLE001
            log.debug("market news refresh failed", exc_info=True)
        if _MARKET_REFRESH_HOURS <= 0:
            return
        time.sleep(_MARKET_REFRESH_HOURS * 3600)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Bharat Stocks app (auto-configured).")
    ap.add_argument("--host", default=settings.host)
    ap.add_argument("--port", type=int, default=settings.port)
    ap.add_argument("--debug", action="store_true", default=settings.debug)
    ap.add_argument("--no-warm", action="store_true", help="skip background news warm-up")
    ap.add_argument("--refresh-news", type=int, default=0, metavar="N",
                    help="fetch company news for N symbols at startup")
    ap.add_argument("--sync-now", action="store_true",
                    help="run an incremental data sync at startup (background)")
    ap.add_argument("--no-scheduler", action="store_true",
                    help="disable the daily post-close sync scheduler")
    ap.add_argument("--no-intraday", action="store_true",
                    help="disable the intraday (market-hours) price auto-refresh")
    args = ap.parse_args()

    log.info("Initialising store at %s", settings.db_path)
    store.init_db()
    log.info("Universe: %d symbols | cache: %s", universe.count(),
             "redis" if settings.redis_url else "in-memory")

    # Boot-time provider self-check: warns loudly (with fix guidance) if every
    # data source is unreachable — e.g. corporate SSL interception. Runs in the
    # background so it never delays server start; result is cached for /readyz
    # and /api/status.
    def _provider_check():
        try:
            import app as _app
            _app.PROVIDER_DIAGNOSIS = diagnostics.log_startup_check()
        except Exception:  # noqa: BLE001
            log.debug("provider self-check thread failed", exc_info=True)

    threading.Thread(target=_provider_check, name="provider-check", daemon=True).start()

    if not args.no_warm:
        threading.Thread(target=_warm_news, args=(args.refresh_news,),
                         name="news-warmup", daemon=True).start()

    # Optional one-off incremental sync at startup (non-blocking).
    if args.sync_now:
        threading.Thread(
            target=lambda: sync_engine.sync(limit=settings.sync_limit),
            name="startup-sync", daemon=True).start()

    # Daily post-close scheduler (Sec. 2). Honours config + CLI override.
    if settings.enable_scheduler and not args.no_scheduler:
        threading.Thread(target=sync_engine.run_scheduler,
                         name="sync-scheduler", daemon=True).start()
        log.info("Daily-sync scheduler enabled (%02d:%02d local).",
                 settings.sync_hour, settings.sync_minute)

    # Intraday auto-refresh: keep prices near-live during market hours (Sec. 2).
    if settings.enable_intraday_sync and not args.no_intraday:
        threading.Thread(target=sync_engine.run_intraday_scheduler,
                         name="intraday-sync-scheduler", daemon=True).start()
        log.info("Intraday price auto-refresh enabled (every %d min, %02d:%02d-%02d:%02d).",
                 settings.intraday_sync_minutes,
                 settings.market_open_hour, settings.market_open_minute,
                 settings.market_close_hour, settings.market_close_minute)

    # Import here so the Flask app is created after config/store are ready.
    from app import app
    log.info("Starting Bharat Stocks on http://%s:%s (debug=%s)",
             args.host, args.port, args.debug)
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
