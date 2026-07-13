"""Command-line incremental data sync (Sec. 2 / 17).

Brings the local SQLite archive up to date using the incremental sync engine
(:mod:`analysis.sync`). Safe to run repeatedly and from cron — it only fetches
missing bars and upserts, so it never duplicates data.

Usage
-----
    python scripts/sync_data.py                  # sync the most-liquid slice
    python scripts/sync_data.py --limit 500      # first 500 symbols
    python scripts/sync_data.py --symbols RELIANCE TCS INFY
    python scripts/sync_data.py --all            # entire universe
    python scripts/sync_data.py --force          # re-fetch full history
    python scripts/sync_data.py --backfill RELIANCE --period 10y   # one symbol

Exit code is non-zero if the run ends in an ``error`` state, so cron/CI can
detect failures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make the project importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import store, sync, universe  # noqa: E402
from analysis.config import settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Incremental NSE data sync into SQLite.")
    ap.add_argument("--symbols", nargs="*", default=None, help="specific base symbols")
    ap.add_argument("--limit", type=int, default=None, help="cap number of symbols")
    ap.add_argument("--all", action="store_true", help="sync the entire universe")
    ap.add_argument("--force", action="store_true", help="re-fetch full history")
    ap.add_argument("--backfill", metavar="SYMBOL", default=None,
                    help="backfill a single symbol's history")
    ap.add_argument("--period", default=None, help="history window for --backfill (e.g. 10y)")
    ap.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"])
    args = ap.parse_args()

    store.init_db()
    t0 = time.time()

    if args.backfill:
        # Historical backfill for a single stock (Sec. 2: selectable range).
        res = sync.sync_symbol(args.backfill, args.exchange, force=True)
        print(f"Backfill {args.backfill}: {res}", flush=True)
        print("DB stats:", store.stats(), flush=True)
        return 0 if res["status"] in ("updated", "up_to_date", "fresh") else 1

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.all:
        symbols = [d["symbol"] for d in universe.all_symbols()]
    else:
        symbols = None  # engine picks the most-liquid slice

    result = sync.sync(symbols=symbols,
                       limit=args.limit or (None if args.all else settings.sync_limit),
                       force=args.force, exchange=args.exchange)
    print("-" * 60, flush=True)
    print(f"Sync result: {result}", flush=True)
    print("DB stats:", store.stats(), flush=True)
    print(f"Elapsed: {time.time() - t0:.0f}s", flush=True)
    return 0 if result.get("status") in ("ok", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
