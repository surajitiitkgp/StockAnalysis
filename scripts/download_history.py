"""Bulk-download daily OHLCV history for the whole NSE universe into SQLite.

Downloads up to 10 years of daily bars per stock from Yahoo Finance (in batches)
and stores them in ``analysis/data/history.db`` via ``analysis.store``. The job is
**resumable**: symbols already up to date are skipped unless ``--force`` is set.

Usage
-----
    python scripts/download_history.py                 # full universe, 10y
    python scripts/download_history.py --period 5y     # shorter window
    python scripts/download_history.py --limit 100     # first 100 symbols
    python scripts/download_history.py --symbols RELIANCE TCS INFY
    python scripts/download_history.py --force         # re-download everything
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Make the project importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import store, universe  # noqa: E402


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=["Close"])
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def _is_fresh(symbol: str, max_age_days: int = 4) -> bool:
    last = store.last_date(symbol)
    if not last:
        return False
    try:
        last_dt = datetime.strptime(last, "%Y-%m-%d")
    except ValueError:
        return False
    return (datetime.now() - last_dt) <= timedelta(days=max_age_days)


def _run_pass(todo, period, batch_size, pause, label="pass"):
    """Download one pass over ``todo``. Returns (written, ok, failed_list)."""
    written_total = 0
    ok = 0
    failed: list[str] = []
    n_batches = (len(todo) + batch_size - 1) // batch_size

    for bi in range(n_batches):
        batch = todo[bi * batch_size:(bi + 1) * batch_size]
        tickers = [universe.to_yahoo(s, "NSE") for s in batch]
        try:
            data = yf.download(
                tickers, period=period, interval="1d", group_by="ticker",
                auto_adjust=False, threads=True, progress=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{label} {bi+1}/{n_batches}] download error: {exc}", flush=True)
            failed.extend(batch)
            time.sleep(pause * 3)
            continue

        for sym, tk in zip(batch, tickers):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if tk not in data.columns.get_level_values(0):
                        failed.append(sym)
                        continue
                    sub = data[tk]
                else:
                    sub = data  # single-ticker batch
                sub = _clean(sub)
                if sub.empty:
                    failed.append(sym)
                    continue
                written_total += store.upsert_history(sym, sub)
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  {sym}: parse error: {exc}", flush=True)
                failed.append(sym)

        print(f"[{label} {bi+1}/{n_batches}] ok={ok} failed={len(failed)} "
              f"rows={written_total}", flush=True)
        time.sleep(pause)

    return written_total, ok, failed


def download(symbols, period="10y", batch_size=40, force=False, pause=1.0):
    store.init_db()
    todo = symbols if force else [s for s in symbols if not _is_fresh(s)]
    skipped = len(symbols) - len(todo)
    print(f"Universe: {len(symbols)} | to download: {len(todo)} | skipped (fresh): {skipped}",
          flush=True)

    written, ok, failed = _run_pass(todo, period, batch_size, pause, label="batch")

    # Retry pass for failures (often transient rate limiting) with smaller
    # batches and a longer pause. Many genuine failures are delisted symbols.
    if failed:
        print(f"Retrying {len(failed)} failed symbols with smaller batches…", flush=True)
        time.sleep(pause * 4)
        w2, ok2, failed = _run_pass(failed, period, max(5, batch_size // 4),
                                    pause * 2, label="retry")
        written += w2
        ok += ok2

    print("-" * 60, flush=True)
    print(f"Finished. ok={ok} still_failed={len(failed)} rows_written={written}", flush=True)
    if failed:
        print(f"Unavailable symbols ({len(failed)}): {', '.join(failed[:40])}"
              + (" ..." if len(failed) > 40 else ""), flush=True)
    print("DB stats:", store.stats(), flush=True)
    return ok, failed


def main():
    ap = argparse.ArgumentParser(description="Download NSE daily history into SQLite.")
    ap.add_argument("--period", default="10y", help="yfinance period (default 10y)")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="limit number of symbols")
    ap.add_argument("--symbols", nargs="*", default=None, help="specific base symbols")
    ap.add_argument("--force", action="store_true", help="re-download even if fresh")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between batches")
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = [d["symbol"] for d in universe.all_symbols()]
        if args.limit:
            symbols = symbols[: args.limit]

    t0 = time.time()
    download(symbols, period=args.period, batch_size=args.batch_size,
             force=args.force, pause=args.pause)
    print(f"Elapsed: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
