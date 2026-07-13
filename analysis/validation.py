"""Data-quality validation for OHLCV frames.

Bad market data (zero/negative prices, duplicated or out-of-order dates,
absurd single-day jumps, all-zero volume) silently corrupts every downstream
indicator and ML feature. This module sanitises frames and reports what it
found so the API/UI can surface data-quality warnings instead of pretending
the numbers are clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .logging_config import get_logger

log = get_logger(__name__)

_OHLC = ["Open", "High", "Low", "Close"]
# A single-session move larger than this (absolute fraction) is almost always
# a bad tick or an unadjusted split, not a real return.
_MAX_DAILY_MOVE = 0.60


@dataclass
class QualityReport:
    ok: bool = True
    rows_in: int = 0
    rows_out: int = 0
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped": self.rows_in - self.rows_out,
            "issues": self.issues,
        }


# Fixed NSE trading holidays are impractical to hard-code far into the future,
# so we approximate the expected trading calendar with business days (Mon-Fri).
# This slightly over-counts (flags exchange holidays as "missing"), so callers
# treat the count as an upper bound / staleness signal, not a hard error.
def expected_sessions(start, end) -> int:
    """Approximate NSE trading sessions between two dates (inclusive), Mon-Fri."""
    try:
        return int(np.busday_count(pd.Timestamp(start).date(),
                                   pd.Timestamp(end).date()) + 1)
    except Exception:  # noqa: BLE001
        return 0


def missing_sessions(df: pd.DataFrame, max_gap_report: int = 10) -> dict:
    """Detect gaps in a daily OHLCV index against the business-day calendar.

    Returns a summary with the expected vs. present session counts, the number
    of missing business days, and the largest contiguous gaps (so the UI can
    warn about suspended/illiquid instruments or provider outages). Weekend-only
    gaps are ignored; exchange holidays may be over-counted (see module note).
    """
    if df is None or df.empty:
        return {"expected": 0, "present": 0, "missing": 0, "coverage_pct": None,
                "largest_gaps": []}
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    present = len(idx)
    exp = expected_sessions(idx.min(), idx.max())
    full = pd.bdate_range(idx.min(), idx.max())
    missing_days = full.difference(idx.normalize())
    gaps = []
    if len(missing_days):
        # Group consecutive missing business days into runs.
        run_start = prev = missing_days[0]
        for d in missing_days[1:]:
            if (d - prev).days <= 3:  # bridge weekends
                prev = d
                continue
            gaps.append((run_start, prev))
            run_start = prev = d
        gaps.append((run_start, prev))
        gaps.sort(key=lambda g: (g[1] - g[0]).days, reverse=True)
    largest = [
        {"from": a.strftime("%Y-%m-%d"), "to": b.strftime("%Y-%m-%d"),
         "sessions": int(np.busday_count(a.date(), b.date()) + 1)}
        for a, b in gaps[:max_gap_report]
    ]
    return {
        "expected": int(exp),
        "present": int(present),
        "missing": int(max(0, exp - present)),
        "coverage_pct": round(present / exp * 100, 1) if exp else None,
        "largest_gaps": largest,
    }


def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Return a cleaned copy of ``df`` plus a report of what was fixed.

    Cleaning steps (all non-destructive to good rows):
      - keep only OHLCV columns, coerce to numeric
      - drop rows without a Close
      - drop non-positive prices
      - de-duplicate dates (keep last) and sort ascending
      - repair High/Low envelope so High>=max(O,C) and Low<=min(O,C)
      - drop implausible single-day moves (bad ticks / unadjusted splits)
    """
    report = QualityReport()
    if df is None or df.empty:
        report.ok = False
        report.issues.append("empty")
        return pd.DataFrame(), report

    out = df.copy()
    report.rows_in = len(out)

    keep = [c for c in _OHLC + ["Volume"] if c in out.columns]
    out = out[keep]
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "Close" not in out.columns:
        report.ok = False
        report.issues.append("no_close_column")
        return pd.DataFrame(), report

    before = len(out)
    out = out.dropna(subset=["Close"])
    if len(out) < before:
        report.issues.append(f"dropped_{before - len(out)}_rows_missing_close")

    # Non-positive prices are invalid.
    price_cols = [c for c in _OHLC if c in out.columns]
    bad_price = (out[price_cols] <= 0).any(axis=1)
    if bad_price.any():
        report.issues.append(f"dropped_{int(bad_price.sum())}_nonpositive_price_rows")
        out = out[~bad_price]

    # Ensure a sorted, unique DatetimeIndex.
    if not out.index.is_monotonic_increasing:
        report.issues.append("reordered_dates")
        out = out.sort_index()
    dup = out.index.duplicated(keep="last")
    if dup.any():
        report.issues.append(f"deduped_{int(dup.sum())}_dates")
        out = out[~dup]

    # Repair the High/Low envelope where OHLC is internally inconsistent.
    if set(_OHLC).issubset(out.columns):
        hi = out[["Open", "Close", "High"]].max(axis=1)
        lo = out[["Open", "Close", "Low"]].min(axis=1)
        fixed = int((hi != out["High"]).sum() + (lo != out["Low"]).sum())
        if fixed:
            report.issues.append(f"repaired_{fixed}_high_low_bounds")
        out["High"] = hi
        out["Low"] = lo

    # Flag & drop implausible single-day moves (likely bad ticks / raw splits).
    if len(out) > 2:
        ret = out["Close"].pct_change().abs()
        spikes = ret > _MAX_DAILY_MOVE
        if spikes.any():
            report.issues.append(f"dropped_{int(spikes.sum())}_price_spikes")
            out = out[~spikes]

    if "Volume" in out.columns:
        out["Volume"] = out["Volume"].fillna(0).clip(lower=0)
        if (out["Volume"] == 0).all():
            report.issues.append("all_zero_volume")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    report.rows_out = len(out)
    report.ok = report.rows_out > 0
    if report.issues:
        log.debug("data quality issues: %s", ", ".join(report.issues))
    return out, report
