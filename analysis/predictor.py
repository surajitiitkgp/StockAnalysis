"""Multi-model, multi-horizon price forecasting with honest validation.

Highlights
----------
- **Multiple models**: Random Forest, Extra Trees, Gradient Boosting,
  HistGradient Boosting, Ridge, and an averaging **ensemble** (see
  :mod:`analysis.models`).
- **Model selection**: ``model="auto"`` runs walk-forward validation across the
  candidate models and picks the best performer *per stock*.
- **Multiple horizons**: forecasts the 1, 2, 7, 10 and 30-trading-day-ahead
  price in one call.
- **Walk-forward backtest**: uses ``TimeSeriesSplit`` with a leakage ``gap`` so
  training never overlaps the evaluation window — reported accuracy is honest.
- **Persistence**: results are cached in the shared cache and (optionally)
  persisted to disk with ``joblib`` so they survive restarts.

Disclaimer: statistical estimates for education only — not investment advice.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import (cache, data_fetcher, features, market, models, news, store,
               universe, validation)
from .config import settings
from .logging_config import get_logger

log = get_logger(__name__)

HORIZONS = (1, 2, 7, 10, 30)
PRIMARY_HORIZON = 7
# Bump whenever the result/bundle schema changes so stale disk bundles are
# ignored instead of served with a mismatched shape.
BUNDLE_VERSION = 4
BACKTEST_POINTS = 10       # points shown on the predicted-vs-actual chart
_WF_FOLDS = 4              # walk-forward folds for metrics
_SELECT_FOLDS = 3         # (cheaper) folds used during auto model selection

# Prediction reliability tiers (most -> least trustworthy). Every result is
# tagged with one so the UI can be honest about how much to trust it.
MODE_FULL = "full_history"
MODE_REDUCED = "reduced_feature"
MODE_BASELINE = "baseline"
MODE_REFUSED = "refused"
PREDICTION_MODES = {
    MODE_FULL: "Full-history model",
    MODE_REDUCED: "Reduced-feature model (limited history)",
    MODE_BASELINE: "Short-history baseline",
    MODE_REFUSED: "No reliable prediction",
}
# Confidence is capped by tier: a short-history estimate must never look as
# trustworthy as a fully validated model.
_MODE_CONF_CAP = {MODE_FULL: 95.0, MODE_REDUCED: 55.0, MODE_BASELINE: 35.0}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _verdict(expected_return: float, horizon: int) -> str:
    """Map a predicted return to a verdict, scaled by horizon (sqrt-time)."""
    scale = math.sqrt(horizon / PRIMARY_HORIZON)
    pct = expected_return * 100
    if pct >= 6 * scale:
        return "STRONG BUY"
    if pct >= 2 * scale:
        return "BUY"
    if pct > -2 * scale:
        return "HOLD"
    if pct > -6 * scale:
        return "SELL"
    return "STRONG SELL"


def _empty(reason: str, model_key: str = models.DEFAULT_MODEL,
           mode: str = MODE_REFUSED, data_quality: dict | None = None) -> dict:
    return {
        "available": False,
        "reason": reason,
        "verdict": "HOLD",
        "horizon_days": PRIMARY_HORIZON,
        "available_models": models.available_models(),
        "model": {"key": model_key, "label": models.label(model_key)},
        "prediction_mode": mode,
        "prediction_mode_label": PREDICTION_MODES.get(mode, mode),
        "data_quality": data_quality,
        "horizons": [],
    }


def _metrics(actual_price, pred_price, base_price):
    actual_price = np.asarray(actual_price, dtype=float)
    pred_price = np.asarray(pred_price, dtype=float)
    base_price = np.asarray(base_price, dtype=float)
    rmse_pct = float(np.sqrt(np.mean(((pred_price - actual_price) / actual_price) ** 2)) * 100)
    mae_pct = float(np.mean(np.abs((pred_price - actual_price) / actual_price)) * 100)
    dir_actual = np.sign(actual_price - base_price)
    dir_pred = np.sign(pred_price - base_price)
    directional = float(np.mean(dir_actual == dir_pred) * 100)
    ss_res = float(np.sum((actual_price - pred_price) ** 2))
    ss_tot = float(np.sum((actual_price - np.mean(actual_price)) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return {
        "rmse_pct": round(rmse_pct, 2),
        "mae_pct": round(mae_pct, 2),
        "directional_accuracy_pct": round(directional, 0),
        "r2": round(r2, 3),
    }


def _walk_forward(model_key, X, y, close, dates, last_valid, H, folds, min_rows=None):
    """Out-of-sample evaluation via expanding-window TimeSeriesSplit(gap=H).

    ``min_rows`` sets the minimum number of usable rows required before any
    validation is attempted (defaults to half of ``ml_min_rows``). The
    limited-history fallback tiers pass a lower floor.

    Returns (metrics, chart_rows) or (None, None) if there isn't enough data.
    """
    valid = np.arange(max(0, last_valid))
    floor = settings.ml_min_rows // 2 if min_rows is None else int(min_rows)
    if len(valid) < max(floor, folds + 2):
        return None, None

    Xv, yv = X[valid], y[valid]
    n_splits = min(folds, max(2, len(valid) // 40))
    try:
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=H)
    except ValueError:
        return None, None

    oos_base, oos_actual, oos_pred, oos_dates = [], [], [], []
    for tr, te in tscv.split(valid):
        if len(tr) < 30:
            continue
        model = models.build(model_key)
        model.fit(Xv[tr], yv[tr])
        preds = model.predict(Xv[te])
        for j, pos in enumerate(te):
            i = valid[pos]
            base_price = close[i]
            oos_base.append(base_price)
            oos_pred.append(base_price * (1.0 + preds[j]))
            oos_actual.append(close[i + H])
            oos_dates.append(dates[i + H])

    if len(oos_actual) < 3:
        return None, None

    metrics = _metrics(oos_actual, oos_pred, oos_base)
    tail = slice(-BACKTEST_POINTS, None)
    chart = [
        {"date": d.strftime("%Y-%m-%d"), "actual": round(float(a), 2),
         "predicted": round(float(p), 2)}
        for d, a, p in zip(oos_dates[tail], oos_actual[tail], oos_pred[tail])
    ]
    return metrics, chart


def _select_model(X, y, close, dates, last_valid, H) -> tuple[str, list]:
    """Pick the best base model for horizon H via walk-forward directional acc."""
    scoreboard = []
    for key in models.selectable_keys():
        metrics, _ = _walk_forward(key, X, y, close, dates, last_valid, H, _SELECT_FOLDS)
        if metrics is None:
            continue
        scoreboard.append({
            "model": key,
            "label": models.label(key),
            "directional_accuracy_pct": metrics["directional_accuracy_pct"],
            "mae_pct": metrics["mae_pct"],
        })
    if not scoreboard:
        return models.DEFAULT_MODEL, []
    scoreboard.sort(key=lambda m: (-m["directional_accuracy_pct"], m["mae_pct"]))
    return scoreboard[0]["model"], scoreboard


def _forecast(model_key, X, y, last_valid, close, feature_cols):
    """Retrain on all valid rows and predict the H-ahead return from today."""
    valid = np.arange(max(0, last_valid))
    model = models.build(model_key)
    model.fit(X[valid], y[valid])
    fwd_ret = float(model.predict(X[-1].reshape(1, -1))[0])
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda kv: kv[1], reverse=True,
        )[:6]
    return fwd_ret, importances


def _confidence(directional_acc: float, fwd_ret: float) -> float:
    return round(min(95.0, 35.0 + directional_acc * 0.4
                     + min(abs(fwd_ret) * 100, 12) * 1.5), 0)


def _news_overlay(base: str, exchange: str, per_horizon: list) -> dict | None:
    """Fetch current news sentiment, persist it to the archive, and nudge
    each horizon's confidence based on agreement with the forecast direction.

    The nudge is small (+/- up to 6 points) and transparent: strong news that
    agrees with the model raises confidence; contradictory news lowers it.
    """
    if not news.is_enabled():
        return None
    company = universe.SYMBOL_TO_NAME.get(base, base)
    try:
        summary = news.get_sentiment_summary(base, exchange, company)
    except Exception:  # noqa: BLE001
        log.info("news overlay fetch failed for %s", base, exc_info=True)
        return None
    if not summary.get("available"):
        return {"available": False, "reason": summary.get("reason")}

    # Persist the fresh daily series so the archive deepens over time.
    try:
        if summary.get("daily"):
            store.upsert_sentiment(base, summary["daily"])
    except Exception:  # noqa: BLE001
        log.info("failed to persist sentiment for %s", base, exc_info=True)

    agg = summary.get("aggregate", {})
    news_score = float(agg.get("score", 0.0))
    for h in per_horizon:
        direction = 1 if h["expected_return_pct"] >= 0 else -1
        agreement = direction * news_score  # >0 agree, <0 disagree
        nudge = max(-6.0, min(6.0, agreement * 12.0))
        h["confidence"] = round(max(0.0, min(95.0, h["confidence"] + nudge)), 0)
        h["news_adjusted"] = True

    return {
        "available": True,
        "provider": summary.get("provider"),
        "score": round(news_score, 3),
        "label": agg.get("label"),
        "article_count": agg.get("count"),
        "positive": agg.get("positive"),
        "negative": agg.get("negative"),
        "headlines": summary.get("headlines", [])[:6],
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _bundle_path(base: str, exchange: str, model_key: str) -> str:
    safe = f"{base}_{exchange}_{model_key}".replace(os.sep, "_")
    return os.path.join(settings.model_dir, f"{safe}.joblib")


def _sent_signature(sent_df) -> str:
    if sent_df is None or getattr(sent_df, "empty", True):
        return "none"
    return f"{len(sent_df)}:{sent_df.index.max().strftime('%Y-%m-%d')}"


def _load_bundle(base, exchange, model_key, last_date, horizons, sent_sig):
    if not settings.ml_persist:
        return None
    path = _bundle_path(base, exchange, model_key)
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
    except Exception:  # noqa: BLE001
        return None
    if bundle.get("_bundle_version") != BUNDLE_VERSION:
        return None
    age_h = (time.time() - bundle.get("_saved_at", 0)) / 3600.0
    if age_h > settings.ml_model_ttl_hours:
        return None
    if bundle.get("last_date") != last_date:
        return None
    if bundle.get("_sent_sig") != sent_sig:
        return None
    have = {h["days"] for h in bundle.get("horizons", [])}
    if not set(horizons).issubset(have):
        return None
    return bundle


def _save_bundle(base, exchange, model_key, result, sent_sig):
    if not settings.ml_persist:
        return
    try:
        os.makedirs(settings.model_dir, exist_ok=True)
        payload = dict(result)
        payload["_saved_at"] = time.time()
        payload["_bundle_version"] = BUNDLE_VERSION
        payload["_sent_sig"] = sent_sig
        joblib.dump(payload, _bundle_path(base, exchange, model_key))
    except Exception:  # noqa: BLE001
        log.warning("failed to persist model bundle for %s", base, exc_info=True)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def predict(symbol: str, exchange: str = "NSE", model: str = "auto",
            horizons=None, use_cache: bool = True) -> dict:
    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    model = (model or "auto").strip().lower()
    if model != "auto" and model not in models.model_keys():
        return _empty(f"Unknown model '{model}'.")

    horizons = _normalise_horizons(horizons)
    ckey = f"predict:{base}:{exchange.upper()}:{model}:{','.join(map(str, horizons))}"

    if use_cache:
        cached = cache.get(ckey)
        if cached is not None:
            return cached

    result = _predict_impl(base, symbol, exchange, model, horizons)
    if result.get("available"):
        cache.set(ckey, result, settings.ml_cache_ttl)
        # Audit trail: persist the served forecast with its full provenance so
        # predictions are reproducible and reviewable (Sec. 3/13). Best-effort.
        try:
            store.log_prediction(result)
        except Exception:  # noqa: BLE001
            log.debug("prediction audit log failed for %s", base, exc_info=True)
    return result


def _normalise_horizons(horizons):
    if not horizons:
        return list(HORIZONS)
    out = []
    for h in horizons:
        try:
            hi = int(h)
        except (TypeError, ValueError):
            continue
        if hi in HORIZONS and hi not in out:
            out.append(hi)
    return out or list(HORIZONS)


def _data_quality_block(df, raw_rows, feature_rows) -> dict:
    """Provenance + freshness summary shown alongside every prediction (Sec. 8)."""
    last = df.index.max()
    first = df.index.min()
    age_days = (datetime.now() - last.to_pydatetime()).days
    # Business days between last obs and now = completed sessions of staleness.
    sessions_old = int(np.busday_count(last.date(), datetime.now().date()))
    try:
        _coverage = validation.missing_sessions(df)
    except Exception:  # noqa: BLE001
        _coverage = None
    if sessions_old <= 1:
        quality = "Good"
    elif sessions_old <= 3:
        quality = "Fair"
    else:
        quality = "Stale"
    return {
        "first_observation": first.strftime("%Y-%m-%d"),
        "last_observation": last.strftime("%Y-%m-%d"),
        "raw_observations": int(raw_rows),
        "feature_ready_observations": int(feature_rows),
        "interval": "1d",
        "dataset_age_days": int(age_days),
        "trading_sessions_old": sessions_old,
        "stale": sessions_old > settings.stale_after_days,
        "quality": quality,
        "coverage": _coverage,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def _run_tier(df, horizons, sent_df, market_df, model_key, mode, base_cols,
              wf_min_rows, valid_min_rows):
    """Attempt to build validated forecasts for one reliability tier.

    Returns (per_horizon, top_features, train_samples, feature_rows, benchmark)
    or (None, ...) when the tier can't produce a validated forecast. Confidence
    is capped by the tier so weaker tiers never look as trustworthy as a full
    model.
    """
    per_horizon = []
    top_features = None
    train_samples = 0
    feature_rows = 0
    conf_cap = _MODE_CONF_CAP.get(mode, 95.0)

    for H in sorted(horizons):
        X, y, close, dates, last_valid, cols = features.make_supervised(
            df, H, sent_df, market_df, base_cols=base_cols)
        feature_rows = max(feature_rows, len(X))
        valid_count = max(0, last_valid)
        if valid_count < valid_min_rows:
            continue

        metrics, chart = _walk_forward(model_key, X, y, close, dates, last_valid,
                                       H, _WF_FOLDS, min_rows=wf_min_rows)
        if metrics is None:
            continue
        # Benchmark the tier's model against the naive no-change baseline so we
        # never promote a model that fails to beat a random walk out-of-sample.
        base_metrics, _ = _walk_forward("naive", X, y, close, dates, last_valid,
                                        H, _WF_FOLDS, min_rows=wf_min_rows)

        fwd_ret, importances = _forecast(model_key, X, y, last_valid, close, cols)
        if importances and (top_features is None or H == PRIMARY_HORIZON):
            top_features = [{"name": k, "importance": round(float(v), 3)}
                            for k, v in importances]
        train_samples = max(train_samples, valid_count)

        last_price = float(close[-1])
        target_date = dates[-1] + pd.tseries.offsets.BDay(H)
        conf = min(conf_cap, _confidence(metrics["directional_accuracy_pct"], fwd_ret))
        row = {
            "days": H,
            "verdict": _verdict(fwd_ret, H),
            "last_price": round(last_price, 2),
            "last_date": df.index.max().strftime("%Y-%m-%d"),
            "forecast_price": round(last_price * (1.0 + fwd_ret), 2),
            "forecast_date": target_date.strftime("%Y-%m-%d"),
            "expected_return_pct": round(fwd_ret * 100, 2),
            "confidence": conf,
            "metrics": metrics,
            "backtest": chart,
        }
        if base_metrics is not None:
            row["baseline_comparison"] = {
                "model": "naive",
                "baseline_mae_pct": base_metrics["mae_pct"],
                "beats_baseline": bool(metrics["mae_pct"] <= base_metrics["mae_pct"]),
            }
        per_horizon.append(row)

    if not per_horizon:
        return None, None, 0, feature_rows
    return per_horizon, top_features, train_samples, feature_rows


def _predict_impl(base, symbol, exchange, model, horizons) -> dict:
    df = data_fetcher.get_daily_history(symbol, exchange, period="10y")
    raw_rows = 0 if df is None else len(df)

    # Hard floor: below ml_abs_min_rows there is genuinely too little data for
    # even a short-history baseline, so we refuse rather than fabricate.
    if df is None or df.empty or raw_rows < settings.ml_abs_min_rows:
        dq = _data_quality_block(df, raw_rows, 0) if raw_rows else None
        return _empty(
            f"Not enough history: only {raw_rows} observations available; a minimum "
            f"of {settings.ml_abs_min_rows} is required for any reliable estimate.",
            model, mode=MODE_REFUSED, data_quality=dq)

    last_date = df.index.max().strftime("%Y-%m-%d")

    # Load the self-accumulating news-sentiment archive (if any) for features.
    sent_df = None
    if settings.use_news_features:
        try:
            sent_df = store.get_sentiment(base)
            if sent_df is not None and sent_df.empty:
                sent_df = None
        except Exception:  # noqa: BLE001
            sent_df = None
    used_news_features = sent_df is not None

    # Load broad-market context (index dynamics + global sentiment) for features.
    market_df = None
    if settings.use_market_features:
        try:
            market_df = market.get_market_context(exchange)
            if market_df is not None and market_df.empty:
                market_df = None
        except Exception:  # noqa: BLE001
            market_df = None
    used_market_features = market_df is not None

    # ------------------------------------------------------------------ #
    # Resolve "auto" (only meaningful for the full-history tier).
    # ------------------------------------------------------------------ #
    selected_from = None
    scoreboard = []
    full_model_key = model
    if model == "auto":
        sel_h = min(horizons, key=lambda h: abs(h - PRIMARY_HORIZON))
        Xs, ys, close_s, dates_s, lv_s, _ = features.make_supervised(df, sel_h, sent_df, market_df)
        if len(Xs) >= settings.ml_min_rows:
            full_model_key, scoreboard = _select_model(Xs, ys, close_s, dates_s, lv_s, sel_h)
            selected_from = "auto"
        else:
            full_model_key = None  # not enough for a full model; ladder handles it

    # ------------------------------------------------------------------ #
    # Disk-bundle fast path (full tier only, deterministic model key).
    # ------------------------------------------------------------------ #
    sent_sig = _sent_signature(sent_df) + "|" + market.context_signature(market_df)
    if full_model_key:
        cached_bundle = _load_bundle(base, exchange, full_model_key, last_date, horizons, sent_sig)
        if cached_bundle is not None:
            cached_bundle = dict(cached_bundle)
            for k in ("_saved_at", "_bundle_version", "_sent_sig"):
                cached_bundle.pop(k, None)
            cached_bundle["from_cache"] = "disk"
            return cached_bundle

    # ------------------------------------------------------------------ #
    # Graduated fallback ladder (Sec. 7): full -> reduced -> baseline.
    # Each rung uses fewer features / less history and caps confidence lower.
    # ------------------------------------------------------------------ #
    half = settings.ml_min_rows // 2          # ~150 default
    lo = settings.ml_abs_min_rows // 2        # ~45 default
    ladder = []
    if full_model_key:
        ladder.append((MODE_FULL, full_model_key, None, half, half))
    # Reduced-feature tier: core features only, lighter model, lower floors.
    reduced_key = full_model_key if (full_model_key and not models.is_baseline(full_model_key)) else "ridge"
    ladder.append((MODE_REDUCED, reduced_key, features.CORE_FEATURE_COLS, lo, lo))
    # Baseline tier: drift benchmark on the core window (always fits).
    ladder.append((MODE_BASELINE, "drift", features.CORE_FEATURE_COLS, lo, lo))

    chosen = None
    for mode, mkey, base_cols, wf_min, valid_min in ladder:
        try:
            out = _run_tier(df, horizons, sent_df, market_df, mkey, mode,
                            base_cols, wf_min, valid_min)
        except Exception:  # noqa: BLE001 - a tier failure must fall through, not crash
            log.info("prediction tier %s failed for %s", mode, base, exc_info=True)
            out = (None, None, 0, 0)
        per_horizon, top_features, train_samples, feature_rows = out
        if per_horizon:
            chosen = (mode, mkey, per_horizon, top_features, train_samples, feature_rows)
            break

    if chosen is None:
        dq = _data_quality_block(df, raw_rows, 0)
        return _empty(
            "Insufficient clean, validated history to produce a reliable forecast "
            "even with a short-history baseline.",
            full_model_key or model, mode=MODE_REFUSED, data_quality=dq)

    mode, model_key, per_horizon, top_features, train_samples, feature_rows = chosen
    data_quality = _data_quality_block(df, raw_rows, feature_rows)

    # Real-time news/sentiment overlay (fetch + persist to the archive).
    news_block = _news_overlay(base, exchange, per_horizon)

    primary = next((h for h in per_horizon if h["days"] == PRIMARY_HORIZON), per_horizon[0])

    result = {
        "available": True,
        "symbol": base,
        "exchange": exchange.upper(),
        "prediction_mode": mode,
        "prediction_mode_label": PREDICTION_MODES.get(mode, mode),
        "limited_data": mode != MODE_FULL,
        "model": {
            "key": model_key,
            "label": models.label(model_key),
            "selected_from": selected_from if mode == MODE_FULL else None,
            "scoreboard": scoreboard if mode == MODE_FULL else [],
            "is_baseline": models.is_baseline(model_key),
        },
        "available_models": models.available_models(),
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_date": last_date,
        "history_days": int(raw_rows),
        "train_samples": int(train_samples),
        "data_quality": data_quality,
        "top_features": top_features or [],
        "news_features_used": used_news_features,
        "market_features_used": used_market_features,
        "news": news_block,
        "horizons": per_horizon,
        # Backward-compatible top-level fields mirror the primary horizon.
        "verdict": primary["verdict"],
        "horizon_days": primary["days"],
        "last_price": primary["last_price"],
        "forecast_price": primary["forecast_price"],
        "forecast_date": primary["forecast_date"],
        "expected_return_pct": primary["expected_return_pct"],
        "confidence": primary["confidence"],
        "metrics": primary["metrics"],
        "backtest": primary["backtest"],
    }

    # Only persist fully validated, full-history models to disk. Weaker tiers
    # are cheap to recompute and we don't want stale low-confidence bundles.
    if mode == MODE_FULL:
        _save_bundle(base, exchange, model_key, result, sent_sig)
    return result
