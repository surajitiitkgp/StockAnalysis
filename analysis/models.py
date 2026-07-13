"""Model zoo for price forecasting.

A registry of reliable scikit-learn regressors plus a stacking/averaging
ensemble. Everything is scikit-learn only, so there are no fragile native
dependencies (XGBoost/LightGBM) to install. Linear models are wrapped in a
scaling pipeline so they're comparable to the tree models.

Use :func:`available_models` to list choices for the UI, :func:`build` to
instantiate one by key, and the ``"auto"`` key (handled in the predictor) to
let walk-forward validation pick the best performer per stock.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Baseline models (honest benchmarks + safe low-data fallbacks)
# --------------------------------------------------------------------------- #
class NaiveReturn(BaseEstimator, RegressorMixin):
    """Predict zero forward return (random-walk / no-change benchmark)."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class DriftMean(BaseEstimator, RegressorMixin):
    """Predict the mean training return (constant-drift benchmark)."""

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        m = float(np.nanmean(y)) if y.size else 0.0
        self.mean_ = m if np.isfinite(m) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), getattr(self, 'mean_', 0.0), dtype=float)


class FusionRegressor(BaseEstimator, RegressorMixin):
    """Multi-source fusion of price, news and geopolitics signals.

    An innovative, robust two-level ("stacked") learner:

    1. **Group specialists.** The feature matrix is partitioned by economic
       signal source (price/technical, company news, geopolitics/macro) using
       :func:`analysis.features.group_indices`. A gradient-boosted specialist is
       trained on *each* group so every source can model its own non-linear
       relationship to forward returns independently.

    2. **Leak-free meta-blender.** The specialists' predictions are combined by
       a **non-negative** Huber meta-regressor. Crucially, the meta-learner is
       trained on **out-of-fold** specialist predictions generated with an
       expanding-window ``TimeSeriesSplit`` (respecting the ``gap``), so the
       blend never sees a specialist's in-sample fit — this is what makes the
       stack honest on time-series data.

    Non-negativity + a robust (outlier-resistant) loss keep the blend
    interpretable and stable: the learned weights become the model's
    **signal attribution** (how much each source drives the forecast), exposed
    via :attr:`group_weights_` and :meth:`attribution`.

    The design degrades gracefully: with only one signal group present it
    reduces to that single specialist; missing groups simply don't get a weight.
    """

    def __init__(self, group_map: dict | None = None, gap: int = 1,
                 n_splits: int = 3, random_state: int = RANDOM_STATE):
        # group_map: {group_name: [feature indices]} — set by the predictor
        # from the actual feature columns in play.
        self.group_map = group_map
        self.gap = gap
        self.n_splits = n_splits
        self.random_state = random_state

    # -- specialist factory: strong on non-linearities, cheap to fit -------- #
    def _make_specialist(self):
        return HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.05,
            l2_regularization=1.0, random_state=self.random_state)

    def _resolved_groups(self, n_features: int) -> dict:
        if self.group_map:
            # Keep only groups whose indices are valid for this matrix.
            gm = {g: [i for i in idx if i < n_features]
                  for g, idx in self.group_map.items()}
            return {g: idx for g, idx in gm.items() if idx}
        # No map => treat everything as one "price" group (graceful default).
        return {"price": list(range(n_features))}

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, n_feat = X.shape
        groups = self._resolved_groups(n_feat)
        self.groups_ = list(groups.keys())
        self.group_cols_ = groups

        # --- generate out-of-fold specialist predictions (leak-free) ------- #
        oof = {g: np.full(n, np.nan) for g in self.groups_}
        n_splits = max(2, min(self.n_splits, max(2, n // 60)))
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=self.gap)
            splits = list(tscv.split(np.arange(n)))
        except ValueError:
            splits = []

        for tr, te in splits:
            if len(tr) < 30:
                continue
            for g, idx in groups.items():
                m = self._make_specialist()
                m.fit(X[np.ix_(tr, idx)], y[tr])
                oof[g][te] = m.predict(X[np.ix_(te, idx)])

        # Rows where every specialist has an OOF prediction feed the blender.
        oof_mat = np.column_stack([oof[g] for g in self.groups_])
        mask = ~np.isnan(oof_mat).any(axis=1)

        # --- fit the non-negative robust meta-blender ---------------------- #
        if mask.sum() >= max(20, len(self.groups_) + 5):
            meta = HuberRegressor(fit_intercept=True, alpha=0.0001, max_iter=500)
            meta.fit(oof_mat[mask], y[mask])
            w = np.clip(meta.coef_, 0.0, None)  # enforce non-negativity
            if w.sum() <= 1e-9:
                w = np.ones(len(self.groups_))
            self.meta_intercept_ = float(meta.intercept_)
        else:
            # Too few clean OOF rows: fall back to an equal-weight blend.
            w = np.ones(len(self.groups_))
            self.meta_intercept_ = 0.0

        self.raw_weights_ = w.astype(float)
        self.group_weights_ = dict(zip(self.groups_, (w / w.sum()).tolist()))

        # --- refit each specialist on ALL data for final inference --------- #
        self.specialists_ = {}
        for g, idx in groups.items():
            m = self._make_specialist()
            m.fit(X[:, idx], y)
            self.specialists_[g] = m
        return self

    def _specialist_matrix(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([
            self.specialists_[g].predict(X[:, self.group_cols_[g]])
            for g in self.groups_
        ])

    def predict(self, X):
        preds = self._specialist_matrix(X)
        return preds @ self.raw_weights_ / max(self.raw_weights_.sum(), 1e-9) \
            + self.meta_intercept_

    def attribution(self, X) -> dict:
        """Decompose the *latest* forecast into per-group contributions.

        Returns ``{group: signed_contribution}`` in return units, so the caller
        can report how much price / news / geopolitics each pushed the forecast
        up or down. Contributions sum (with the intercept) to the prediction.
        """
        preds = self._specialist_matrix(np.asarray(X, dtype=float)[-1:])
        total_w = max(self.raw_weights_.sum(), 1e-9)
        contrib = {}
        for j, g in enumerate(self.groups_):
            contrib[g] = float(preds[0, j] * self.raw_weights_[j] / total_w)
        return contrib


class QuantileBands(BaseEstimator, RegressorMixin):
    """P10 / P50 / P90 forecast bands via quantile gradient boosting.

    Gives an *uncertainty interval* around the point forecast instead of a bare
    number, so the UI can show a realistic price range. ``predict`` returns the
    median (P50) for API compatibility; :meth:`predict_bands` returns all three.
    """

    def __init__(self, quantiles=(0.1, 0.5, 0.9), random_state: int = RANDOM_STATE):
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(self, X, y):
        self.models_ = {}
        for q in self.quantiles:
            gbr = GradientBoostingRegressor(
                loss="quantile", alpha=q, n_estimators=150, max_depth=3,
                learning_rate=0.05, subsample=0.8, random_state=self.random_state)
            gbr.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
            self.models_[q] = gbr
        return self

    def predict(self, X):
        return self.models_[0.5].predict(np.asarray(X, dtype=float))

    def predict_bands(self, X) -> dict:
        X = np.asarray(X, dtype=float)
        # Enforce monotonicity (P10 <= P50 <= P90) to avoid crossed quantiles.
        lo = self.models_[0.1].predict(X)
        mid = self.models_[0.5].predict(X)
        hi = self.models_[0.9].predict(X)
        lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
        return {"p10": lo, "p50": mid, "p90": hi}


def _naive():
    return NaiveReturn()


def _drift():
    return DriftMean()



def _random_forest():
    return RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
    )


def _extra_trees():
    return ExtraTreesRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
    )


def _gradient_boosting():
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE,
    )


def _hist_gradient_boosting():
    return HistGradientBoostingRegressor(
        max_iter=250, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def _fusion():
    """Multi-source fusion model (price + news + geopolitics specialists)."""
    return FusionRegressor()


def build_fusion(group_map: dict, gap: int = 1):
    """Instantiate the fusion model bound to a concrete signal-group map."""
    return FusionRegressor(group_map=group_map, gap=gap)


def _ensemble():
    """Averaging ensemble of complementary base learners."""
    return VotingRegressor(
        estimators=[
            ("rf", _random_forest()),
            ("hgb", _hist_gradient_boosting()),
            ("ridge", _ridge()),
        ],
        n_jobs=-1,
    )


# key -> (human label, factory, is_selectable_by_auto, is_baseline)
_REGISTRY = {
    "random_forest": ("Random Forest", _random_forest, True, False),
    "extra_trees": ("Extra Trees", _extra_trees, True, False),
    "gradient_boosting": ("Gradient Boosting", _gradient_boosting, True, False),
    "hist_gradient_boosting": ("HistGradient Boosting", _hist_gradient_boosting, True, False),
    "ridge": ("Ridge Regression", _ridge, True, False),
    "fusion": ("Multi-Source Fusion (price+news+geopolitics)", _fusion, True, False),
    "ensemble": ("Ensemble (RF+HGB+Ridge)", _ensemble, False, False),
    "naive": ("Naive (no-change baseline)", _naive, False, True),
    "drift": ("Drift (mean-return baseline)", _drift, False, True),
}

DEFAULT_MODEL = "ensemble"
# Ordered fallback ladder for the limited-history workflow. Ridge needs the
# least data of the "real" models; drift/naive always fit. Tried in order.
FALLBACK_LADDER = ["ridge", "drift", "naive"]


def model_keys() -> list[str]:
    return list(_REGISTRY.keys())


def selectable_keys() -> list[str]:
    """Models eligible for automatic selection (excludes ensemble + baselines)."""
    return [k for k, (_, _, auto, _) in _REGISTRY.items() if auto]


def baseline_keys() -> list[str]:
    """Baseline benchmark models (naive / drift)."""
    return [k for k, (_, _, _, base) in _REGISTRY.items() if base]


def is_baseline(key: str) -> bool:
    entry = _REGISTRY.get(key)
    return bool(entry and entry[3])


def label(key: str) -> str:
    entry = _REGISTRY.get(key)
    return entry[0] if entry else key


def build(key: str):
    """Instantiate a fresh model by key. Raises KeyError for unknown keys."""
    if key not in _REGISTRY:
        raise KeyError(f"unknown model '{key}'")
    return _REGISTRY[key][1]()


def available_models() -> list[dict]:
    """List models for the UI, with the ``auto`` selector first.

    Baselines are advertised with ``baseline=True`` so the UI can group them
    apart from the primary user choices.
    """
    out = [{"key": "auto", "label": "Auto (best per stock)"}]
    for key, (lbl, _, _, base) in _REGISTRY.items():
        out.append({"key": key, "label": lbl, "baseline": base})
    return out
