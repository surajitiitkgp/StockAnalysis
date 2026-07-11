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

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


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


# key -> (human label, factory, is_selectable_by_auto)
_REGISTRY = {
    "random_forest": ("Random Forest", _random_forest, True),
    "extra_trees": ("Extra Trees", _extra_trees, True),
    "gradient_boosting": ("Gradient Boosting", _gradient_boosting, True),
    "hist_gradient_boosting": ("HistGradient Boosting", _hist_gradient_boosting, True),
    "ridge": ("Ridge Regression", _ridge, True),
    "ensemble": ("Ensemble (RF+HGB+Ridge)", _ensemble, False),
}

DEFAULT_MODEL = "ensemble"


def model_keys() -> list[str]:
    return list(_REGISTRY.keys())


def selectable_keys() -> list[str]:
    """Models eligible for automatic selection (excludes the heavy ensemble)."""
    return [k for k, (_, _, auto) in _REGISTRY.items() if auto]


def label(key: str) -> str:
    entry = _REGISTRY.get(key)
    return entry[0] if entry else key


def build(key: str):
    """Instantiate a fresh model by key. Raises KeyError for unknown keys."""
    if key not in _REGISTRY:
        raise KeyError(f"unknown model '{key}'")
    return _REGISTRY[key][1]()


def available_models() -> list[dict]:
    """List models for the UI, with the ``auto`` selector first."""
    out = [{"key": "auto", "label": "Auto (best per stock)"}]
    for key, (lbl, _, _) in _REGISTRY.items():
        out.append({"key": key, "label": lbl})
    return out
