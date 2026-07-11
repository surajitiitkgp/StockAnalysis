"""Tests for the GUI-editable runtime config layer and its API."""

from __future__ import annotations

import pytest

import app as app_module
from analysis import runtime_config
from analysis.config import settings


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Persist to a temp file and restore any settings mutated by a test."""
    monkeypatch.setattr(runtime_config, "_config_path",
                        lambda s: str(tmp_path / "runtime_config.json"))
    snapshot = {spec["key"]: getattr(settings, spec["key"], None)
                for spec in runtime_config.FIELD_SPECS}
    yield
    for key, val in snapshot.items():
        object.__setattr__(settings, key, val)


@pytest.fixture
def logged_in():
    app_module.app.config.update(TESTING=True)
    client = app_module.app.test_client()
    with client.session_transaction() as sess:
        sess["user"] = "tester"
        sess["_csrf_token"] = "test-csrf-token"
    return client


# --------------------------------------------------------------------------- #
# Coercion / validation
# --------------------------------------------------------------------------- #
def test_coerce_bool_and_int():
    spec_bool = runtime_config.SPEC_BY_KEY["news_enabled"]
    assert runtime_config._coerce(spec_bool, "on") is True
    assert runtime_config._coerce(spec_bool, False) is False

    spec_int = runtime_config.SPEC_BY_KEY["news_lookback_days"]
    assert runtime_config._coerce(spec_int, "30") == 30
    with pytest.raises(runtime_config.ConfigError):
        runtime_config._coerce(spec_int, "abc")
    with pytest.raises(runtime_config.ConfigError):
        runtime_config._coerce(spec_int, 9999)  # above max


def test_coerce_csv_to_tuple():
    spec = runtime_config.SPEC_BY_KEY["news_providers"]
    assert runtime_config._coerce(spec, "Finnhub, GNews ,") == ("finnhub", "gnews")


def test_update_applies_and_persists():
    res = runtime_config.update(settings, {"news_lookback_days": 45, "news_enabled": False})
    assert set(res["applied"]) == {"news_lookback_days", "news_enabled"}
    assert settings.news_lookback_days == 45
    assert settings.news_enabled is False
    # Persisted overrides reload cleanly onto a fresh apply.
    overrides = runtime_config.load_overrides(settings)
    assert overrides["news_lookback_days"] == 45


def test_update_rejects_unknown_key():
    with pytest.raises(runtime_config.ConfigError):
        runtime_config.update(settings, {"totally_made_up": 1})


def test_secret_blank_is_unchanged():
    object.__setattr__(settings, "finnhub_api_key", "existing")
    res = runtime_config.update(settings, {"finnhub_api_key": ""})
    assert "finnhub_api_key" not in res["applied"]
    assert settings.finnhub_api_key == "existing"


def test_secret_updates_when_provided():
    runtime_config.update(settings, {"gnews_api_key": "new-key"})
    assert settings.gnews_api_key == "new-key"


def test_current_values_masks_secrets():
    object.__setattr__(settings, "finnhub_api_key", "sekret")
    fields = {f["key"]: f for f in runtime_config.current_values(settings)}
    assert "value" not in fields["finnhub_api_key"]
    assert fields["finnhub_api_key"]["configured"] is True


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def test_api_config_requires_auth():
    client = app_module.app.test_client()
    assert client.get("/api/config").status_code == 401


def test_api_config_get(logged_in):
    data = logged_in.get("/api/config").get_json()
    keys = {f["key"] for f in data["settings"]}
    assert {"news_enabled", "use_vix_feature", "finnhub_api_key"}.issubset(keys)


def test_api_config_post_requires_csrf(logged_in):
    r = logged_in.post("/api/config", json={"changes": {"news_enabled": False}})
    assert r.status_code == 403


def test_api_config_post_applies(logged_in):
    r = logged_in.post(
        "/api/config",
        json={"changes": {"news_lookback_days": 21}},
        headers={"X-CSRF-Token": "test-csrf-token"},
    )
    assert r.status_code == 200
    assert "news_lookback_days" in r.get_json()["applied"]
    assert settings.news_lookback_days == 21


def test_api_config_post_validation_error(logged_in):
    r = logged_in.post(
        "/api/config",
        json={"changes": {"news_lookback_days": 99999}},
        headers={"X-CSRF-Token": "test-csrf-token"},
    )
    assert r.status_code == 400
