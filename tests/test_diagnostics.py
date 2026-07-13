"""Tests for the provider connectivity self-check (no network)."""

from __future__ import annotations

from analysis import diagnostics


def _patch_health(monkeypatch, rows):
    monkeypatch.setattr(diagnostics.providers, "provider_health", lambda *a, **k: rows)


def test_diagnose_ok(monkeypatch):
    _patch_health(monkeypatch, [{"name": "yahoo", "status": "ok", "rows": 5}])
    d = diagnostics.diagnose()
    assert d["ok"] is True
    assert d["severity"] == "ok"
    assert d["ssl_issue"] is False
    assert d["fixes"] == []


def test_diagnose_ssl_issue(monkeypatch):
    _patch_health(monkeypatch, [
        {"name": "yahoo", "status": "degraded",
         "detail": "curl: (60) SSL certificate problem: unable to get local issuer certificate"},
        {"name": "stooq", "status": "degraded", "detail": "stooq returned no usable data"},
    ])
    d = diagnostics.diagnose()
    assert d["ok"] is False
    assert d["ssl_issue"] is True
    assert d["severity"] == "offline"
    assert "SSL" in d["title"] or "certificate" in d["title"].lower()
    assert any("pip-system-certs" in f for f in d["fixes"])


def test_diagnose_generic_outage(monkeypatch):
    _patch_health(monkeypatch, [
        {"name": "yahoo", "status": "degraded", "detail": "timed out"},
        {"name": "stooq", "status": "degraded", "detail": "no data"},
    ])
    d = diagnostics.diagnose()
    assert d["ok"] is False
    assert d["ssl_issue"] is False
    assert d["severity"] in ("offline", "degraded")
    assert d["fixes"]


def test_startup_check_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(diagnostics.providers, "provider_health", boom)
    # Must degrade gracefully — startup diagnostics can never crash the app.
    d = diagnostics.log_startup_check()
    assert isinstance(d, dict)


def test_looks_like_ssl():
    assert diagnostics._looks_like_ssl("SSL certificate problem")
    assert diagnostics._looks_like_ssl("proxy handshake failed")
    assert not diagnostics._looks_like_ssl("connection timed out")
