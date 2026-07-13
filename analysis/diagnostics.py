"""Provider connectivity self-check + actionable guidance.

Corporate/Windows networks that SSL-inspect HTTPS frequently break every price
provider at once (``yfinance``/``curl_cffi`` don't trust the proxy's root CA).
Because a single failure doesn't trip a provider's circuit breaker, the app can
look "healthy" while fetching nothing. This module runs a real reachability
probe and turns the raw provider details into a plain-language diagnosis the
UI, the readiness probe, and the startup log can all share.
"""

from __future__ import annotations

from . import providers, store
from .logging_config import get_logger

log = get_logger(__name__)

# Substrings that identify a TLS/certificate/proxy failure in a provider detail.
_SSL_MARKERS = ("ssl", "certificate", "cert ", "self signed", "self-signed",
                "ca ", "proxy", "tlsv", "handshake")


def _looks_like_ssl(detail: str) -> bool:
    d = (detail or "").lower()
    return any(m in d for m in _SSL_MARKERS)


def diagnose(probe_ticker: str = "RELIANCE.NS") -> dict:
    """Probe the daily-provider chain and summarise connectivity health.

    Returns a dict with:
      - ``ok``            : at least one provider returned data
      - ``reachable``     : same as ok (kept for readability)
      - ``ssl_issue``     : every failure looks like a TLS/cert/proxy problem
      - ``providers``     : per-provider {name: status}
      - ``details``       : per-provider failure detail (truncated)
      - ``has_local_data``: the local archive has *some* rows to fall back on
      - ``severity``      : "ok" | "degraded" | "offline"
      - ``title`` / ``hint`` / ``fixes``: user-facing guidance
    """
    try:
        health = providers.provider_health(probe_ticker)
    except Exception as exc:  # noqa: BLE001
        health = []
        log.debug("provider_health probe failed: %s", exc, exc_info=True)

    statuses = {h["name"]: h.get("status", "unknown") for h in health}
    details = {h["name"]: (h.get("detail") or "")[:300] for h in health if h.get("detail")}
    ok = any(h.get("status") == "ok" for h in health)
    failing = [h for h in health if h.get("status") not in ("ok",)]
    ssl_issue = (not ok) and any(_looks_like_ssl(h.get("detail", "")) for h in failing)

    try:
        has_local = store.stats().get("rows", 0) > 0
    except Exception:  # noqa: BLE001
        has_local = False

    if ok:
        severity = "ok"
        title = "Market data providers reachable"
        hint = ""
        fixes: list[str] = []
    elif ssl_issue:
        severity = "offline"
        title = "Data providers blocked by an SSL/certificate error"
        hint = ("Your network is intercepting HTTPS (common on corporate "
                "machines) and the data library doesn't trust the proxy's root "
                "certificate, so every price fetch fails.")
        fixes = [
            "pip install pip-system-certs   (uses the OS/corporate trust store)",
            "or set CURL_CA_BUNDLE and REQUESTS_CA_BUNDLE to your corporate root CA .pem",
            "or download data once from an un-inspected network (e.g. a phone hotspot), "
            "then the local archive serves it offline",
        ]
    else:
        severity = "offline" if not has_local else "degraded"
        title = "Market data providers are unreachable"
        hint = ("Every data source returned nothing. This is usually a network "
                "or provider outage rather than a problem with a specific stock.")
        fixes = [
            "check your internet/proxy connection",
            "retry in a moment (providers may be rate-limiting)",
            "populate the local archive with: python scripts/sync_data.py",
        ]

    return {
        "ok": ok, "reachable": ok, "ssl_issue": ssl_issue,
        "severity": severity, "providers": statuses, "details": details,
        "has_local_data": has_local, "title": title, "hint": hint, "fixes": fixes,
    }


def log_startup_check(probe_ticker: str = "RELIANCE.NS") -> dict:
    """Run the probe at boot and log a clear, actionable message.

    Never raises — diagnostics must not block startup. Returns the diagnosis so
    the caller may cache it for the readiness/status endpoints.
    """
    try:
        d = diagnose(probe_ticker)
    except Exception:  # noqa: BLE001
        log.debug("startup provider check failed", exc_info=True)
        return {"ok": None, "severity": "unknown"}

    if d["ok"]:
        log.info("Provider self-check: OK (%s)", d["providers"])
        return d

    banner = "=" * 68
    lines = [banner, f"DATA PROVIDER SELF-CHECK: {d['title']}", banner]
    if d["hint"]:
        lines.append(d["hint"])
    if d["fixes"]:
        lines.append("How to fix:")
        lines.extend(f"  - {f}" for f in d["fixes"])
    if d["has_local_data"]:
        lines.append("Note: the local archive has data, so cached stocks still work offline.")
    else:
        lines.append("Note: the local archive is empty, so analysis will fail until this is fixed.")
    lines.append(banner)
    log.warning("\n".join(lines))
    return d
