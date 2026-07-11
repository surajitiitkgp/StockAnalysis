"""News aggregation layer: company + market/geopolitical headlines.

Pluggable providers (all optional, key-gated, dependency-free REST):
  - **Finnhub**        company-news + general news
  - **NewsAPI.ai**     (EventRegistry) articles w/ provider sentiment
  - **NewsData.io**    latest news search
  - **GNews**          search + business top-headlines

Providers are tried in the configured order; only those with an API key are
used. Everything degrades gracefully (returns "unavailable") when no keys are
set, so the app works exactly as before without news.

Each provider normalises to a common article schema::

    {"title", "description", "url", "source", "published_at" (datetime|None),
     "sentiment" (float in [-1,1] | None)}

``get_sentiment_summary`` turns a headline stream into an aggregate score plus
a **daily sentiment series** used both for the UI and for the ML feature
pipeline (see :mod:`analysis.features`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from . import cache, sentiment
from .config import settings
from .logging_config import get_logger
from .providers import CircuitBreaker, retry_with_backoff

log = get_logger(__name__)

_USER_AGENT = "Mozilla/5.0 (BharatStocks NewsClient)"


def _http_json(url: str, timeout: int | None = None) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout or settings.fetch_timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        # Surface the provider's error message (e.g. "activate your account",
        # "daily limit reached") instead of a bare status code.
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            body = ""
        raise urllib.error.HTTPError(
            exc.url, exc.code, f"{exc.reason} — {body}", exc.headers, None) from None


def _parse_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):  # unix seconds (Finnhub)
        try:
            return datetime.utcfromtimestamp(int(value))
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        v = value.strip().replace("Z", "").replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S+00:00",
                    "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(v[:len(fmt) + 2].strip(), fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class NewsProvider:
    name = "base"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.breaker = CircuitBreaker(
            f"news:{self.name}", settings.breaker_fail_threshold, settings.breaker_reset_seconds)

    def company_news(self, symbol: str, company: str, frm: str, to: str,
                     limit: int) -> list:  # pragma: no cover
        raise NotImplementedError

    def market_news(self, limit: int) -> list:  # pragma: no cover
        raise NotImplementedError


class FinnhubProvider(NewsProvider):
    name = "finnhub"

    def company_news(self, symbol, company, frm, to, limit):
        # Finnhub's company-news is keyed by ticker symbol, not company name.
        url = ("https://finnhub.io/api/v1/company-news?symbol="
               f"{urllib.parse.quote(symbol)}&from={frm}&to={to}&token={self.api_key}")
        data = _http_json(url)
        out = []
        for a in (data or [])[:limit]:
            out.append({
                "title": a.get("headline"), "description": a.get("summary"),
                "url": a.get("url"), "source": a.get("source"),
                "published_at": _parse_dt(a.get("datetime")), "sentiment": None,
            })
        return out

    def market_news(self, limit):
        url = f"https://finnhub.io/api/v1/news?category=general&token={self.api_key}"
        data = _http_json(url)
        out = []
        for a in (data or [])[:limit]:
            out.append({
                "title": a.get("headline"), "description": a.get("summary"),
                "url": a.get("url"), "source": a.get("source"),
                "published_at": _parse_dt(a.get("datetime")), "sentiment": None,
            })
        return out


class GNewsProvider(NewsProvider):
    name = "gnews"

    def _map(self, data, limit):
        out = []
        for a in (data.get("articles") or [])[:limit]:
            src = a.get("source") or {}
            out.append({
                "title": a.get("title"), "description": a.get("description"),
                "url": a.get("url"), "source": src.get("name"),
                "published_at": _parse_dt(a.get("publishedAt")), "sentiment": None,
            })
        return out

    def company_news(self, symbol, company, frm, to, limit):
        # Keyword search works best with the company name.
        q = urllib.parse.quote(f'"{company or symbol}"')
        url = (f"https://gnews.io/api/v4/search?q={q}&lang=en&max={min(limit, 25)}"
               f"&from={frm}T00:00:00Z&apikey={self.api_key}")
        return self._map(_http_json(url), limit)

    def market_news(self, limit):
        url = (f"https://gnews.io/api/v4/top-headlines?category=business&lang=en"
               f"&max={min(limit, 25)}&apikey={self.api_key}")
        return self._map(_http_json(url), limit)


class NewsDataProvider(NewsProvider):
    name = "newsdata"

    def _map(self, data, limit):
        out = []
        for a in (data.get("results") or [])[:limit]:
            sent = None
            s = (a.get("sentiment") or "").lower()
            if s == "positive":
                sent = 0.6
            elif s == "negative":
                sent = -0.6
            elif s == "neutral":
                sent = 0.0
            out.append({
                "title": a.get("title"), "description": a.get("description"),
                "url": a.get("link"), "source": a.get("source_id"),
                "published_at": _parse_dt(a.get("pubDate")), "sentiment": sent,
            })
        return out

    def company_news(self, symbol, company, frm, to, limit):
        q = urllib.parse.quote(company or symbol)
        url = (f"https://newsdata.io/api/1/news?apikey={self.api_key}&q={q}"
               f"&language=en")
        return self._map(_http_json(url), limit)

    def market_news(self, limit):
        url = (f"https://newsdata.io/api/1/news?apikey={self.api_key}"
               f"&category=business,politics,world&language=en")
        return self._map(_http_json(url), limit)


class NewsApiAiProvider(NewsProvider):
    """NewsAPI.ai / EventRegistry — includes provider sentiment in [-1, 1]."""

    name = "newsapi_ai"

    def _map(self, data, limit):
        results = ((data.get("articles") or {}).get("results")) or []
        out = []
        for a in results[:limit]:
            src = a.get("source") or {}
            out.append({
                "title": a.get("title"), "description": a.get("body", "")[:280],
                "url": a.get("url"), "source": src.get("title"),
                "published_at": _parse_dt(a.get("dateTime") or a.get("date")),
                "sentiment": a.get("sentiment"),
            })
        return out

    def _query(self, params, limit):
        base = "https://eventregistry.org/api/v1/article/getArticles?"
        params.update({
            "resultType": "articles", "articlesSortBy": "date", "lang": "eng",
            "includeArticleSentiment": "true", "articlesCount": str(min(limit, 100)),
            "apiKey": self.api_key,
        })
        return self._map(_http_json(base + urllib.parse.urlencode(params)), limit)

    def company_news(self, symbol, company, frm, to, limit):
        return self._query({"keyword": company or symbol, "dateStart": frm, "dateEnd": to}, limit)

    def market_news(self, limit):
        return self._query({"keyword": "geopolitics OR economy OR markets"}, limit)


_PROVIDER_CLASSES = {
    "finnhub": FinnhubProvider,
    "gnews": GNewsProvider,
    "newsdata": NewsDataProvider,
    "newsapi_ai": NewsApiAiProvider,
}


_PROVIDER_CACHE: dict[str, NewsProvider] = {}


def _active_providers() -> list:
    """Return keyed provider singletons (so circuit-breaker state persists)."""
    keys = settings.news_keys()
    providers = []
    for name in settings.news_providers:
        cls = _PROVIDER_CLASSES.get(name)
        key = keys.get(name)
        if not (cls and key):
            continue
        cached = _PROVIDER_CACHE.get(name)
        if cached is None or cached.api_key != key:
            cached = cls(key)
            _PROVIDER_CACHE[name] = cached
        providers.append(cached)
    return providers


def reset_providers() -> None:
    """Drop cached provider singletons (after a key / order change)."""
    _PROVIDER_CACHE.clear()


def is_enabled() -> bool:
    return settings.news_enabled and bool(_active_providers())


def _dedupe(articles: list) -> list:
    seen = set()
    out = []
    for a in articles:
        title = (a.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        out.append(a)
    return out


def _windows() -> list[tuple[str, str]]:
    """Date windows to try for company news, widening on empty results.

    The wider fallback makes company lookups resilient to sparse provider
    coverage *and* to a system clock that runs ahead of the provider's latest
    data (a trailing 30-day window would otherwise land in the future and
    return nothing).
    """
    now = datetime.utcnow()
    spans = sorted({settings.news_lookback_days, 365})
    return [((now - timedelta(days=d)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"))
            for d in spans]


def _finnhub_symbol(base: str, exchange: str) -> str:
    """Map a base ticker to Finnhub's symbol convention per exchange."""
    ex = (exchange or "").upper()
    if ex == "NSE":
        return f"{base}.NS"
    if ex == "BSE":
        return f"{base}.BO"
    return base


def _fetch(kind: str, symbol: str | None, company: str | None,
           limit: int) -> tuple[list, str | None]:
    """Fetch company/market news from the first working provider."""
    windows = _windows() if kind == "company" else [(None, None)]
    # Finnhub's free tier only covers US/global company news; skip it for
    # Indian tickers (it returns 403 "no access") so we go straight to a
    # keyword provider instead of burning a request + logging noise.
    skip_finnhub_company = (
        kind == "company" and (symbol or "").upper().endswith((".NS", ".BO")))
    for provider in _active_providers():
        if provider.breaker.is_open:
            continue
        if skip_finnhub_company and provider.name == "finnhub":
            continue
        for frm, to in windows:
            try:
                def _call(p=provider, _frm=frm, _to=to):
                    if kind == "company":
                        return p.company_news(symbol, company, _frm, _to, limit)
                    return p.market_news(limit)
                articles = retry_with_backoff(_call, retries=2, label=f"{provider.name}.{kind}")
                provider.breaker.record_success()
                articles = _dedupe([a for a in articles if a.get("title")])
                if articles:
                    return articles, provider.name
            except Exception as exc:  # noqa: BLE001
                provider.breaker.record_failure()
                log.info("news provider %s failed: %s", provider.name, exc)
                break  # provider errored; move to the next provider
    return [], None


def _daily_series(articles: list) -> list:
    """Aggregate articles into a per-day sentiment series (causal-friendly)."""
    by_day: dict[str, list] = {}
    for a in articles:
        dt = a.get("published_at")
        if dt is None:
            continue
        day = dt.strftime("%Y-%m-%d")
        text = " ".join(filter(None, [a.get("title"), a.get("description")]))
        res = sentiment.score_text(text)
        s = res.score
        if isinstance(a.get("sentiment"), (int, float)):
            s = 0.5 * s + 0.5 * max(-1.0, min(1.0, float(a["sentiment"])))
        by_day.setdefault(day, []).append(s)
    series = [{"date": d, "sentiment": round(sum(v) / len(v), 4), "count": len(v)}
              for d, v in sorted(by_day.items())]
    return series


def get_sentiment_summary(symbol: str, exchange: str = "NSE",
                          company_name: str | None = None) -> dict:
    """Company news + aggregate sentiment + daily series (cached)."""
    if not is_enabled():
        return {"available": False, "reason": "no news provider configured"}

    base = symbol.upper().replace(".NS", "").replace(".BO", "")
    ckey = f"news_sent:{base}:{exchange.upper()}"
    cached = cache.get(ckey)
    if cached is not None:
        return cached

    fin_symbol = _finnhub_symbol(base, exchange)
    company = company_name or base
    articles, provider = _fetch("company", fin_symbol, company, settings.news_max_articles)
    if not articles:
        result = {"available": False, "reason": "no articles found", "provider": provider}
        cache.set(ckey, result, settings.news_cache_ttl // 2)
        return result

    agg = sentiment.score_headlines(articles)
    series = _daily_series(articles)
    headlines = [{
        "title": a["title"], "source": a.get("source"), "url": a.get("url"),
        "published_at": a["published_at"].strftime("%Y-%m-%d") if a.get("published_at") else None,
        "sentiment": sentiment.score_text(
            " ".join(filter(None, [a.get("title"), a.get("description")]))).score,
    } for a in articles[:12]]

    result = {
        "available": True, "provider": provider, "symbol": base,
        "aggregate": agg, "daily": series, "headlines": headlines,
    }
    cache.set(ckey, result, settings.news_cache_ttl)
    return result


def get_market_news() -> dict:
    """Market-wide / geopolitical news + aggregate sentiment (cached)."""
    if not is_enabled():
        return {"available": False, "reason": "no news provider configured"}
    ckey = "news_market"
    cached = cache.get(ckey)
    if cached is not None:
        return cached

    articles, provider = _fetch("market", None, None, settings.news_max_articles)
    if not articles:
        result = {"available": False, "reason": "no articles found", "provider": provider}
        cache.set(ckey, result, settings.news_cache_ttl // 2)
        return result

    agg = sentiment.score_headlines(articles)
    series = _daily_series(articles)
    headlines = [{
        "title": a["title"], "source": a.get("source"), "url": a.get("url"),
        "published_at": a["published_at"].strftime("%Y-%m-%d") if a.get("published_at") else None,
    } for a in articles[:12]]
    result = {"available": True, "provider": provider, "aggregate": agg,
              "daily": series, "headlines": headlines}
    cache.set(ckey, result, settings.news_cache_ttl)
    return result


def status() -> dict:
    return {
        "enabled": is_enabled(),
        "providers": [p.name for p in _active_providers()],
    }
