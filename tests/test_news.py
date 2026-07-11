"""Tests for the news provider layer (mocked HTTP, no network)."""

from __future__ import annotations

from analysis import cache, news


def test_finnhub_parsing(monkeypatch):
    payload = [
        {"headline": "Profit surges", "summary": "beats estimates",
         "url": "http://x", "source": "Reuters", "datetime": 1700000000},
    ]
    monkeypatch.setattr(news, "_http_json", lambda url, timeout=None: payload)
    p = news.FinnhubProvider("KEY")
    out = p.company_news("RELIANCE.NS", "Reliance", "2024-01-01", "2024-02-01", 10)
    assert len(out) == 1
    assert out[0]["title"] == "Profit surges"
    assert out[0]["published_at"] is not None


def test_gnews_parsing(monkeypatch):
    payload = {"articles": [
        {"title": "Stock rallies", "description": "up sharply", "url": "http://y",
         "source": {"name": "Forbes"}, "publishedAt": "2026-07-10T12:00:00Z"},
    ]}
    monkeypatch.setattr(news, "_http_json", lambda url, timeout=None: payload)
    p = news.GNewsProvider("KEY")
    out = p.company_news("TCS", "Tata Consultancy", "2026-06-10", "2026-07-10", 10)
    assert out[0]["source"] == "Forbes"
    assert out[0]["published_at"].year == 2026


def test_newsapi_ai_sentiment_passthrough(monkeypatch):
    payload = {"articles": {"results": [
        {"title": "Markets steady", "body": "calm session", "url": "http://z",
         "source": {"title": "EventReg"}, "dateTime": "2026-07-09T09:00:00Z",
         "sentiment": 0.4},
    ]}}
    monkeypatch.setattr(news, "_http_json", lambda url, timeout=None: payload)
    p = news.NewsApiAiProvider("KEY")
    out = p.company_news("INFY", "Infosys", "2026-06-09", "2026-07-09", 10)
    assert out[0]["sentiment"] == 0.4


def test_daily_series_aggregation():
    from datetime import datetime
    articles = [
        {"title": "surges on record profit", "description": "",
         "published_at": datetime(2026, 7, 1), "sentiment": None},
        {"title": "another strong gain", "description": "",
         "published_at": datetime(2026, 7, 1), "sentiment": None},
        {"title": "plunges on fraud", "description": "",
         "published_at": datetime(2026, 7, 2), "sentiment": None},
    ]
    series = news._daily_series(articles)
    assert len(series) == 2
    assert series[0]["date"] == "2026-07-01"
    assert series[0]["count"] == 2
    assert series[0]["sentiment"] > 0
    assert series[1]["sentiment"] < 0


def test_dedupe_by_title():
    arts = [{"title": "Same"}, {"title": "same"}, {"title": "Different"}]
    assert len(news._dedupe(arts)) == 2


def test_disabled_without_keys(monkeypatch):
    monkeypatch.setattr(news, "_active_providers", lambda: [])
    assert not news.is_enabled()
    summary = news.get_sentiment_summary("RELIANCE")
    assert not summary["available"]


def test_summary_with_mocked_provider(monkeypatch):
    cache.clear()

    class FakeProvider(news.NewsProvider):
        name = "fake"

        def company_news(self, symbol, company, frm, to, limit):
            from datetime import datetime
            return [
                {"title": "Profit surges to record", "description": "beats",
                 "url": "http://a", "source": "X", "published_at": datetime(2026, 7, 1),
                 "sentiment": None},
                {"title": "Shares plunge on probe", "description": "fraud",
                 "url": "http://b", "source": "Y", "published_at": datetime(2026, 7, 2),
                 "sentiment": None},
            ]

    monkeypatch.setattr(news, "_active_providers", lambda: [FakeProvider("KEY")])
    monkeypatch.setattr(news, "is_enabled", lambda: True)
    summary = news.get_sentiment_summary("RELIANCE", "NSE", "Reliance")
    assert summary["available"]
    assert summary["provider"] == "fake"
    assert summary["aggregate"]["count"] == 2
    assert len(summary["daily"]) == 2
    assert len(summary["headlines"]) == 2
