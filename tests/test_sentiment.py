"""Tests for the financial sentiment scorer."""

from __future__ import annotations

from analysis import sentiment


def test_positive_headline():
    r = sentiment.score_text("Profit surges, company beats estimates and is upgraded")
    assert r.score > 0.3
    assert r.label == "positive"


def test_negative_headline():
    r = sentiment.score_text("Shares plunge on fraud probe, downgraded amid recession fears")
    assert r.score < -0.3
    assert r.label == "negative"


def test_negation_flips_polarity():
    pos = sentiment.score_text("strong growth").score
    neg = sentiment.score_text("not strong growth").score
    assert pos > 0
    assert neg < pos


def test_neutral_text():
    r = sentiment.score_text("The company held its annual general meeting on Tuesday")
    assert r.label == "neutral"


def test_empty_text():
    assert sentiment.score_text("").score == 0.0


def test_score_bounded():
    r = sentiment.score_text("surge surge surge boom record profit rally soar gain " * 5)
    assert -1.0 <= r.score <= 1.0


def test_aggregate_headlines():
    items = [
        {"title": "stock soars on record profit"},
        {"title": "shares crash on lawsuit and fraud"},
        {"title": "company holds meeting"},
    ]
    agg = sentiment.score_headlines(items)
    assert agg["count"] == 3
    assert agg["positive"] >= 1
    assert agg["negative"] >= 1
    assert -1.0 <= agg["score"] <= 1.0


def test_aggregate_blends_provider_sentiment():
    items = [{"title": "neutral wording here", "sentiment": 0.8}]
    agg = sentiment.score_headlines(items)
    assert agg["score"] > 0  # provider positivity pulls it up
