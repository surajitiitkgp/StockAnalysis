"""Lightweight, dependency-free financial-news sentiment scorer.

A finance-tuned lexicon scorer (in the spirit of VADER, but domain-specific and
with no external dependencies). It handles negations ("not profitable"),
intensifiers ("sharply higher"), and returns a normalised polarity in
``[-1, 1]``.

If a news provider supplies its own sentiment (e.g. NewsAPI.ai / Finnhub), the
caller can use that directly; this scorer is the always-available fallback and
is used to aggregate headline streams into a daily signal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Finance-oriented polarity lexicon (word -> weight in [-3, 3]).
_POSITIVE = {
    "surge": 3, "surges": 3, "surged": 3, "soar": 3, "soars": 3, "soared": 3,
    "rally": 2, "rallies": 2, "rallied": 2, "jump": 2, "jumps": 2, "jumped": 2,
    "gain": 2, "gains": 2, "gained": 2, "rise": 1, "rises": 1, "rose": 1,
    "beat": 2, "beats": 2, "outperform": 2, "outperforms": 2, "upgrade": 3,
    "upgraded": 3, "bullish": 3, "record": 2, "profit": 2, "profits": 2,
    "growth": 2, "strong": 2, "robust": 2, "boost": 2, "boosts": 2, "boosted": 2,
    "expand": 1, "expansion": 1, "approval": 2, "approved": 2, "wins": 2, "win": 2,
    "dividend": 1, "buyback": 2, "optimistic": 2, "recovery": 2, "rebound": 2,
    "high": 1, "higher": 1, "positive": 2, "success": 2, "successful": 2,
    "breakthrough": 3, "milestone": 2, "outlook": 1, "raise": 1, "raised": 1,
    "topped": 2, "exceeds": 2, "exceeded": 2, "accelerate": 2, "momentum": 1,
}
_NEGATIVE = {
    "plunge": -3, "plunges": -3, "plunged": -3, "crash": -3, "crashes": -3,
    "crashed": -3, "slump": -3, "slumps": -3, "slumped": -3, "tumble": -3,
    "tumbles": -3, "tumbled": -3, "fall": -2, "falls": -2, "fell": -2,
    "drop": -2, "drops": -2, "dropped": -2, "decline": -2, "declines": -2,
    "declined": -2, "loss": -2, "losses": -2, "miss": -2, "misses": -2,
    "missed": -2, "downgrade": -3, "downgraded": -3, "bearish": -3, "weak": -2,
    "weakness": -2, "cut": -2, "cuts": -2, "slashed": -3, "warning": -2,
    "warns": -2, "warned": -2, "fraud": -3, "probe": -2, "lawsuit": -2, "fine": -2,
    "fined": -2, "ban": -3, "banned": -3, "sanction": -3, "sanctions": -3,
    "recall": -2, "recalls": -2, "default": -3, "bankruptcy": -3, "bankrupt": -3,
    "layoff": -2, "layoffs": -2, "scandal": -3, "concern": -1, "concerns": -1,
    "risk": -1, "risks": -1, "low": -1, "lower": -1, "negative": -2, "slowdown": -2,
    "recession": -3, "crisis": -3, "war": -3, "conflict": -2, "tensions": -2,
    "tariff": -2, "tariffs": -2, "inflation": -1, "selloff": -3, "downturn": -2,
    "halts": -2, "halt": -2, "delay": -1, "delays": -1, "investigation": -2,
}
_NEGATIONS = {"not", "no", "never", "without", "fails", "fail", "failed", "un"}
_INTENSIFIERS = {"very": 1.5, "sharply": 1.6, "significantly": 1.5, "hugely": 1.7,
                 "slightly": 0.5, "marginally": 0.5, "modestly": 0.6, "steeply": 1.6}

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class SentimentResult:
    score: float          # normalised polarity in [-1, 1]
    label: str            # positive / negative / neutral
    pos_hits: int = 0
    neg_hits: int = 0


def _label(score: float) -> str:
    if score > 0.15:
        return "positive"
    if score < -0.15:
        return "negative"
    return "neutral"


def score_text(text: str) -> SentimentResult:
    """Score a single piece of text (headline/description)."""
    if not text:
        return SentimentResult(0.0, "neutral")
    tokens = _TOKEN_RE.findall(text.lower())
    total = 0.0
    pos_hits = neg_hits = 0
    for i, tok in enumerate(tokens):
        weight = _POSITIVE.get(tok, 0) + _NEGATIVE.get(tok, 0)
        if weight == 0:
            continue
        # Intensifier immediately before the word.
        if i > 0 and tokens[i - 1] in _INTENSIFIERS:
            weight *= _INTENSIFIERS[tokens[i - 1]]
        # Negation within the previous two tokens flips polarity.
        window = tokens[max(0, i - 2):i]
        if any(w in _NEGATIONS for w in window):
            weight = -weight
        total += weight
        if weight > 0:
            pos_hits += 1
        elif weight < 0:
            neg_hits += 1
    # Squash the raw sum into [-1, 1] (tanh-like normalisation).
    score = math.tanh(total / 4.0)
    return SentimentResult(round(score, 4), _label(score), pos_hits, neg_hits)


def score_headlines(items: list) -> dict:
    """Aggregate a list of articles into a single sentiment summary.

    ``items`` is a list of dicts with ``title`` (and optional ``description``
    and provider ``sentiment``). Provider-supplied sentiment, when present, is
    blended with the lexicon score.
    """
    if not items:
        return {"score": 0.0, "label": "neutral", "count": 0,
                "positive": 0, "negative": 0, "neutral": 0}

    scores = []
    pos = neg = neu = 0
    for it in items:
        text = " ".join(filter(None, [it.get("title"), it.get("description")]))
        res = score_text(text)
        s = res.score
        provided = it.get("sentiment")
        if isinstance(provided, (int, float)):
            # Blend external sentiment (already ~[-1,1]) with lexicon score.
            s = 0.5 * s + 0.5 * max(-1.0, min(1.0, float(provided)))
        scores.append(s)
        if s > 0.15:
            pos += 1
        elif s < -0.15:
            neg += 1
        else:
            neu += 1

    avg = sum(scores) / len(scores)
    return {
        "score": round(avg, 4),
        "label": _label(avg),
        "count": len(scores),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
    }
