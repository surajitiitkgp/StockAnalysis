"""Recommendation engine for intraday, short-term and long-term horizons.

Each horizon produces:
  - a numeric score in [-100, 100]
  - a verdict label (STRONG BUY .. STRONG SELL)
  - a list of human-readable signals with direction (bull/bear/neutral)
  - suggested trade levels (entry / target / stop-loss) derived from ATR

The logic is purely technical (plus a light fundamental overlay for the
long-term view). It is a decision-support tool, not financial advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from . import indicators
from .data_fetcher import StockData


@dataclass
class Signal:
    label: str
    direction: str   # "bull", "bear", "neutral"
    detail: str = ""


@dataclass
class Recommendation:
    horizon: str
    verdict: str
    score: float                       # -100 .. 100
    confidence: float                  # 0 .. 100
    entry: float | None = None
    target: float | None = None
    stop_loss: float | None = None
    risk_reward: float | None = None
    signals: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["signals"] = [asdict(s) if isinstance(s, Signal) else s for s in self.signals]
        return d


def _verdict(score: float) -> str:
    if score >= 50:
        return "STRONG BUY"
    if score >= 18:
        return "BUY"
    if score > -18:
        return "HOLD"
    if score > -50:
        return "SELL"
    return "STRONG SELL"


def _safe(value):
    if value is None:
        return None
    try:
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


class _Scorer:
    """Accumulates weighted contributions and the reasons behind them."""

    def __init__(self):
        self.score = 0.0
        self.max_abs = 0.0
        self.signals: list[Signal] = []

    def add(self, weight: float, value: float, label: str, detail: str = ""):
        """value in [-1, 1]; weight is the maximum magnitude of the contribution."""
        contrib = weight * max(-1.0, min(1.0, value))
        self.score += contrib
        self.max_abs += weight
        if value > 0.15:
            direction = "bull"
        elif value < -0.15:
            direction = "bear"
        else:
            direction = "neutral"
        self.signals.append(Signal(label=label, direction=direction, detail=detail))

    def finalize(self) -> tuple[float, float]:
        if self.max_abs == 0:
            return 0.0, 0.0
        normalized = 100.0 * self.score / self.max_abs
        confidence = min(100.0, 100.0 * abs(self.score) / self.max_abs + 20)
        return normalized, confidence


def _levels(entry: float, atr: float, direction: int, rr: float = 2.0,
            sl_mult: float = 1.5):
    """Compute entry/target/stop-loss. direction: +1 long, -1 short."""
    if entry is None or atr is None or atr <= 0 or direction == 0:
        return None, None, None, None
    risk = sl_mult * atr
    if direction > 0:
        stop = entry - risk
        target = entry + rr * risk
    else:
        stop = entry + risk
        target = entry - rr * risk
    return (round(entry, 2), round(target, 2), round(stop, 2), round(rr, 2))


# --------------------------------------------------------------------------- #
# Long-term (positional, months to years)
# --------------------------------------------------------------------------- #
def long_term(stock: StockData, df: pd.DataFrame | None = None) -> Recommendation:
    if df is None:
        df = indicators.add_daily_indicators(stock.history)
    s = _Scorer()
    if df is None or df.empty or len(df) < 60:
        return Recommendation("long_term", "HOLD", 0, 0,
                              signals=[Signal("Insufficient data", "neutral")])
    last = df.iloc[-1]
    close = last["Close"]

    # Price vs long-term moving averages.
    if not np.isnan(last.get("SMA200", np.nan)):
        diff = (close - last["SMA200"]) / last["SMA200"]
        s.add(20, np.tanh(diff * 8), "Price vs 200-day SMA",
              f"{'Above' if diff > 0 else 'Below'} 200-DMA by {diff*100:.1f}%")
    if not np.isnan(last.get("SMA50", np.nan)) and not np.isnan(last.get("SMA200", np.nan)):
        cross = 1.0 if last["SMA50"] > last["SMA200"] else -1.0
        s.add(18, cross, "Golden/Death cross (50 vs 200)",
              "50-DMA above 200-DMA (golden)" if cross > 0 else "50-DMA below 200-DMA (death)")

    # Long-term trend slope (200d regression direction via 100d change).
    if len(df) >= 100:
        change = (close - df["Close"].iloc[-100]) / df["Close"].iloc[-100]
        s.add(15, np.tanh(change * 4), "100-day trend",
              f"{change*100:+.1f}% over ~100 sessions")

    # Distance from 52-week high/low (momentum + room to run).
    hi = stock.info.get("fiftyTwoWeekHigh") or df["Close"].tail(252).max()
    lo = stock.info.get("fiftyTwoWeekLow") or df["Close"].tail(252).min()
    if hi and lo and hi > lo:
        pos = (close - lo) / (hi - lo)  # 0 near low, 1 near high
        # Sweet spot: strong but not euphoric (0.5-0.85 bullish, >0.97 caution).
        val = (pos - 0.4) * 2 if pos < 0.9 else (0.9 - pos) * 4 + 1.0
        s.add(10, max(-1, min(1, val)), "Position in 52-week range",
              f"{pos*100:.0f}% of 52-week range")

    # Monthly RSI proxy (avoid chasing overbought for long entries).
    rsi_v = last.get("RSI14")
    if rsi_v is not None and not np.isnan(rsi_v):
        if rsi_v < 35:
            s.add(8, 0.7, "RSI", f"Oversold ({rsi_v:.0f}) — value zone")
        elif rsi_v > 75:
            s.add(8, -0.6, "RSI", f"Overbought ({rsi_v:.0f}) — wait for pullback")
        else:
            s.add(8, (55 - abs(rsi_v - 55)) / 55 * 0.4, "RSI", f"Neutral ({rsi_v:.0f})")

    # Light fundamental overlay.
    info = stock.info
    pe = info.get("trailingPE")
    if pe and pe > 0:
        if pe < 25:
            s.add(8, 0.6, "Valuation (P/E)", f"P/E {pe:.1f} — reasonable")
        elif pe > 60:
            s.add(8, -0.6, "Valuation (P/E)", f"P/E {pe:.1f} — expensive")
        else:
            s.add(8, 0.1, "Valuation (P/E)", f"P/E {pe:.1f}")
    roe = info.get("returnOnEquity")
    if roe is not None:
        s.add(7, np.tanh((roe - 0.12) * 8), "Return on equity", f"ROE {roe*100:.1f}%")
    eg = info.get("earningsGrowth")
    if eg is not None:
        s.add(7, np.tanh(eg * 3), "Earnings growth", f"{eg*100:+.1f}% YoY")
    d2e = info.get("debtToEquity")
    if d2e is not None:
        s.add(5, -np.tanh((d2e - 80) / 80), "Debt/Equity", f"{d2e:.0f}")

    score, conf = s.finalize()
    atr_v = last.get("ATR14")
    direction = 1 if score > 0 else (-1 if score < -10 else 0)
    # Long-term levels use a wider stop (3x ATR) and 3:1 reward.
    entry, target, stop, rr = _levels(close, atr_v, direction, rr=3.0, sl_mult=3.0)
    return Recommendation(
        "long_term", _verdict(score), round(score, 1), round(conf, 0),
        entry=_safe(entry), target=_safe(target), stop_loss=_safe(stop),
        risk_reward=_safe(rr), signals=s.signals,
    )


# --------------------------------------------------------------------------- #
# Short-term (swing, days to a few weeks)
# --------------------------------------------------------------------------- #
def short_term(stock: StockData, df: pd.DataFrame | None = None) -> Recommendation:
    if df is None:
        df = indicators.add_daily_indicators(stock.history)
    s = _Scorer()
    if df is None or df.empty or len(df) < 30:
        return Recommendation("short_term", "HOLD", 0, 0,
                              signals=[Signal("Insufficient data", "neutral")])
    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["Close"]

    # EMA9 vs EMA21 crossover (swing trend).
    if not np.isnan(last["EMA9"]) and not np.isnan(last["EMA21"]):
        gap = (last["EMA9"] - last["EMA21"]) / last["EMA21"]
        s.add(18, np.tanh(gap * 30), "EMA 9/21 trend",
              "9-EMA above 21-EMA" if gap > 0 else "9-EMA below 21-EMA")

    # Price vs SMA20 / SMA50.
    if not np.isnan(last["SMA20"]):
        d20 = (close - last["SMA20"]) / last["SMA20"]
        s.add(12, np.tanh(d20 * 15), "Price vs 20-DMA", f"{d20*100:+.1f}% vs 20-DMA")
    if not np.isnan(last["SMA50"]):
        d50 = (close - last["SMA50"]) / last["SMA50"]
        s.add(10, np.tanh(d50 * 10), "Price vs 50-DMA", f"{d50*100:+.1f}% vs 50-DMA")

    # MACD histogram momentum + crossover.
    if not np.isnan(last["MACD_HIST"]):
        rising = last["MACD_HIST"] > prev["MACD_HIST"]
        sign = 1 if last["MACD_HIST"] > 0 else -1
        val = sign * (0.7 if rising == (sign > 0) else 0.3)
        cross = ""
        if prev["MACD"] <= prev["MACD_SIGNAL"] and last["MACD"] > last["MACD_SIGNAL"]:
            val = 1.0
            cross = " (bullish crossover)"
        elif prev["MACD"] >= prev["MACD_SIGNAL"] and last["MACD"] < last["MACD_SIGNAL"]:
            val = -1.0
            cross = " (bearish crossover)"
        s.add(18, val, "MACD", f"Histogram {last['MACD_HIST']:+.2f}{cross}")

    # RSI.
    rsi_v = last["RSI14"]
    if not np.isnan(rsi_v):
        if rsi_v < 30:
            s.add(12, 0.8, "RSI", f"Oversold ({rsi_v:.0f}) — bounce likely")
        elif rsi_v > 70:
            s.add(12, -0.8, "RSI", f"Overbought ({rsi_v:.0f}) — pullback risk")
        else:
            s.add(12, (rsi_v - 50) / 50 * 0.6, "RSI", f"{rsi_v:.0f}")

    # Bollinger position.
    if not np.isnan(last["BB_UPPER"]) and last["BB_UPPER"] > last["BB_LOWER"]:
        pos = (close - last["BB_LOWER"]) / (last["BB_UPPER"] - last["BB_LOWER"])
        if pos < 0.1:
            s.add(8, 0.6, "Bollinger Bands", "Near lower band — mean-reversion up")
        elif pos > 0.9:
            s.add(8, -0.5, "Bollinger Bands", "Near upper band — stretched")
        else:
            s.add(8, (pos - 0.5) * 0.6, "Bollinger Bands", f"{pos*100:.0f}% of band")

    # Trend strength (ADX) amplifies directional conviction.
    adx_v = last["ADX14"]
    if not np.isnan(adx_v):
        if adx_v > 25:
            s.add(8, 0.5 if s.score > 0 else -0.5, "ADX trend strength",
                  f"Strong trend (ADX {adx_v:.0f})")
        else:
            s.add(8, 0.0, "ADX trend strength", f"Weak/choppy (ADX {adx_v:.0f})")

    # Volume confirmation.
    if not np.isnan(last["VOL_SMA20"]) and last["VOL_SMA20"] > 0:
        vr = last["Volume"] / last["VOL_SMA20"]
        up_day = close >= prev["Close"]
        val = np.tanh((vr - 1) * 1.2) * (1 if up_day else -1)
        s.add(6, val, "Volume", f"{vr:.1f}x 20-day avg on {'up' if up_day else 'down'} day")

    score, conf = s.finalize()
    atr_v = last["ATR14"]
    direction = 1 if score > 0 else (-1 if score < -10 else 0)
    entry, target, stop, rr = _levels(close, atr_v, direction, rr=2.0, sl_mult=2.0)
    return Recommendation(
        "short_term", _verdict(score), round(score, 1), round(conf, 0),
        entry=_safe(entry), target=_safe(target), stop_loss=_safe(stop),
        risk_reward=_safe(rr), signals=s.signals,
    )


# --------------------------------------------------------------------------- #
# Intraday (same-session momentum)
# --------------------------------------------------------------------------- #
def intraday(stock: StockData, df: pd.DataFrame | None = None) -> Recommendation:
    s = _Scorer()
    intra = stock.intraday
    daily = df if df is not None else indicators.add_daily_indicators(stock.history)

    if daily is None or daily.empty:
        return Recommendation("intraday", "HOLD", 0, 0,
                              signals=[Signal("Insufficient data", "neutral")])

    last_daily = daily.iloc[-1]
    prev_close = daily["Close"].iloc[-2] if len(daily) > 1 else last_daily["Close"]

    has_intra = intra is not None and not intra.empty and len(intra) > 10
    if has_intra:
        intra = intra.copy()
        # Restrict to the most recent session for VWAP relevance.
        last_day = intra.index[-1].date()
        session = intra[intra.index.map(lambda x: x.date()) == last_day]
        if len(session) < 6:
            session = intra.tail(75)
        vw = indicators.vwap(session)
        price = session["Close"].iloc[-1]
        cur_vwap = vw.iloc[-1]

        # Price vs VWAP — the core intraday bias.
        if cur_vwap and not np.isnan(cur_vwap):
            d = (price - cur_vwap) / cur_vwap
            s.add(25, np.tanh(d * 60), "Price vs VWAP",
                  f"{'Above' if d > 0 else 'Below'} VWAP by {d*100:.2f}%")

        # Intraday RSI on 5-min bars.
        r = indicators.rsi(session["Close"], 14).iloc[-1]
        if not np.isnan(r):
            if r < 30:
                s.add(15, 0.7, "Intraday RSI", f"Oversold ({r:.0f})")
            elif r > 70:
                s.add(15, -0.7, "Intraday RSI", f"Overbought ({r:.0f})")
            else:
                s.add(15, (r - 50) / 50 * 0.7, "Intraday RSI", f"{r:.0f}")

        # Recent momentum (last ~30 min vs session open).
        mom = (price - session["Open"].iloc[0]) / session["Open"].iloc[0]
        s.add(15, np.tanh(mom * 40), "Session momentum", f"{mom*100:+.2f}% from open")

        # Intraday volume surge.
        vol_now = session["Volume"].tail(3).sum()
        vol_avg = session["Volume"].mean() * 3
        if vol_avg > 0:
            vr = vol_now / vol_avg
            s.add(10, np.tanh((vr - 1) * 1.5) * (1 if mom >= 0 else -1),
                  "Intraday volume", f"{vr:.1f}x recent average")
        cur_price = price
    else:
        cur_price = last_daily["Close"]
        s.signals.append(Signal("Intraday bars unavailable", "neutral",
                                "Using end-of-day momentum as a proxy"))

    # Gap vs previous close.
    gap = (cur_price - prev_close) / prev_close
    s.add(12, np.tanh(gap * 50), "Gap from prev close", f"{gap*100:+.2f}%")

    # Daily EMA9/EMA21 bias for trend alignment.
    if not np.isnan(last_daily["EMA9"]) and not np.isnan(last_daily["EMA21"]):
        bias = 1 if last_daily["EMA9"] > last_daily["EMA21"] else -1
        s.add(10, bias * 0.6, "Daily trend bias",
              "Up-trend (EMA9>EMA21)" if bias > 0 else "Down-trend (EMA9<EMA21)")

    # Volatility check — enough ATR to be tradable intraday.
    atr_v = last_daily["ATR14"]
    if atr_v and cur_price:
        atr_pct = atr_v / cur_price
        if atr_pct < 0.012:
            s.add(8, -0.4, "Volatility (ATR)", f"Low range ({atr_pct*100:.1f}%) — limited scope")
        else:
            s.add(8, 0.3, "Volatility (ATR)", f"Tradable range ({atr_pct*100:.1f}%)")

    score, conf = s.finalize()
    direction = 1 if score > 0 else (-1 if score < -10 else 0)
    # Intraday: tighter stop (1x ATR) and 1.5:1 target.
    entry, target, stop, rr = _levels(cur_price, atr_v, direction, rr=1.5, sl_mult=1.0)
    return Recommendation(
        "intraday", _verdict(score), round(score, 1), round(conf, 0),
        entry=_safe(entry), target=_safe(target), stop_loss=_safe(stop),
        risk_reward=_safe(rr), signals=s.signals,
    )


def analyze(stock: StockData) -> dict:
    """Run all three horizons and return a serialisable dict.

    Indicators are computed once and shared across horizons (they were
    previously recomputed for each horizon and again for the chart).
    """
    df = indicators.add_daily_indicators(stock.history)
    return {
        "intraday": intraday(stock, df).to_dict(),
        "short_term": short_term(stock, df).to_dict(),
        "long_term": long_term(stock, df).to_dict(),
    }
