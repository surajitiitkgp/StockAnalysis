# Bharat Stocks — Indian Stock Market Analysis

A web application for analysing Indian equities (NSE / BSE) and generating
algorithmic trading signals across three horizons:

- **Intraday** — same-session momentum (VWAP, intraday RSI, gap, volume surge, ATR range)
- **Short-term** — swing trades over days to weeks (EMA 9/21, MACD, RSI, Bollinger, ADX, volume)
- **Long-term** — positional investing (200-DMA trend, golden/death cross, 52-week position, RSI, plus a light fundamental overlay: P/E, ROE, earnings growth, debt)

Each recommendation comes with a verdict (STRONG BUY → STRONG SELL), a score
(−100…+100), and ATR-based **entry / target / stop-loss** levels. A market-wide
**Screener** ranks the Nifty universe for any horizon.

> ⚠️ **Disclaimer:** This is an educational decision-support tool. It produces
> algorithmic technical signals, **not investment advice**. Always do your own
> research and manage risk. Markets are risky.

## Data source

Historical OHLCV data comes from **Yahoo Finance** via `yfinance`, which serves
clean data for both NSE (`.NS`) and BSE (`.BO`) listings. NSE/BSE do not provide
a reliable public bulk API (their sites block automated requests), so Yahoo is
the most dependable free source. Responses are cached for 15 minutes.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
python app.py

# 3. Open the dashboard
#    http://127.0.0.1:5000
```

## Project structure

```
stock_market_analysis/
├── app.py                  # Flask app + REST API
├── analysis/
│   ├── universe.py         # NSE/BSE symbol universe (Nifty 50 + extras)
│   ├── data_fetcher.py     # yfinance fetching + in-memory TTL cache
│   ├── indicators.py       # RSI, MACD, SMA/EMA, Bollinger, VWAP, ATR, ADX, Stoch, OBV
│   └── strategy.py         # scoring engine for the three horizons
├── templates/index.html    # dashboard UI
├── static/css/style.css
├── static/js/app.js        # frontend logic + Plotly charts
├── requirements.txt
└── README.md
```

## API

| Endpoint | Description |
|---|---|
| `GET /api/stocks` | Full screening universe |
| `GET /api/search?q=` | Search universe by name/symbol |
| `GET /api/analyze?symbol=RELIANCE&exchange=NSE` | Full analysis (chart + signals + recommendations) |
| `GET /api/screener?horizon=short_term&exchange=NSE` | Rank universe (`intraday` / `short_term` / `long_term`) |

## Extending the universe

Add names to `STOCKS` in `analysis/universe.py` using the base NSE symbol
(e.g. `"Company Name": "SYMBOL"`).
