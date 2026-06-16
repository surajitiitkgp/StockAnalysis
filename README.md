# Bharat Stocks — Indian Stock Market Analysis

A web application for analysing Indian equities (NSE / BSE) and generating
algorithmic trading signals across three horizons:

- **Intraday** — same-session momentum (VWAP, intraday RSI, gap, volume surge, ATR range)
- **Short-term** — swing trades over days to weeks (EMA 9/21, MACD, RSI, Bollinger, ADX, volume)
- **Long-term** — positional investing (200-DMA trend, golden/death cross, 52-week position, RSI, plus a light fundamental overlay: P/E, ROE, earnings growth, debt)

Each recommendation comes with a verdict (STRONG BUY → STRONG SELL), a score
(−100…+100), and ATR-based **entry / target / stop-loss** levels. A market-wide
**Screener** ranks the most liquid stocks for any horizon.

The app is gated behind a simple **login screen** and searches the **entire NSE
universe** (~2,300+ listed equities), not just a handful of names.

> ⚠️ **Disclaimer:** This is an educational decision-support tool. It produces
> algorithmic technical signals, **not investment advice**. Always do your own
> research and manage risk. Markets are risky.

## Stock universe (the full Indian list)

The complete list of NSE-listed equities is loaded from the official NSE equity
master, `EQUITY_L.csv` (`analysis/universe.py`):

1. On startup it refreshes a local snapshot from the live NSE archive
   (`https://archives.nseindia.com/content/equities/EQUITY_L.csv`), at most once
   every 7 days.
2. If the download is unavailable, it uses the **bundled snapshot** in
   `analysis/data/EQUITY_L.csv`, so the full list works offline too.
3. If both are missing, it falls back to a small Nifty 50 list.

This makes **all ~2,300+ NSE stocks** searchable. The Screener still scans only
the most liquid names first (capped via `?limit=`) because scanning every symbol
live would be slow and rate-limited.

## Data source

Historical OHLCV data comes from **Yahoo Finance** via `yfinance`, which serves
clean data for both NSE (`.NS`) and BSE (`.BO`) listings. NSE/BSE do not provide
a reliable public bulk *price* API (their sites block automated requests), so
Yahoo is the most dependable free source. Responses are cached for 15 minutes.

## Local 10-year history archive

A bulk downloader stores up to **10 years** of daily OHLCV for the whole NSE
universe in a local SQLite database (`analysis/data/history.db`):

```bash
# Download 10 years of daily history for every NSE stock (resumable)
python scripts/download_history.py

# Options
python scripts/download_history.py --period 5y          # shorter window
python scripts/download_history.py --symbols RELIANCE TCS
python scripts/download_history.py --force               # re-download all
```

The app is **local-first**: `data_fetcher` serves daily history from the store
(fast, works offline) and only tops up the most recent bars from Yahoo when the
local copy is a few days stale. The job is resumable — symbols already up to date
are skipped. Re-run it (e.g. daily) to keep the archive current.

## ML price forecast & verdict

`analysis/predictor.py` trains a **Random Forest regression** model per stock on
the local history and predicts the **7-trading-day-ahead return**, producing:

- a 7-day **price target** and expected % move,
- an **out-of-sample backtest** of the last 7 days (predicted vs. actual),
- a **verdict** — STRONG BUY / BUY / HOLD / SELL / STRONG SELL — derived from the
  predicted return, with a confidence blended from directional hit-rate and move size.

Features are all causal (returns over multiple lookbacks, price-vs-SMA ratios,
RSI, MACD, volatility, volume ratio, 52-week position), and training data ends
before the backtest window so the reported accuracy is honest. The forecast is
shown as its own card on the Analyze view, alongside the rule-based signals.

> Note: 7-day price prediction is genuinely hard; treat the forecast as a
> statistical estimate, **not** a guarantee. It is decision-support, not advice.

## Login

The dashboard requires sign-in. A default account is seeded on first run:

| Username | Password |
|---|---|
| `admin` | `admin123` |

You can register additional users from the login screen, or override the default
with environment variables before first run:

```bash
export APP_USERNAME=myuser
export APP_PASSWORD='a-strong-password'
export SECRET_KEY='any-long-random-string'   # optional: stable session secret
```

Credentials are stored hashed (PBKDF2-HMAC-SHA256) in `analysis/data/users.json`,
which is git-ignored.

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
├── app.py                  # Flask app + REST API + login/session gate
├── analysis/
│   ├── universe.py         # Full NSE universe loader (live + bundled CSV fallback)
│   ├── auth.py             # File-backed users + PBKDF2 password hashing
│   ├── store.py            # SQLite local history store (read/write)
│   ├── data_fetcher.py     # local-first history + yfinance top-up + TTL cache
│   ├── indicators.py       # RSI, MACD, SMA/EMA, Bollinger, VWAP, ATR, ADX, Stoch, OBV
│   ├── strategy.py         # rule-based scoring engine for the three horizons
│   ├── predictor.py        # Random Forest 7-day price forecast + backtest + verdict
│   └── data/EQUITY_L.csv   # bundled NSE equity master (offline fallback)
├── scripts/download_history.py  # bulk 10-year history downloader -> SQLite
├── templates/login.html    # login / register screen
├── templates/index.html    # dashboard UI
├── static/css/style.css
├── static/js/app.js        # frontend logic + Plotly charts
├── requirements.txt
└── README.md
```

## API

All API and page routes require an authenticated session (APIs return `401`
when logged out).

| Endpoint | Description |
|---|---|
| `GET /login`, `GET /logout` | Login / register screen, sign out |
| `GET /api/stocks` | Full screening universe (all NSE stocks) |
| `GET /api/search?q=` | Search the full universe by name/symbol |
| `GET /api/analyze?symbol=RELIANCE&exchange=NSE` | Full analysis (chart + signals + recommendations + ML forecast) |
| `GET /api/predict?symbol=RELIANCE&exchange=NSE` | ML 7-day price forecast + backtest + verdict |
| `GET /api/screener?horizon=short_term&exchange=NSE&limit=150` | Rank the most liquid stocks (`intraday` / `short_term` / `long_term`) |

## Refreshing the stock list

The universe auto-refreshes from NSE every 7 days. To force a refresh, delete
`analysis/data/EQUITY_L.csv` (it will re-download on next start) or call
`universe.reload_universe()`.
