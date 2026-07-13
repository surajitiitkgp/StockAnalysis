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

## Data sources

Historical OHLCV data comes from a **resilient provider chain** (retry + circuit
breaker + validation), tried in order and returning the first good result:

| Provider | Role | Key | Notes |
|---|---|---|---|
| **Yahoo Finance** (`yfinance`) | primary | — | Clean NSE (`.NS`) / BSE (`.BO`) data, split/div adjusted |
| **Stooq** | fallback | — | Daily CSV, no key |
| **Twelve Data** | optional fallback | `TWELVEDATA_API_KEY` | Good NSE/BSE coverage (~800 req/day free) |
| **Alpha Vantage** | optional fallback | `ALPHAVANTAGE_API_KEY` | Best-effort (India via `.BSE`; ~25 req/day free) |
| **NSE India** ([NseIndiaApi](https://bennythadikaran.github.io/NseIndiaApi/)) | optional, direct-from-source | — (`USE_NSE_API`) | Unofficial; NSE `.NS` equities + **India VIX**. `pip install "nse[local]"` |

Yahoo stays primary because NSE/BSE have no reliable public bulk price API; the
extra providers add **redundancy and cross-source resilience** and are only used
when enabled. Responses are cached for 15 minutes.

The **NSE India API** pulls straight from NSE (no key, self-throttled to 3 req/s)
and additionally powers the **India VIX** feature (see below). It's unofficial and
NSE blocks many non-Indian/cloud IPs, so it's fully optional and degrades
gracefully — set `USE_NSE_API=false` where NSE isn't reachable.

> On Finnhub: its **free tier has no historical OHLCV** (candles are premium), so
> Finnhub is used for *news/sentiment*, not price data.

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

## Daily data synchronisation (incremental + scheduled)

Beyond the one-shot bulk downloader, the app has an **incremental sync engine**
(`analysis/sync.py`) that keeps the local archive current cheaply:

- **Incremental** — only fetches bars newer than each symbol's last stored date.
- **Idempotent** — upserts, so re-running never duplicates rows.
- **Resilient** — reuses the retry + circuit-breaker provider chain and isolates
  per-symbol failures (one bad symbol never aborts the run).
- **Observable** — records a per-run health summary (`sync_health` table) shown
  on the Status page and `/readyz`.

Run it from the CLI (safe for cron):

```bash
python scripts/sync_data.py                 # sync the most-liquid slice
python scripts/sync_data.py --limit 500     # first 500 symbols
python scripts/sync_data.py --all           # entire universe
python scripts/sync_data.py --symbols RELIANCE TCS INFY
python scripts/sync_data.py --backfill TATACONSUM   # backfill one symbol
```

A **built-in scheduler** runs a post-close sync automatically (default 18:30
local). It starts with `run.py` and is fully configurable:

```bash
python run.py --sync-now        # also run one sync at startup (background)
python run.py --no-scheduler    # disable the daily scheduler
```

Configure via env vars: `ENABLE_SCHEDULER` (default `true`), `SYNC_HOUR` (`18`),
`SYNC_MINUTE` (`30`), `SYNC_LIMIT` (`300`). You can also trigger a sync from the
GUI (**Status → Sync data now**), which calls `POST /api/sync`.

## ML price forecast & verdict (multi-model, multi-horizon)

`analysis/predictor.py` forecasts prices across **five horizons — 1, 2, 7, 10 and
30 trading days** — in a single call, using a choice of models:

- **Multiple models** (`analysis/models.py`): Random Forest, Extra Trees,
  Gradient Boosting, HistGradient Boosting, Ridge, and an **averaging ensemble**
  (RF + HGB + Ridge). All scikit-learn, so there are no fragile native deps.
- **Model selection** — pass `model=auto` (the default) and the predictor runs
  **walk-forward validation** across the candidate models and picks the best
  performer *for that specific stock*, returning a scoreboard of how each did.
- Each horizon produces a **price target**, expected % move, a **verdict**
  (STRONG BUY … STRONG SELL) with horizon-scaled thresholds, a **confidence**,
  and a **walk-forward backtest** (predicted vs. actual, out-of-sample).
- **Honest validation** — `TimeSeriesSplit` with a leakage `gap` equal to the
  horizon means training never overlaps the evaluation window.
- **Persistence** — results are cached (shared cache) and optionally written to
  disk with `joblib` (`ML_PERSIST`), so they survive restarts.

Features are all causal and span three groups:

- **Price/technical** — returns over multiple lookbacks, price-vs-SMA ratios,
  RSI, MACD, volatility, volume ratio, 52-week position, light seasonality.
- **News sentiment** (per stock) — trailing 1/3/7-day sentiment, news volume and
  news flow, from the self-accumulating archive (see below).
- **Broad-market context** (`analysis/market.py`) — index (NIFTY `^NSEI` /
  SENSEX `^BSESN`) returns & volatility, **relative strength** (stock vs index),
  and **global market/geopolitical sentiment**. This means a single macro/news
  signal informs *every* stock — even those with no per-company news coverage
  ("the market moves the stock; news moves the market"). Toggle with
  `USE_MARKET_FEATURES`.
- **India VIX** (`mkt_vix`, `mkt_vix_chg_5`) — the market-wide volatility /
  "fear" gauge, sourced directly from NSE via the NSE India API. Added to the
  market context automatically when NSE is reachable (`USE_VIX_FEATURE`), and
  the model bundle self-invalidates when this feature group appears or drops out.

The forecast card on the Analyze view lets you switch models and horizons live.

> Note: multi-day price prediction is genuinely hard; treat the forecast as a
> statistical estimate, **not** a guarantee. It is decision-support, not advice.

## News & sentiment (financial + geopolitical)

The app can enrich the ML models and the UI with **news sentiment** from
pluggable providers (`analysis/news.py`), all called over plain REST and
**key-gated** — nothing runs unless you supply an API key:

| Provider | Env var | Notes |
|---|---|---|
| [Finnhub](https://finnhub.io/docs/api) | `FINNHUB_API_KEY` | Company + general market news |
| [NewsAPI.ai / EventRegistry](https://www.newsapi.ai/) | `NEWSAPI_AI_KEY` | Ships provider **sentiment**, entities, archive to 2014 |
| [NewsData.io](https://newsdata.io/) | `NEWSDATA_API_KEY` | Company/business/politics/world search |
| [GNews](https://gnews.io/) | `GNEWS_API_KEY` | Search + business top-headlines (archive from 2020) |

Providers are tried in `NEWS_PROVIDERS` order; only those with a key are used.
A dependency-free, finance-tuned **sentiment scorer** (`analysis/sentiment.py`)
scores every headline (handling negations and intensifiers) and blends in any
provider-supplied sentiment.

**The historical-coverage problem — and how this design solves it.** The price
models train on up to *10 years* of history, but free news tiers only expose a
recent window (GNews free ≈ 30 days, Finnhub ≈ 1 year). So news can't
retroactively become a deep training feature on day one. Instead:

- Fetched daily sentiment is **persisted to a self-accumulating archive**
  (`sentiment_daily` in SQLite). Run `scripts/refresh_news.py` on a schedule and
  the archive **deepens every day**, progressively turning short free-tier
  windows into a growing historical dataset the models can learn from.
- The feature pipeline adds **causal** sentiment features (trailing 1/3/7-day
  sentiment, news volume, news flow) wherever the archive has coverage, and
  neutral-fills the rest — so training never breaks and stays leak-free.
- A **real-time news overlay** annotates each forecast immediately (sentiment
  score + headlines) and nudges confidence up/down by a small, transparent
  amount when news agrees/disagrees with the model.
- If you later add a paid archive key (NewsAPI.ai to 2014, GNews to 2020), the
  *same* pipeline instantly benefits from the deeper history — no code changes.

Everything degrades gracefully: with no keys, the app behaves exactly as before.

```bash
# One-off / scheduled sentiment refresh (grows the archive):
python scripts/refresh_news.py                 # priority names
python scripts/refresh_news.py --limit 200     # first 200 symbols
python scripts/refresh_news.py --symbols RELIANCE TCS INFY --market
```

## Robustness & operations

The app is built to degrade gracefully rather than fail:

- **Resilient data layer** (`analysis/providers.py`): retries with exponential
  backoff, a **circuit breaker** per provider, and a **fallback provider**
  (Stooq) when Yahoo is unavailable. Prices are **split/dividend adjusted**.
- **Data-quality validation** (`analysis/validation.py`): bad ticks, non-positive
  prices, duplicate/out-of-order dates and implausible spikes are cleaned and
  reported; the UI shows a **data-freshness badge** (source, as-of date, staleness).
- **Pluggable cache** (`analysis/cache.py`): in-memory by default, **Redis** when
  `REDIS_URL` is set, with **single-flight** protection so the screener doesn't
  fire duplicate fetches.
- **Structured logging** (`analysis/logging_config.py`, plain or `LOG_JSON`),
  optional **Sentry** via `SENTRY_DSN`, and centralised env-driven **config**
  (`analysis/config.py`).
- **Security**: scoped CORS, hardened session cookies, **login rate-limiting**,
  **API rate-limiting**, and **CSRF** protection on the auth forms.
- **Health probes**: `GET /healthz` (liveness) and `GET /readyz` (DB + provider
  breaker + universe readiness + **connectivity self-check**).
- **Provider self-check & recovery UX**: a boot-time probe detects when every
  data source is blocked (e.g. corporate SSL interception) and logs the fix;
  the Status page and Analyze recovery panel surface the same guidance.
- **Tests + CI**: `pytest` suite (indicators, strategy, validation, cache,
  features, predictor, API) and a GitHub Actions workflow.

Run the tests with:

```bash
pip install -r requirements.txt
pytest -q
```

### Troubleshooting: "no data" / SSL certificate errors

If **every** stock reports *"no data"* and the logs show:

```
SSL certificate problem: unable to get local issuer certificate
```

then your network is **SSL-inspecting HTTPS** (common on corporate / managed
Windows machines) and the data library (`yfinance` / `curl_cffi`) doesn't trust
the proxy's root certificate — so every price fetch fails. This is an
environment issue, not an app bug. Fixes, in order of preference:

1. **Trust the OS certificate store** (recommended on Windows):

   ```bash
   pip install pip-system-certs
   ```

   This makes Python's HTTPS clients use the Windows trust store, which already
   trusts your corporate CA. It's included in `requirements.txt` on Windows.

2. **Point at your corporate root CA** explicitly:

   ```powershell
   $env:CURL_CA_BUNDLE   = "C:\path\to\corporate-root-ca.pem"
   $env:REQUESTS_CA_BUNDLE = $env:CURL_CA_BUNDLE
   ```

3. **Seed the local archive from an un-inspected network** (e.g. a phone
   hotspot), after which the app serves those stocks **offline**:

   ```bash
   python scripts/sync_data.py --symbols TATACONSUM TCS INFY
   python scripts/sync_data.py --limit 300      # or the popular slice
   ```

The app helps you diagnose this itself:

- On startup, a background **provider self-check** logs a clear, actionable
  warning (with these exact fixes) when it detects every provider is blocked.
- The **Status** tab shows a connectivity banner (green / amber / red) plus
  per-provider health, and a **Sync data now** button.
- When a stock has no data, the Analyze view shows a **recovery panel**
  (Retry · Download this stock · Open Status) instead of a dead-end error, and
  labels the cause (unknown symbol vs. providers unreachable vs. SSL error).

### Configuration (environment variables)

All are optional; sensible defaults apply. Common ones:

| Variable | Default | Purpose |
|---|---|---|
| `DEBUG` | `false` | Flask debug (keep off in production) |
| `SECRET_KEY` | auto | Stable session secret |
| `CORS_ORIGINS` | (same-origin) | Comma-separated allowed origins |
| `REDIS_URL` | — | Shared cache backend |
| `SENTRY_DSN` | — | Error tracking |
| `ML_PERSIST` | `true` | Persist trained model bundles to disk |
| `FETCH_RETRIES` / `FETCH_BACKOFF` | `3` / `0.6` | Data-fetch retry policy |
| `LOGIN_MAX_ATTEMPTS` / `LOGIN_WINDOW_SECONDS` | `8` / `300` | Login throttle |
| `LOG_JSON` | `false` | JSON structured logs |
| `NEWS_ENABLED` | `true` | Master switch for news/sentiment |
| `FINNHUB_API_KEY` / `NEWSAPI_AI_KEY` / `NEWSDATA_API_KEY` / `GNEWS_API_KEY` | — | News provider keys (any/all) |
| `NEWS_PROVIDERS` | `finnhub,newsapi_ai,newsdata,gnews` | Provider try-order |
| `USE_NEWS_FEATURES` | `true` | Feed the sentiment archive into ML features |
| `NEWS_LOOKBACK_DAYS` / `NEWS_CACHE_TTL` | `30` / `1800` | News window / cache TTL |
| `USE_MARKET_FEATURES` | `true` | Feed index context + global sentiment into every model |
| `USE_NSE_API` | `true` | Use the unofficial NSE India API (equity history + India VIX) |
| `USE_VIX_FEATURE` | `true` | Include India VIX in the market context (needs NSE API) |
| `NSE_SERVER_MODE` | `false` | Use `nse[server]` (httpx/http2) on cloud/servers |
| `TWELVEDATA_API_KEY` / `ALPHAVANTAGE_API_KEY` | — | Optional extra price-data fallbacks |

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

**One command, fully auto-configured** (recommended):

```bash
pip install -r requirements.txt
python run.py
#  -> http://127.0.0.1:5000
```

`run.py` is the single entry point: it loads config + secrets from `.env`,
initialises the SQLite store, warms the news/sentiment archive in the background
(non-blocking), then serves the app. Handy flags:

```bash
python run.py --refresh-news 25   # also fetch company news for 25 symbols on start
python run.py --no-warm           # skip the background news warm-up
python run.py --port 8080         # override host/port
```

`python app.py` still works if you only want the bare Flask server.

### Secrets & config (`.env`)

Put keys in a git-ignored `.env` at the repo root (loaded automatically):

```dotenv
FINNHUB_API_KEY=your-finnhub-key
GNEWS_API_KEY=your-gnews-key
NEWS_PROVIDERS=finnhub,gnews,newsdata,newsapi_ai
USE_NEWS_FEATURES=true
USE_MARKET_FEATURES=true
# Direct-from-source NSE India data (equity history + India VIX):
USE_NSE_API=true
USE_VIX_FEATURE=true
# Optional extra price providers (fallbacks):
# TWELVEDATA_API_KEY=your-twelvedata-key
# ALPHAVANTAGE_API_KEY=your-alphavantage-key
```

Real environment variables always override `.env`.

### Settings panel (live, in the GUI)

The dashboard has a **⚙ Settings** button (top-right) that edits a curated set of
options **without a restart** — feature toggles (news, market context, India VIX,
NSE API), news tuning, API keys, and ML knobs. Changes are validated, applied
live (the provider chain, news-provider cache and NSE client are rebuilt and the
result cache is cleared), and persisted to a git-ignored
`analysis/data/runtime_config.json` so they survive restarts.

Precedence: dataclass defaults < `.env` / env vars < settings-panel overrides.
API keys are **write-only** in the UI — the API only reports whether each is set,
never the value. The endpoint (`GET`/`POST /api/config`) is login-gated and the
`POST` requires a CSRF token.

## Project structure

```
stock_market_analysis/
├── run.py                  # single entry point: config + init + news warm-up + serve
├── app.py                  # Flask app + REST API + auth gate + health probes
├── analysis/
│   ├── config.py           # centralised, env-driven settings
│   ├── runtime_config.py   # GUI-editable, persisted config overrides (settings panel)
│   ├── logging_config.py   # structured logging (+ optional Sentry)
│   ├── cache.py            # pluggable TTL cache (in-memory/Redis) + single-flight
│   ├── validation.py       # OHLCV data-quality cleaning + reports
│   ├── providers.py        # data providers (Yahoo/Stooq/TwelveData/AlphaVantage/NSE) w/ retry + breaker
│   ├── nse_client.py       # optional NSE India API wrapper (equity history + India VIX)
│   ├── security.py         # rate limiting + CSRF helpers
│   ├── universe.py         # Full NSE universe loader (live + bundled CSV fallback)
│   ├── auth.py             # File-backed users + PBKDF2 password hashing
│   ├── store.py            # SQLite local history store (schema-versioned)
│   ├── data_fetcher.py     # local-first history + provider top-up + freshness meta
│   ├── indicators.py       # RSI, MACD, SMA/EMA, Bollinger, VWAP, ATR, ADX, Stoch, OBV
│   ├── strategy.py         # rule-based scoring engine for the three horizons
│   ├── features.py         # causal ML feature engineering (+ news sentiment)
│   ├── models.py           # model zoo (RF/ET/GB/HGB/Ridge/ensemble)
│   ├── predictor.py        # multi-model, multi-horizon forecast + walk-forward
│   ├── sentiment.py        # dependency-free financial sentiment scorer
│   ├── news.py             # pluggable news providers (Finnhub/NewsAPI.ai/NewsData/GNews)
│   ├── market.py           # broad-market index context + global sentiment + India VIX
│   └── data/EQUITY_L.csv   # bundled NSE equity master (offline fallback)
├── scripts/download_history.py  # bulk 10-year history downloader -> SQLite
├── scripts/refresh_news.py      # populate the self-accumulating sentiment archive
├── tests/                  # pytest suite (unit + API integration)
├── .github/workflows/ci.yml     # lint + test CI
├── templates/login.html    # login / register screen (CSRF-protected)
├── templates/index.html    # dashboard UI
├── static/css/style.css
├── static/js/app.js        # frontend logic + Plotly charts
├── requirements.txt
├── pytest.ini
└── README.md
```

## API

All API and page routes require an authenticated session (APIs return `401`
when logged out).

| Endpoint | Description |
|---|---|
| `GET /login`, `GET /logout` | Login / register screen, sign out |
| `GET /healthz`, `GET /readyz` | Liveness / readiness probes (unauthenticated) |
| `GET /api/models` | Available ML models + supported horizons |
| `GET /api/stocks` | Full screening universe (all NSE stocks) |
| `GET /api/search?q=` | Search the full universe by name/symbol |
| `GET /api/analyze?symbol=RELIANCE&exchange=NSE&model=auto` | Full analysis (chart + signals + multi-horizon ML + news + data freshness) |
| `GET /api/predict?symbol=RELIANCE&exchange=NSE&model=auto&horizons=1,7,30` | Multi-model / multi-horizon forecast + walk-forward backtest |
| `GET /api/news?symbol=RELIANCE&exchange=NSE` / `?scope=market` | Company or market/geopolitical news + aggregate sentiment |
| `GET /api/screener?horizon=short_term&exchange=NSE&limit=150` | Rank the most liquid stocks (`intraday` / `short_term` / `long_term`) |
| `GET /api/config` / `POST /api/config` | Read / update GUI-editable settings (login-gated; `POST` needs CSRF, secrets masked) |
| `GET /api/status` | Operational status: store, providers, provider health, connectivity, sync, predictions, models |
| `POST /api/sync` | Trigger an incremental data sync (login-gated; needs CSRF) |
| `GET /api/predictions?symbol=` | Prediction audit trail (model/data/feature/timestamp provenance) |
| `GET /api/nse?resource=status\|actions\|quote\|health` | Optional NSE enrichment (market status, corporate actions, quote) |

## Refreshing the stock list

The universe auto-refreshes from NSE every 7 days. To force a refresh, delete
`analysis/data/EQUITY_L.csv` (it will re-download on next start) or call
`universe.reload_universe()`.
