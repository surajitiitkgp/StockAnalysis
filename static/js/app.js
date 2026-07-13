"use strict";

const state = {
  exchange: "NSE",
  scanExchange: "NSE",
  horizon: "intraday",
  selected: null,
  model: "auto",
  prediction: null,
  activeHorizon: null,
};

const $ = (id) => document.getElementById(id);

// Wrap fetch so an expired session bounces the user back to login.
async function apiFetch(url) {
  const res = await fetch(url);
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("Session expired. Redirecting to login…");
  }
  return res;
}

// ----- Tab switching -----
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    const tab = t.dataset.tab;
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    $(tab + "-view").classList.add("active");
    if (tab === "status") loadStatus();
  });
});

// ----- Exchange toggles -----
document.querySelectorAll(".ex").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".ex").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.exchange = b.dataset.ex;
  });
});
document.querySelectorAll(".ex2").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".ex2").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.scanExchange = b.dataset.ex;
  });
});
document.querySelectorAll(".seg-btn").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    state.horizon = b.dataset.h;
  });
});

// ----- Model selector (populated from /api/models) -----
const modelSelect = $("modelSelect");
modelSelect.addEventListener("change", () => { state.model = modelSelect.value; });

async function loadModels() {
  try {
    const res = await apiFetch("/api/models");
    const data = await res.json();
    if (!data.models) return;
    modelSelect.innerHTML = data.models
      .map((m) => `<option value="${m.key}">${m.label}</option>`)
      .join("");
    modelSelect.value = state.model;
  } catch (_) { /* keep the default option */ }
}
loadModels();

// ----- Search with suggestions -----
const searchEl = $("search");
const sugEl = $("suggestions");
let searchTimer = null;

searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = searchEl.value.trim();
  searchTimer = setTimeout(() => fetchSuggestions(q), 180);
});
searchEl.addEventListener("focus", () => {
  if (searchEl.value.trim() === "") fetchSuggestions("");
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) sugEl.classList.add("hidden");
});

async function fetchSuggestions(q) {
  try {
    const res = await apiFetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (!data.length) { sugEl.classList.add("hidden"); return; }
    sugEl.innerHTML = data
      .map((d) => `<div class="sug" data-sym="${d.symbol}" data-name="${d.name}">
          <span>${d.name}</span><span class="sym">${d.symbol}</span></div>`)
      .join("");
    sugEl.classList.remove("hidden");
    sugEl.querySelectorAll(".sug").forEach((el) => {
      el.addEventListener("click", () => {
        state.selected = { symbol: el.dataset.sym, name: el.dataset.name };
        searchEl.value = `${el.dataset.name} (${el.dataset.sym})`;
        sugEl.classList.add("hidden");
        analyze();
      });
    });
  } catch (_) { sugEl.classList.add("hidden"); }
}

// ----- Analyze -----
$("analyzeBtn").addEventListener("click", () => {
  if (!state.selected) {
    const v = searchEl.value.trim();
    if (v) state.selected = { symbol: v.replace(/\s*\(.*\)$/, "").trim(), name: v };
  }
  analyze();
});

async function analyze() {
  if (!state.selected) { showError("Please search and pick a stock first."); return; }
  hide("errorBox"); hide("result"); show("loader");
  try {
    const url = `/api/analyze?symbol=${encodeURIComponent(state.selected.symbol)}&exchange=${state.exchange}&model=${encodeURIComponent(state.model)}`;
    const res = await apiFetch(url);
    const data = await res.json();
    if (!res.ok) {
      // A structured "no data" response carries recovery actions we can render.
      if (data && data.recovery) { hide("loader"); renderRecovery(data); return; }
      throw new Error(data.error || "Analysis failed");
    }
    renderResult(data);
    hide("loader"); show("result");
  } catch (e) {
    hide("loader"); showError(e.message);
  }
}

// Render an actionable recovery panel when a stock has no data available.
function renderRecovery(data) {
  const box = $("errorBox");
  const acts = (data.recovery && data.recovery.actions) || [];
  const diag = data.diagnosis || {};
  const chips = [];
  if (diag.symbol_known === false) chips.push("unknown symbol");
  else if (diag.ssl_issue) chips.push("SSL / certificate error");
  else if (diag.all_providers_down) chips.push("providers unreachable");
  if (diag.has_local_data) chips.push("stale local copy exists");
  const chipHtml = chips.length
    ? `<div class="rec-chips">${chips.map((c) => `<span class="rec-chip">${c}</span>`).join("")}</div>` : "";

  const btns = acts.map((a) =>
    `<button class="rec-act" data-act="${a.id}" data-sym="${a.symbol || ""}">${a.label}</button>`).join("");

  box.className = "errorbox recovery";
  box.innerHTML = `
    <div class="rec-title">${data.error || "No data available"}</div>
    <div class="rec-hint">${data.hint || ""}</div>
    ${chipHtml}
    <div class="rec-actions">${btns}</div>
    <div id="recMsg" class="rec-msg hidden"></div>`;
  box.classList.remove("hidden");

  box.querySelectorAll(".rec-act").forEach((b) => {
    b.addEventListener("click", () => handleRecovery(b.dataset.act, b.dataset.sym));
  });
}

async function handleRecovery(act, sym) {
  if (act === "retry") { analyze(); return; }
  if (act === "status") { document.querySelector('.tab[data-tab="status"]').click(); return; }
  if (act === "download") {
    const msg = $("recMsg");
    msg.className = "rec-msg"; msg.textContent = `Starting download for ${sym}…`;
    try {
      const res = await fetch("/api/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
        body: JSON.stringify({ symbols: [sym], force: true }),
      });
      if (res.status === 401) { window.location.href = "/login"; return; }
      const d = await res.json();
      if (d.status === "already_running") {
        msg.textContent = "A data sync is already running — try Retry in a moment.";
      } else {
        msg.textContent = `Downloading ${sym} in the background. Click Retry in ~15–30s.`;
      }
    } catch (_) {
      msg.className = "rec-msg err";
      msg.textContent = "Couldn’t start the download. Check the Status page.";
    }
  }
}

function fmt(n, d = 2) {
  if (n === null || n === undefined || isNaN(n)) return "—";
  return Number(n).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
}
function fmtCr(n) {
  if (!n) return "—";
  if (n >= 1e7) return "₹" + (n / 1e7).toFixed(2) + " Cr";
  return "₹" + fmt(n, 0);
}
function verdictClass(v) {
  return "v-" + v.toLowerCase().replace(/\s+/g, "-");
}

function renderResult(data) {
  const i = data.info;
  $("stockName").textContent = `${i.name}`;
  $("stockMeta").textContent = `${i.ticker} · ${i.exchange}` +
    (i.sector ? ` · ${i.sector}` : "") + (i.industry ? ` · ${i.industry}` : "");
  $("stockPrice").textContent = "₹" + fmt(i.price);
  const ch = $("stockChange");
  if (i.change_pct !== null) {
    const up = i.change >= 0;
    ch.textContent = `${up ? "▲" : "▼"} ₹${fmt(Math.abs(i.change))} (${fmt(Math.abs(i.change_pct))}%)`;
    ch.className = "change " + (up ? "up" : "down");
  } else ch.textContent = "";

  const stats = [
    ["Market Cap", fmtCr(i.market_cap)],
    ["P/E", fmt(i.pe)],
    ["P/B", fmt(i.pb)],
    ["52W High", i.week52_high ? "₹" + fmt(i.week52_high) : "—"],
    ["52W Low", i.week52_low ? "₹" + fmt(i.week52_low) : "—"],
    ["Beta", fmt(i.beta)],
    ["ROE", i.roe !== null ? fmt(i.roe * 100) + "%" : "—"],
    ["Div Yield", i.dividend_yield !== null ? fmt(i.dividend_yield * 100) + "%" : "—"],
  ];
  $("statStrip").innerHTML = stats
    .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`)
    .join("");

  const labels = { intraday: "Intraday", short_term: "Short-term Swing", long_term: "Long-term Invest" };
  ["intraday", "short_term", "long_term"].forEach((h) => {
    renderRec($("rec-" + h), labels[h], data.recommendations[h]);
  });

  renderDataBadge(data.data);
  renderPrediction(data.prediction);
  renderNews(data.news);
  drawCharts(data.chart);
}

function sentimentClass(label) {
  if (label === "positive") return "v-buy";
  if (label === "negative") return "v-sell";
  return "v-hold";
}

function renderNews(n) {
  const card = $("newsCard");
  if (!n || !n.available) {
    // Only show the card when there's something to say.
    if (n && n.reason && /no news provider/i.test(n.reason)) {
      card.classList.add("hidden");
      return;
    }
    card.classList.remove("hidden");
    $("newsSentiment").textContent = "N/A";
    $("newsSentiment").className = "verdict v-hold";
    $("newsSub").textContent = (n && n.reason) || "No recent news found.";
    $("newsList").innerHTML = "";
    return;
  }

  card.classList.remove("hidden");
  const agg = n.aggregate || {};
  const badge = $("newsSentiment");
  badge.textContent = (agg.label || "neutral").toUpperCase();
  badge.className = "verdict " + sentimentClass(agg.label);
  $("newsSub").textContent =
    `${n.provider || "news"} · ${agg.count || 0} articles · ` +
    `${agg.positive || 0}▲ / ${agg.negative || 0}▼ · score ${fmt(agg.score, 2)}`;

  const items = n.headlines || [];
  $("newsList").innerHTML = items.length
    ? items.map((h) => {
        const cls = h.sentiment > 0.15 ? "up" : (h.sentiment < -0.15 ? "down" : "neu");
        const src = [h.source, h.published_at].filter(Boolean).join(" · ");
        const title = h.url
          ? `<a href="${h.url}" target="_blank" rel="noopener">${h.title}</a>`
          : h.title;
        return `<div class="news-item">
            <span class="news-dot ${cls}"></span>
            <div class="news-body"><div class="news-h">${title}</div>
            <div class="news-src">${src}</div></div>
          </div>`;
      }).join("")
    : `<div class="predict-note">No recent headlines.</div>`;
}

function renderDataBadge(meta) {
  const el = $("dataBadge");
  if (!meta) { el.classList.add("hidden"); return; }
  const f = meta.freshness || {};
  const parts = [];
  if (meta.provider) parts.push(`Source: ${meta.provider}`);
  else if (meta.source) parts.push(`Source: ${meta.source}`);
  if (f.last_date) parts.push(`As of ${f.last_date}`);
  if (f.rows) parts.push(`${f.rows} bars`);
  const stale = f.stale;
  el.className = "data-badge" + (stale ? " stale" : "");
  el.innerHTML =
    `<span class="dot ${stale ? "bear" : "bull"}"></span>` +
    `<span>${parts.join(" · ") || "Data loaded"}</span>` +
    (stale ? `<span class="stale-tag">stale (${f.age_days}d old)</span>` : "");
  el.classList.remove("hidden");
}

function renderPrediction(p) {
  const card = $("predictCard");
  state.prediction = p;
  card.classList.remove("hidden");

  if (!p || !p.available) {
    $("predictVerdict").textContent = "N/A";
    $("predictVerdict").className = "verdict v-hold";
    $("modelInfo").classList.add("hidden");
    $("horizonSelect").classList.add("hidden");
    $("predictBody").innerHTML =
      `<div class="predict-note">${(p && p.reason) || "Forecast unavailable for this stock."}</div>`;
    $("predictFoot").textContent = "";
    try { Plotly.purge("predictChart"); } catch (_) {}
    return;
  }

  // Model info + optional auto-selection scoreboard.
  const mi = $("modelInfo");
  const model = p.model || {};
  let miHtml = `<span class="mi-label">Model:</span> <b>${model.label || model.key}</b>`;
  if (model.selected_from === "auto") miHtml += ` <span class="mi-auto">auto-selected</span>`;
  if (p.news_features_used) miHtml += ` <span class="mi-news">news-tuned</span>`;
  if (p.news && p.news.available)
    miHtml += ` <span class="mi-news">news ${p.news.label || ""} ${fmt(p.news.score, 2)}</span>`;
  if (p.from_cache) miHtml += ` <span class="mi-cache">cached</span>`;
  if (model.scoreboard && model.scoreboard.length) {
    miHtml += `<div class="scoreboard">` +
      model.scoreboard.map((s, idx) =>
        `<span class="sb ${idx === 0 ? "best" : ""}">${s.label}: ${fmt(s.directional_accuracy_pct, 0)}%</span>`
      ).join("") + `</div>`;
  }
  mi.innerHTML = miHtml;
  mi.classList.remove("hidden");

  $("predictSub").textContent =
    `${model.label || "ML"} · horizons ${(p.horizons || []).map((h) => h.days + "d").join(" / ")}`;

  // Horizon selector chips.
  const horizons = p.horizons || [];
  const primary = horizons.find((h) => h.days === (p.horizon_days || 7)) || horizons[0];
  state.activeHorizon = state.activeHorizon && horizons.some((h) => h.days === state.activeHorizon)
    ? state.activeHorizon : (primary ? primary.days : null);

  const hs = $("horizonSelect");
  hs.innerHTML = horizons.map((h) =>
    `<button class="hz-btn ${h.days === state.activeHorizon ? "active" : ""}" data-h="${h.days}">
       ${h.days}d
       <span class="hz-move ${h.expected_return_pct >= 0 ? "up" : "down"}">${h.expected_return_pct >= 0 ? "+" : ""}${fmt(h.expected_return_pct, 1)}%</span>
     </button>`).join("");
  hs.classList.remove("hidden");
  hs.querySelectorAll(".hz-btn").forEach((b) => {
    b.addEventListener("click", () => {
      state.activeHorizon = parseInt(b.dataset.h, 10);
      renderHorizon();
    });
  });

  renderHorizon();
}

function renderHorizon() {
  const p = state.prediction;
  if (!p || !p.available) return;
  const h = (p.horizons || []).find((x) => x.days === state.activeHorizon) || p.horizons[0];
  if (!h) return;

  document.querySelectorAll(".hz-btn").forEach((b) =>
    b.classList.toggle("active", parseInt(b.dataset.h, 10) === h.days));

  const v = $("predictVerdict");
  v.textContent = h.verdict;
  v.className = "verdict " + verdictClass(h.verdict);

  const up = h.expected_return_pct >= 0;
  const m = h.metrics || {};
  $("predictBody").innerHTML = `
    <div class="predict-grid">
      <div class="pstat"><div class="k">Current</div><div class="val">₹${fmt(h.last_price)}</div></div>
      <div class="pstat"><div class="k">Forecast (${h.days}d)</div>
        <div class="val ${up ? "up" : "down"}">₹${fmt(h.forecast_price)}</div></div>
      <div class="pstat"><div class="k">Expected move</div>
        <div class="val ${up ? "up" : "down"}">${up ? "+" : ""}${fmt(h.expected_return_pct)}%</div></div>
      <div class="pstat"><div class="k">Confidence</div><div class="val">${fmt(h.confidence, 0)}%</div></div>
      <div class="pstat"><div class="k">Dir. accuracy</div><div class="val">${fmt(m.directional_accuracy_pct, 0)}%</div></div>
      <div class="pstat"><div class="k">Backtest err (MAE)</div><div class="val">±${fmt(m.mae_pct)}%</div></div>
    </div>` + bandHtml(h.forecast_band) + attributionHtml(h.signal_attribution);

  $("predictFoot").textContent =
    `Trained on ${p.train_samples} samples (${p.history_days} trading days), ${p.trained_at || ""}. ` +
    `Target date ≈ ${h.forecast_date}. Walk-forward R²=${fmt(m.r2, 2)}. Key drivers: ` +
    (p.top_features || []).map((f) => f.name).join(", ") +
    ". Statistical estimate — not investment advice.";

  drawPredictChart(h.backtest, h.forecast_date, h.forecast_price, h.last_price);
}

// Probabilistic forecast band (P10 / P50 / P90) from the quantile model.
function bandHtml(band) {
  if (!band) return "";
  const lo = band.p10_price, mid = band.p50_price, hi = band.p90_price;
  const span = Math.max(hi - lo, 1e-6);
  const midPct = Math.max(0, Math.min(100, ((mid - lo) / span) * 100));
  return `
    <div class="fc-band">
      <div class="fc-band-head">Likely range in ${state.activeHorizon}d
        <span class="fc-band-sub">80% confidence interval</span></div>
      <div class="fc-band-track">
        <div class="fc-band-fill"></div>
        <div class="fc-band-mid" style="left:${midPct}%"></div>
      </div>
      <div class="fc-band-labels">
        <span class="down">₹${fmt(lo)}<small>P10 ${fmt(band.p10_return_pct, 1)}%</small></span>
        <span class="mid">₹${fmt(mid)}<small>median</small></span>
        <span class="up">₹${fmt(hi)}<small>P90 +${fmt(band.p90_return_pct, 1)}%</small></span>
      </div>
    </div>`;
}

// Signal attribution: how much price vs. news vs. geopolitics drove the call.
function attributionHtml(attr) {
  if (!attr || !attr.sources || !attr.sources.length) return "";
  const icon = { price: "📈", news: "📰", geopolitics: "🌍" };
  const nice = { price: "Price / technicals", news: "Company news", geopolitics: "Geopolitics / macro" };
  const rows = attr.sources.map((s) => {
    const w = Math.max(0, Math.min(100, s.share_pct));
    const dirCls = s.direction === "up" ? "up" : "down";
    return `<div class="attr-row">
        <div class="attr-name">${icon[s.source] || "•"} ${nice[s.source] || s.source}</div>
        <div class="attr-bar"><div class="attr-fill ${dirCls}" style="width:${w}%"></div></div>
        <div class="attr-val ${dirCls}">${s.share_pct}%</div>
      </div>`;
  }).join("");
  return `<div class="attr-block">
      <div class="attr-head">Signal attribution
        <span class="attr-sub">share of the forecast move by data source</span></div>
      ${rows}
    </div>`;
}

function drawPredictChart(backtest, fDate, fPrice, lastPrice) {
  if (!backtest || !backtest.length) { try { Plotly.purge("predictChart"); } catch (_) {} return; }
  const dates = backtest.map((b) => b.date);
  const actual = backtest.map((b) => b.actual);
  const predicted = backtest.map((b) => b.predicted);

  const actualTrace = {
    type: "scatter", mode: "lines+markers", x: dates, y: actual,
    name: "Actual", line: { color: "#4f8cff", width: 2 }, marker: { size: 6 },
  };
  const predTrace = {
    type: "scatter", mode: "lines+markers", x: dates, y: predicted,
    name: "Predicted", line: { color: "#ffb648", width: 2, dash: "dot" }, marker: { size: 6 },
  };
  const fwdTrace = {
    type: "scatter", mode: "markers+text", x: [fDate], y: [fPrice],
    name: "7d forecast", text: ["▲ forecast"], textposition: "top center",
    marker: { size: 11, color: fPrice >= lastPrice ? "#1fd286" : "#ff5c7a", symbol: "diamond" },
  };
  const layout = {
    ...layoutBase, margin: { l: 50, r: 20, t: 8, b: 28 },
    yaxis: { ...layoutBase.yaxis, title: "₹" },
    title: { text: "Last 7 days: predicted vs actual (+ forecast)", font: { size: 11, color: "#8a97b1" }, x: 0.02, y: 0.97 },
  };
  Plotly.newPlot("predictChart", [actualTrace, predTrace, fwdTrace], layout, config);
}

function renderRec(el, title, rec) {
  const pct = Math.max(0, Math.min(100, (rec.score + 100) / 2)); // map -100..100 -> 0..100
  const levels = (rec.entry !== null)
    ? `<div class="levels">
         <div class="lvl"><div class="lk">Entry</div><div class="lv">₹${fmt(rec.entry)}</div></div>
         <div class="lvl target"><div class="lk">Target</div><div class="lv">₹${fmt(rec.target)}</div></div>
         <div class="lvl stop"><div class="lk">Stop</div><div class="lv">₹${fmt(rec.stop_loss)}</div></div>
       </div>` : "";
  const sigs = rec.signals.slice(0, 6).map((s) =>
    `<div class="sig"><span class="dot ${s.direction}"></span>
       <span class="sig-label">${s.label}</span>
       <span class="sig-detail">${s.detail || ""}</span></div>`).join("");

  el.innerHTML = `
    <div class="rc-head">
      <span class="horizon">${title}</span>
      <span class="verdict ${verdictClass(rec.verdict)}">${rec.verdict}</span>
    </div>
    <div class="gauge-wrap">
      <div class="gauge-track">
        <div class="gauge"></div>
        <div class="gauge-fill" style="left:${pct}%"></div>
      </div>
      <span class="score-num">${rec.score > 0 ? "+" : ""}${fmt(rec.score, 0)}</span>
    </div>
    ${levels}
    <div class="signals">${sigs}</div>`;
}

// ----- Charts (Plotly) -----
const layoutBase = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#8a97b1", family: "Inter" },
  margin: { l: 50, r: 20, t: 10, b: 30 },
  xaxis: { gridcolor: "#1c2438", showgrid: true },
  yaxis: { gridcolor: "#1c2438", showgrid: true },
  showlegend: true,
  legend: { orientation: "h", y: 1.12, font: { size: 10 } },
};
const config = { responsive: true, displayModeBar: false };

function drawCharts(c) {
  if (!c || !c.dates) return;
  const x = c.dates;

  const candle = {
    type: "candlestick", x,
    open: c.open, high: c.high, low: c.low, close: c.close,
    name: "Price",
    increasing: { line: { color: "#1fd286" } },
    decreasing: { line: { color: "#ff5c7a" } },
  };
  const line = (y, name, color, width = 1.3, dash = "solid") =>
    ({ type: "scatter", mode: "lines", x, y, name, line: { color, width, dash } });

  const priceData = [
    { type: "scatter", mode: "lines", x, y: c.bb_upper, name: "BB", line: { color: "#46506b", width: 0.8 }, showlegend: false },
    { type: "scatter", mode: "lines", x, y: c.bb_lower, name: "BB", fill: "tonexty", fillcolor: "rgba(76,140,255,0.06)", line: { color: "#46506b", width: 0.8 }, showlegend: false },
    candle,
    line(c.sma20, "SMA20", "#4f8cff"),
    line(c.sma50, "SMA50", "#ffb648"),
    line(c.sma200, "SMA200", "#6c5ce7", 1.6),
  ];
  Plotly.newPlot("priceChart", priceData,
    { ...layoutBase, yaxis: { ...layoutBase.yaxis, title: "₹" }, xaxis: { ...layoutBase.xaxis, rangeslider: { visible: false } } },
    config);

  const rsiData = [
    line(c.rsi, "RSI", "#4f8cff", 1.6),
  ];
  const rsiLayout = {
    ...layoutBase, margin: { l: 50, r: 20, t: 6, b: 24 }, showlegend: false,
    yaxis: { ...layoutBase.yaxis, range: [0, 100], title: "RSI" },
    shapes: [
      { type: "line", x0: x[0], x1: x[x.length - 1], y0: 70, y1: 70, line: { color: "#ff5c7a", width: 0.8, dash: "dot" } },
      { type: "line", x0: x[0], x1: x[x.length - 1], y0: 30, y1: 30, line: { color: "#1fd286", width: 0.8, dash: "dot" } },
    ],
  };
  Plotly.newPlot("rsiChart", rsiData, rsiLayout, config);

  const colors = (c.macd_hist || []).map((v) => (v >= 0 ? "#1fd286" : "#ff5c7a"));
  const macdData = [
    { type: "bar", x, y: c.macd_hist, name: "Hist", marker: { color: colors } },
    line(c.macd, "MACD", "#4f8cff", 1.4),
    line(c.macd_signal, "Signal", "#ffb648", 1.4),
  ];
  Plotly.newPlot("macdChart", macdData,
    { ...layoutBase, margin: { l: 50, r: 20, t: 6, b: 24 }, yaxis: { ...layoutBase.yaxis, title: "MACD" } },
    config);
}

// ----- Screener -----
$("scanBtn").addEventListener("click", scan);

async function scan() {
  hide("screenResult"); show("scanLoader");
  try {
    const res = await apiFetch(`/api/screener?horizon=${state.horizon}&exchange=${state.scanExchange}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Scan failed");
    renderScreen(data);
    hide("scanLoader"); show("screenResult");
  } catch (e) {
    hide("scanLoader"); alert(e.message);
  }
}

function renderScreen(data) {
  $("buyList").innerHTML = data.top_buys.length
    ? data.top_buys.map(rowHtml).join("") : emptyRow("No buy candidates right now.");
  $("sellList").innerHTML = data.top_sells.length
    ? data.top_sells.map(rowHtml).join("") : emptyRow("No sell signals right now.");
  attachRowClicks();
}

function emptyRow(t) { return `<div class="row" style="cursor:default;color:var(--muted)">${t}</div>`; }

function rowHtml(r) {
  const up = (r.change_pct || 0) >= 0;
  const lv = (r.entry !== null)
    ? `<span>Entry <b>₹${fmt(r.entry)}</b></span><span>Target <b>₹${fmt(r.target)}</b></span><span>Stop <b>₹${fmt(r.stop_loss)}</b></span>`
    : "";
  return `<div class="row" data-sym="${r.symbol}" data-name="${r.name}" data-ex="${r.exchange}">
    <div class="r-top">
      <div><div class="r-name">${r.name}</div><span class="r-sym">${r.symbol}.${r.exchange === "BSE" ? "BO" : "NS"}</span></div>
      <div style="text-align:right">
        <span class="pill ${verdictClass(r.verdict)}">${r.verdict}</span>
        <div class="r-price" style="margin-top:6px;color:${up ? "var(--green)" : "var(--red)"}">₹${fmt(r.price)} (${up ? "+" : ""}${fmt(r.change_pct)}%)</div>
      </div>
    </div>
    <div class="r-meta"><span>Score <b>${r.score > 0 ? "+" : ""}${fmt(r.score, 0)}</b></span><span>Conf <b>${fmt(r.confidence, 0)}%</b></span>${lv}</div>
  </div>`;
}

function attachRowClicks() {
  document.querySelectorAll(".row[data-sym]").forEach((el) => {
    el.addEventListener("click", () => {
      state.selected = { symbol: el.dataset.sym, name: el.dataset.name };
      state.exchange = el.dataset.ex;
      document.querySelectorAll(".ex").forEach((x) =>
        x.classList.toggle("active", x.dataset.ex === el.dataset.ex));
      searchEl.value = `${el.dataset.name} (${el.dataset.sym})`;
      document.querySelector('.tab[data-tab="analyze"]').click();
      analyze();
    });
  });
}

// ----- Settings modal -----
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";
let settingsFields = [];

function openSettings() {
  show("settingsOverlay");
  hide("settingsMsg");
  $("settingsBody").innerHTML = '<div class="settings-loading">Loading…</div>';
  loadSettings();
}
function closeSettings() { hide("settingsOverlay"); }

$("settingsBtn").addEventListener("click", openSettings);
$("settingsClose").addEventListener("click", closeSettings);
$("settingsCancel").addEventListener("click", closeSettings);
$("settingsOverlay").addEventListener("click", (e) => {
  if (e.target.id === "settingsOverlay") closeSettings();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("settingsOverlay").classList.contains("hidden")) closeSettings();
});

async function loadSettings() {
  try {
    const res = await apiFetch("/api/config");
    const data = await res.json();
    settingsFields = data.settings || [];
    renderSettings(settingsFields);
  } catch (_) {
    $("settingsBody").innerHTML = '<div class="settings-loading">Failed to load settings.</div>';
  }
}

function renderSettings(fields) {
  const groups = {};
  fields.forEach((f) => { (groups[f.group] = groups[f.group] || []).push(f); });
  const esc = (s) => String(s == null ? "" : s).replace(/"/g, "&quot;");

  $("settingsBody").innerHTML = Object.keys(groups).map((g) => `
    <div class="settings-group">
      <h3>${g}</h3>
      ${groups[g].map(fieldHtml).join("")}
    </div>`).join("");

  function fieldHtml(f) {
    const help = f.help ? `<span class="set-help">${f.help}</span>` : "";
    let control = "";
    if (f.type === "bool") {
      control = `<label class="switch">
        <input type="checkbox" data-key="${f.key}" data-type="bool" ${f.value ? "checked" : ""}/>
        <span class="slider"></span></label>`;
    } else if (f.type === "int") {
      control = `<input class="set-input" type="number" data-key="${f.key}" data-type="int"
        value="${esc(f.value)}" ${f.min != null ? `min="${f.min}"` : ""} ${f.max != null ? `max="${f.max}"` : ""}/>`;
    } else if (f.type === "secret") {
      control = `<input class="set-input" type="password" data-key="${f.key}" data-type="secret"
        placeholder="${f.configured ? "•••••••• (set — leave blank to keep)" : "not set"}" autocomplete="new-password"/>`;
    } else {
      control = `<input class="set-input" type="text" data-key="${f.key}" data-type="${f.type}" value="${esc(f.value)}"/>`;
    }
    return `<div class="set-row">
      <div class="set-label"><span>${f.label}</span>${help}</div>
      <div class="set-control">${control}</div>
    </div>`;
  }
}

$("settingsSave").addEventListener("click", saveSettings);

async function saveSettings() {
  const changes = {};
  document.querySelectorAll("#settingsBody [data-key]").forEach((el) => {
    const key = el.dataset.key;
    const type = el.dataset.type;
    if (type === "bool") {
      changes[key] = el.checked;
    } else if (type === "secret") {
      if (el.value !== "") changes[key] = el.value;  // blank = keep current
    } else {
      changes[key] = el.value;
    }
  });

  const btn = $("settingsSave");
  btn.disabled = true; btn.textContent = "Saving…";
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
      body: JSON.stringify({ changes }),
    });
    if (res.status === 401) { window.location.href = "/login"; return; }
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Save failed");
    settingsMsg(`Saved ${data.applied.length} setting${data.applied.length === 1 ? "" : "s"}. Re-run Analyze to apply.`, true);
    settingsFields = data.settings || settingsFields;
    renderSettings(settingsFields);
  } catch (e) {
    settingsMsg(e.message, false);
  } finally {
    btn.disabled = false; btn.textContent = "Save changes";
  }
}

function settingsMsg(text, ok) {
  const el = $("settingsMsg");
  el.textContent = text;
  el.className = "settings-msg " + (ok ? "ok" : "err");
}

// ----- helpers -----
function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }
function showError(msg) { const e = $("errorBox"); e.textContent = msg; e.classList.remove("hidden"); }


// ----- Status page (Sec. 13) -----
function statusMsg(text, ok) {
  const el = $("statusMsg");
  if (!el) return;
  el.textContent = text;
  el.className = "settings-msg " + (ok ? "ok" : "err");
}

async function loadStatus() {
  const body = $("statusBody");
  body.innerHTML = '<div class="settings-loading">Loading status…</div>';
  try {
    const res = await apiFetch("/api/status");
    const d = await res.json();
    renderStatus(d);
  } catch (e) {
    body.innerHTML = '<div class="settings-loading">Failed to load status.</div>';
  }
}

function pill(state, kind) {
  const good = ["ok", "closed", "reachable", "up_to_date"];
  const bad = ["error", "open", "degraded", "unreachable"];
  const warn = ["limited", "empty", "breaker_open"];
  let cls = "neu";
  if (good.includes(state)) cls = "up";
  else if (warn.includes(state)) cls = "neu";
  else if (bad.includes(state)) cls = "down";
  return `<span class="status-pill ${cls}">${state || "—"}</span>`;
}

function renderStatus(d) {
  const store = d.store || {};
  const preds = d.predictions || {};
  const sent = d.sentiment || {};
  const sync = (d.sync || [])[0];
  const cards = [];

  // Data store card.
  cards.push(`<div class="status-card">
    <h3>📦 Local Data Store</h3>
    <div class="status-kv"><span>Symbols</span><b>${store.symbols ?? "—"}</b></div>
    <div class="status-kv"><span>Price rows</span><b>${(store.rows ?? 0).toLocaleString("en-IN")}</b></div>
    <div class="status-kv"><span>Date range</span><b>${store.min_date || "—"} → ${store.max_date || "—"}</b></div>
    <div class="status-kv"><span>Sentiment rows</span><b>${(sent.rows ?? 0).toLocaleString("en-IN")}</b></div>
  </div>`);

  // Sync health card.
  cards.push(`<div class="status-card">
    <h3>🔄 Data Synchronisation</h3>
    <div class="status-kv"><span>Running now</span><b>${d.sync_running ? "yes" : "no"}</b></div>
    <div class="status-kv"><span>Last run</span>${sync ? pill(sync.status) : "<b>never</b>"}</div>
    <div class="status-kv"><span>When</span><b>${sync ? sync.updated_at : "—"}</b></div>
    <div class="status-kv"><span>Detail</span><b class="status-detail">${sync ? (sync.detail || "") : "Run a sync to populate."}</b></div>
  </div>`);

  // Providers card.
  const ph = d.provider_health || [];
  const provRows = ph.length
    ? ph.map((p) => {
        const note = p.detail ? ` title="${(p.detail || "").replace(/"/g, "&quot;")}"` : "";
        const opt = p.optional ? ' <span class="prov-opt">fallback</span>' : "";
        return `<div class="status-kv"${note}><span>${p.name}${opt}</span>${pill(p.status)}</div>`;
      }).join("")
    : Object.entries(d.providers || {}).map(([k, v]) => `<div class="status-kv"><span>${k}</span>${pill(v)}</div>`).join("");
  cards.push(`<div class="status-card">
    <h3>🌐 Data Providers</h3>${provRows || "<div class='status-kv'><span>none</span></div>"}
  </div>`);

  // Predictions audit card.
  const modes = preds.by_mode || {};
  const modeRows = Object.keys(modes).length
    ? Object.entries(modes).map(([k, v]) => `<div class="status-kv"><span>${k}</span><b>${v}</b></div>`).join("")
    : "<div class='status-kv'><span>none yet</span></div>";
  cards.push(`<div class="status-card">
    <h3>🤖 Predictions (audit)</h3>
    <div class="status-kv"><span>Total logged</span><b>${preds.total ?? 0}</b></div>
    <div class="status-kv"><span>Last</span><b>${preds.last_created || "—"}</b></div>
    ${modeRows}
  </div>`);

  // News + universe card.
  const news = d.news || {};
  cards.push(`<div class="status-card">
    <h3>📰 News &amp; Universe</h3>
    <div class="status-kv"><span>News enabled</span><b>${news.enabled ? "yes" : "no"}</b></div>
    <div class="status-kv"><span>Providers</span><b>${(news.providers || []).join(", ") || "none"}</b></div>
    <div class="status-kv"><span>Universe</span><b>${(d.universe ?? 0).toLocaleString("en-IN")} stocks</b></div>
    <div class="status-kv"><span>ML models</span><b>${(d.models || []).length}</b></div>
  </div>`);

  const banner = renderConnBanner(d.connectivity);
  $("statusBody").innerHTML = banner + `<div class="status-grid">${cards.join("")}</div>`;
}

// A prominent connectivity banner (green = OK, amber/red = degraded/offline).
function renderConnBanner(conn) {
  if (!conn || conn.severity === "unknown") return "";
  if (conn.severity === "ok") {
    return `<div class="conn-banner ok">✓ Market data providers reachable</div>`;
  }
  const fixes = (conn.fixes || []).map((f) => `<li>${f}</li>`).join("");
  const sev = conn.severity === "offline" ? "offline" : "degraded";
  return `<div class="conn-banner ${sev}">
    <div class="conn-title">⚠ ${conn.title || "Data providers unavailable"}</div>
    <div class="conn-hint">${conn.hint || ""}</div>
    ${fixes ? `<ul class="conn-fixes">${fixes}</ul>` : ""}
  </div>`;
}

$("refreshStatusBtn")?.addEventListener("click", loadStatus);
$("syncNowBtn")?.addEventListener("click", async () => {
  const btn = $("syncNowBtn");
  btn.disabled = true; btn.textContent = "Starting…";
  try {
    const res = await fetch("/api/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
      body: JSON.stringify({}),
    });
    if (res.status === 401) { window.location.href = "/login"; return; }
    const d = await res.json();
    if (d.status === "already_running") statusMsg("A sync is already running.", true);
    else statusMsg(`Sync started (up to ${d.limit} symbols). Refresh in a moment to see progress.`, true);
    $("statusMsg").classList.remove("hidden");
  } catch (e) {
    statusMsg("Failed to start sync.", false);
    $("statusMsg").classList.remove("hidden");
  } finally {
    btn.disabled = false; btn.textContent = "Sync data now";
    setTimeout(loadStatus, 2500);
  }
});
