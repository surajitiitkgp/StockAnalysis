"use strict";

const state = {
  exchange: "NSE",
  scanExchange: "NSE",
  horizon: "intraday",
  selected: null,
};

const $ = (id) => document.getElementById(id);

// ----- Tab switching -----
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    const tab = t.dataset.tab;
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    $(tab + "-view").classList.add("active");
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
    const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
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
    const url = `/api/analyze?symbol=${encodeURIComponent(state.selected.symbol)}&exchange=${state.exchange}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Analysis failed");
    renderResult(data);
    hide("loader"); show("result");
  } catch (e) {
    hide("loader"); showError(e.message);
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

  drawCharts(data.chart);
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
    const res = await fetch(`/api/screener?horizon=${state.horizon}&exchange=${state.scanExchange}`);
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

// ----- helpers -----
function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }
function showError(msg) { const e = $("errorBox"); e.textContent = msg; e.classList.remove("hidden"); }
