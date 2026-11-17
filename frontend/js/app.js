/* ============================================================================
   YieldVision shared frontend runtime
   - API client, toast system, formatters
   - Sidebar/topbar shell renderer with the 2026 dark-glass design
   - Plotly theming helpers
   ========================================================================= */

const API = {
  async req(path, opts = {}) {
    const res = await fetch(path, opts);
    const ct = res.headers.get("content-type") || "";
    if (!res.ok) {
      let detail = res.statusText;
      if (ct.includes("application/json")) {
        try { detail = (await res.json()).detail || detail; } catch (_) {}
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    if (ct.includes("application/json")) return res.json();
    return res;
  },
  get(path) { return this.req(path); },
  post(path, body) {
    return this.req(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  },
  patch(path, body) {
    return this.req(path, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  },
  upload(path, file, extra = {}) {
    const fd = new FormData();
    fd.append("file", file);
    for (const [k, v] of Object.entries(extra)) fd.append(k, v);
    return this.req(path, { method: "POST", body: fd });
  },
};

/* ------------------------------------------------------------------ toast -- */
function toast(message, kind = "info", ms = 4200) {
  let host = document.querySelector(".toast-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "toast-host";
    document.body.appendChild(host);
  }
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateY(10px)";
    setTimeout(() => el.remove(), 260);
  }, ms);
}

/* ------------------------------------------------------------- formatting -- */
const fmt = {
  num(v, d = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    return Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  },
  int(v) {
    if (v === null || v === undefined) return "–";
    return Number(v).toLocaleString();
  },
  pct(v, d = 1) {
    if (v === null || v === undefined) return "–";
    return `${(v * 100).toFixed(d)}%`;
  },
  title(s) {
    return String(s || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  },
  date(s) {
    if (!s) return "–";
    try { return new Date(s).toLocaleString(); } catch (_) { return s; }
  },
  text(v) {
    return v === null || v === undefined || v === "" ? "–" : String(v);
  },
};

/* ------------------------------------------------------------------- icons -- */
const ICONS = {
  logo: '<svg viewBox="0 0 24 24" fill="none" stroke="#04120c" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21c0-6 3-10 8-11-1 6-4 9-8 11z"/><path d="M12 21C12 15 9 11 4 10c1 6 4 9 8 11z"/><path d="M12 21v-6"/></svg>',
  home: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>',
  trend: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 3 5-8 4 4"/><path d="M3 21h18"/></svg>',
  layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/></svg>',
  cpu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="7" width="10" height="10" rx="2"/><path d="M4 9h2M4 15h2M18 9h2M18 15h2M9 4v2M15 4v2M9 18v2M15 18v2"/></svg>',
  gauge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 14l4-4"/><path d="M4 18a8 8 0 1 1 16 0"/></svg>',
  bulb: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10.5c.6.6 1 1.4 1 2.5h6c0-1.1.4-1.9 1-2.5A6 6 0 0 0 12 3z"/></svg>',
  target: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/></svg>',
  doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
  menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
  empty: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 10h18M8 15h4"/></svg>',
};

/* -------------------------------------------------------------------- nav -- */
const NAV = [
  { group: "Overview", items: [
    { href: "index.html", label: "Dashboard", icon: "home" },
  ]},
  { group: "Data", items: [
    { href: "upload.html", label: "Upload & Validate", icon: "upload" },
    { href: "explorer.html", label: "Trend Explorer", icon: "trend" },
    { href: "features.html", label: "Feature Engineering", icon: "layers" },
  ]},
  { group: "Modelling", items: [
    { href: "models.html", label: "Train Models", icon: "cpu" },
    { href: "accuracy.html", label: "Accuracy Dashboard", icon: "gauge" },
    { href: "explain.html", label: "Explainability", icon: "bulb" },
  ]},
  { group: "Deliver", items: [
    { href: "predict.html", label: "Predict", icon: "target" },
    { href: "reports.html", label: "Reports & Export", icon: "doc" },
  ]},
];

const PAGE_META = {
  "index.html": { title: "Dashboard", crumb: "Overview" },
  "upload.html": { title: "Upload & Validate", crumb: "Data · ingestion" },
  "explorer.html": { title: "Trend Explorer", crumb: "Data · analytics" },
  "features.html": { title: "Feature Engineering", crumb: "Data · features" },
  "models.html": { title: "Train Models", crumb: "Modelling · training" },
  "accuracy.html": { title: "Accuracy Dashboard", crumb: "Modelling · evaluation" },
  "explain.html": { title: "Explainability", crumb: "Modelling · interpretation" },
  "predict.html": { title: "Predict Yield", crumb: "Deliver · inference" },
  "reports.html": { title: "Reports & Export", crumb: "Deliver · outputs" },
};

function currentPage() {
  const path = location.pathname.split("/").pop() || "index.html";
  return path.includes(".html") ? path : "index.html";
}

function renderShell() {
  const page = currentPage();
  const meta = PAGE_META[page] || { title: "YieldVision", crumb: "" };
  const navHtml = NAV.map((section) => `
    <div class="nav-group-label">${section.group}</div>
    ${section.items.map((item) => `
      <a class="nav-link ${item.href === page ? "active" : ""}" href="${item.href}">
        ${ICONS[item.icon] || ""}<span>${item.label}</span>
      </a>`).join("")}
  `).join("");

  const shell = document.createElement("div");
  shell.className = "shell";
  shell.innerHTML = `
    <aside class="sidebar" id="sidebar">
      <div class="brand">
        <div class="brand-mark">${ICONS.logo}</div>
        <div>
          <div class="brand-name">YieldVision</div>
          <div class="brand-sub">Crop Yield Engine</div>
        </div>
      </div>
      <nav>${navHtml}</nav>
      <div class="sidebar-foot">
        <strong>Climate · Soil · Yield</strong>
        <span id="side-status">Checking data…</span>
      </div>
    </aside>
    <div class="main">
      <header class="topbar">
        <button class="btn btn-ghost btn-sm menu-btn" id="menu-btn">${ICONS.menu}</button>
        <div>
          <div class="crumb">${meta.crumb}</div>
          <h1>${meta.title}</h1>
        </div>
        <div class="topbar-spacer"></div>
        <div id="topbar-actions" class="btn-row"></div>
      </header>
      <main class="view" id="view"></main>
    </div>`;

  document.body.innerHTML = "";
  document.body.appendChild(shell);

  document.getElementById("menu-btn")?.addEventListener("click", () => {
    document.getElementById("sidebar").classList.toggle("open");
  });
  return document.getElementById("view");
}

/* ----------------------------------------------------------- plotly theme -- */
const THEME = {
  paper: "rgba(0,0,0,0)",
  grid: "rgba(150,170,220,0.12)",
  text: "#a7b6d4",
  accent: "#b8ff3d",
  accent2: "#7b8cff",
  accent3: "#c17bff",
  warn: "#ffc14d",
  danger: "#ff6f8b",
  palette: ["#b8ff3d", "#7b8cff", "#46e3ff", "#c17bff", "#ffc14d", "#ff6f8b", "#7ee01a"],
};

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: THEME.paper,
    plot_bgcolor: THEME.paper,
    font: { family: "Inter, sans-serif", color: THEME.text, size: 12 },
    margin: { l: 52, r: 20, t: 42, b: 44 },
    colorway: THEME.palette,
    hoverlabel: { bgcolor: "#0d1220", bordercolor: THEME.grid, font: { color: "#f2f6ff" } },
    xaxis: { gridcolor: THEME.grid, zerolinecolor: THEME.grid, linecolor: THEME.grid },
    yaxis: { gridcolor: THEME.grid, zerolinecolor: THEME.grid, linecolor: THEME.grid },
    legend: { orientation: "h", y: 1.12, x: 0, bgcolor: "rgba(0,0,0,0)" },
  }, extra);
}

const PLOT_CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function drawChart(el, traces, layout = {}) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el) return;
  Plotly.newPlot(el, traces, baseLayout(layout), PLOT_CONFIG);
}

/* --------------------------------------------------------------- helpers -- */
function html(container, markup) {
  if (typeof container === "string") container = document.getElementById(container);
  container.innerHTML = markup;
  return container;
}

function emptyState(message = "No data yet.") {
  return `<div class="empty">${ICONS.empty}<div>${message}</div></div>`;
}

function loading(el, message = "Loading…") {
  html(el, `<div class="empty"><span class="spinner"></span><div style="margin-top:10px">${message}</div></div>`);
}

async function refreshSideStatus() {
  const el = document.getElementById("side-status");
  if (!el) return;
  try {
    const s = await API.get("/api/explorer/summary");
    el.textContent = `${fmt.int(s.rows)} rows · ${s.n_crops} crops · ${s.n_regions} regions`;
  } catch (_) {
    el.textContent = "No dataset loaded yet";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  refreshSideStatus();
});
