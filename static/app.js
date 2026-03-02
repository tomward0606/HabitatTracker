let viewMode = "true"; // "true" | "ndvi"
let lastOut = null;

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function setLoading(on) {
  const el = document.getElementById("loading");
  if (!el) return;
  el.classList.toggle("is-on", on);
  el.setAttribute("aria-hidden", on ? "false" : "true");
}

function setText(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}

function syncDateTitles() {
  const t1 = document.getElementById("t1")?.value;
  const t2 = document.getElementById("t2")?.value;
  setText("titleFirstDate", t1 || "—");
  setText("titleLatestDate", t2 || "—");
}

function setWidth(id, pct) {
  const el = document.getElementById(id);
  if (!el) return;
  const clamped = Math.max(0, Math.min(100, pct));
  el.style.width = `${clamped}%`;
}

function formatKm2(valueM2) {
  if (valueM2 == null || Number.isNaN(Number(valueM2))) return "—";
  const km2 = Number(valueM2) / 1_000_000;
  if (km2 >= 100) return `${km2.toFixed(0)} km²`;
  if (km2 >= 10) return `${km2.toFixed(1)} km²`;
  return `${km2.toFixed(2)} km²`;
}

function renderInsights(out) {
  const gap = Number(out.meta?.gap_years ?? 0);
  const horizon = Number(out.meta?.horizon_years ?? 0);
  const scaleMax = Math.max(gap, horizon, 0.1);

  setWidth("barGap", (gap / scaleMax) * 100);
  setWidth("barHorizon", (horizon / scaleMax) * 100);
  setText("valGap", `${gap.toFixed(1)}y`);
  setText("valHorizon", `${horizon.toFixed(1)}y`);

  const pct = Number(out.stats?.percent_loss ?? 0);
  const clampedPct = Math.max(0, Math.min(100, pct));
  const donut = document.getElementById("riskDonut");
  if (donut) {
    donut.style.background = `conic-gradient(#ef4444 ${clampedPct}%, #e2e8f0 0)`;
  }

  setText("riskPct", `${clampedPct.toFixed(1)}%`);
  setText("vegArea", formatKm2(out.stats?.veg_area_m2));
  setText("lossArea", formatKm2(out.stats?.loss_area_m2));
}

function applyView() {
  if (!lastOut) return;
  const u = lastOut.urls;
  const isNDVI = (viewMode === "ndvi");

  // First + Latest switch between true/NDVI
  document.getElementById("imgFirst").src  = isNDVI ? (u.first_ndvi  || "") : (u.first_true  || "");
  document.getElementById("imgLatest").src = isNDVI ? (u.latest_ndvi || "") : (u.latest_true || "");

  setText("labelFirst",  isNDVI ? "NDVI" : "True colour");
  setText("labelLatest", isNDVI ? "NDVI" : "True colour");
  setText("toggleText",  isNDVI ? "NDVI" : "True Colour");

  // Overlay panel is always latest true + red overlay
  document.getElementById("imgOverlayBase").src = u.latest_true || "";
  document.getElementById("imgOverlayRisk").src = u.risk_overlay || "";
}

async function run() {
  const region = document.getElementById("region")?.value;
  const t1 = document.getElementById("t1")?.value;
  const t2 = document.getElementById("t2")?.value;

  syncDateTitles();

  if (!t1 || !t2) { setText("status", "Pick both dates."); return; }
  if (t2 <= t1) { setText("status", "Latest date must be after first date."); return; }

  setLoading(true);
  setText("status", "Running…");

  try {
    const out = await postJSON("/api/project", { region, t1, t2, size_px: 1200 });
    lastOut = out;

    const gap = out.meta?.gap_years;
    const horizon = out.meta?.horizon_years;
    if (gap != null && horizon != null) {
      setText("horizon", `~${horizon.toFixed(1)} years (gap=${gap.toFixed(1)})`);
    } else {
      setText("horizon", "—");
    }

    setText("status", "Done ✅");
    applyView();
    renderInsights(out);
  } catch (e) {
    setText("status", `Error: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  syncDateTitles();

  document.getElementById("runBtn")?.addEventListener("click", run);

  const toggle = document.getElementById("viewToggle");
  toggle?.addEventListener("change", () => {
    viewMode = toggle.checked ? "ndvi" : "true";
    applyView();
  });
});
