const STUDY_BOUNDS = {
  min_lon: -66.0,
  min_lat: -12.0,
  max_lon: -58.0,
  max_lat: -2.0,
};

const CELL_SIZE_DEG = {
  lon: 0.5,
  lat: 0.5,
};

const state = {
  selectedBBox: null,
  selectedLayer: null,
  studyLayer: null,
  map: null,
  latestTrueUrl: "",
  latestNdviUrl: "",
  riskOverlayUrl: "",
  ndviAvailable: false,
};

function isInStudyArea(lat, lon) {
  return (
    lon >= STUDY_BOUNDS.min_lon &&
    lon <= STUDY_BOUNDS.max_lon &&
    lat >= STUDY_BOUNDS.min_lat &&
    lat <= STUDY_BOUNDS.max_lat
  );
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function cellCenter(bbox) {
  return {
    lon: (bbox.min_lon + bbox.max_lon) / 2,
    lat: (bbox.min_lat + bbox.max_lat) / 2,
  };
}

function isValidBBox(b) {
  if (!b) return false;
  return (
    Number.isFinite(b.min_lon) && Number.isFinite(b.min_lat) &&
    Number.isFinite(b.max_lon) && Number.isFinite(b.max_lat) &&
    b.max_lon > b.min_lon && b.max_lat > b.min_lat &&
    b.min_lon >= -180 && b.max_lon <= 180 &&
    b.min_lat >= -90 && b.max_lat <= 90
  );
}

function snapToGridCell(lat, lon) {
  const safeLon = clamp(lon, STUDY_BOUNDS.min_lon, STUDY_BOUNDS.max_lon - Number.EPSILON);
  const safeLat = clamp(lat, STUDY_BOUNDS.min_lat, STUDY_BOUNDS.max_lat - Number.EPSILON);

  const x = Math.floor((safeLon - STUDY_BOUNDS.min_lon) / CELL_SIZE_DEG.lon);
  const y = Math.floor((safeLat - STUDY_BOUNDS.min_lat) / CELL_SIZE_DEG.lat);

  const min_lon = STUDY_BOUNDS.min_lon + (x * CELL_SIZE_DEG.lon);
  const min_lat = STUDY_BOUNDS.min_lat + (y * CELL_SIZE_DEG.lat);
  const max_lon = Math.min(min_lon + CELL_SIZE_DEG.lon, STUDY_BOUNDS.max_lon);
  const max_lat = Math.min(min_lat + CELL_SIZE_DEG.lat, STUDY_BOUNDS.max_lat);

  return { min_lon, min_lat, max_lon, max_lat };
}

function setReadyState(isReady, message) {
  const runBtn = document.getElementById("runBtn");
  if (runBtn) runBtn.disabled = !isReady;
  HabitatCommon.setText("status", message || (isReady ? "Cell selected. Ready to run." : "Select a grid cell."));
}

function setRunningState(isRunning, message) {
  HabitatCommon.setLoading(isRunning);
  if (isRunning) {
    const runBtn = document.getElementById("runBtn");
    if (runBtn) runBtn.disabled = true;
    HabitatCommon.setText("status", message || "Running…");
  }
}

function setErrorState(message) {
  HabitatCommon.setLoading(false);
  const canRun = isValidBBox(state.selectedBBox);
  const runBtn = document.getElementById("runBtn");
  if (runBtn) runBtn.disabled = !canRun;
  HabitatCommon.setText("status", message);
}

function updateSelectionUI() {
  if (!isValidBBox(state.selectedBBox)) {
    HabitatCommon.setText("bboxText", "—");
    HabitatCommon.setText("cellText", "Click on the map to select a cell");
    setReadyState(false, "Select a grid cell.");
    return;
  }

  const center = cellCenter(state.selectedBBox);
  HabitatCommon.setText("bboxText", HabitatCommon.formatBBox(state.selectedBBox));
  HabitatCommon.setText(
    "cellText",
    `${HabitatCommon.formatBBox(state.selectedBBox, 3)} | center: ${center.lon.toFixed(3)}, ${center.lat.toFixed(3)}`
  );
  setReadyState(true, "Cell selected. Ready to run.");
}

function drawSelectionOnMap() {
  if (!state.map || !isValidBBox(state.selectedBBox)) return;

  if (state.selectedLayer) {
    state.selectedLayer.remove();
    state.selectedLayer = null;
  }

  state.selectedLayer = L.rectangle(
    [
      [state.selectedBBox.min_lat, state.selectedBBox.min_lon],
      [state.selectedBBox.max_lat, state.selectedBBox.max_lon],
    ],
    { color: "#1d4ed8", weight: 2, fillColor: "#1d4ed8", fillOpacity: 0.10 }
  ).addTo(state.map);
}

function setImageWithFade(imgEl, url) {
  if (!imgEl) return;
  if (!url) {
    imgEl.removeAttribute("src");
    imgEl.classList.remove("is-loaded");
    return;
  }

  imgEl.classList.remove("is-loaded");
  imgEl.onload = () => imgEl.classList.add("is-loaded");
  imgEl.onerror = () => imgEl.classList.remove("is-loaded");
  imgEl.src = url;
}

function applyBaseImageByToggle() {
  const useNdvi = document.getElementById("viewToggle")?.checked;
  const baseEl = document.getElementById("imgOverlayBase");
  const chosen = useNdvi && state.ndviAvailable ? state.latestNdviUrl : state.latestTrueUrl;
  setImageWithFade(baseEl, chosen);
}

function validateDates(t1, t2) {
  if (!t1 || !t2) return "Pick both dates.";
  if (t2 <= t1) return "Latest date must be after first date.";
  return null;
}

function validateBBoxForRun(bbox) {
  if (!isValidBBox(bbox)) return "Select a valid grid cell first.";
  if (
    bbox.min_lon < -180 || bbox.max_lon > 180 ||
    bbox.min_lat < -90 || bbox.max_lat > 90
  ) {
    return "Selected bbox is out of lon/lat range.";
  }
  return null;
}

function parseManualBBox() {
  const minLon = Number(document.getElementById("minLon")?.value);
  const minLat = Number(document.getElementById("minLat")?.value);
  const maxLon = Number(document.getElementById("maxLon")?.value);
  const maxLat = Number(document.getElementById("maxLat")?.value);
  return { min_lon: minLon, min_lat: minLat, max_lon: maxLon, max_lat: maxLat };
}

function applyManualBBox() {
  const manual = parseManualBBox();
  if (!isValidBBox(manual)) {
    setErrorState("Manual bbox is invalid. Check ordering and numeric values.");
    return;
  }

  if (
    manual.min_lon < STUDY_BOUNDS.min_lon || manual.max_lon > STUDY_BOUNDS.max_lon ||
    manual.min_lat < STUDY_BOUNDS.min_lat || manual.max_lat > STUDY_BOUNDS.max_lat
  ) {
    setErrorState("Manual bbox must stay inside the blue study area.");
    return;
  }

  state.selectedBBox = manual;
  updateSelectionUI();
  drawSelectionOnMap();
}

function initMap() {
  state.map = L.map("map", { zoomControl: true }).setView([-7.0, -62.0], 6);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(state.map);

  const studyLatLngBounds = [
    [STUDY_BOUNDS.min_lat, STUDY_BOUNDS.min_lon],
    [STUDY_BOUNDS.max_lat, STUDY_BOUNDS.max_lon],
  ];

  state.studyLayer = L.rectangle(studyLatLngBounds, {
    color: "#2563eb",
    weight: 2,
    fillColor: "#2563eb",
    fillOpacity: 0.12,
  }).addTo(state.map);

  state.map.fitBounds(state.studyLayer.getBounds(), { padding: [12, 12] });

  const LegendControl = L.Control.extend({
    options: { position: "topright" },
    onAdd: function () {
      const div = L.DomUtil.create("div", "leaflet-control-mapnote");
      div.innerHTML = "Blue area = usable study region.<br>Click to snap to a fixed-size cell.";
      return div;
    }
  });
  state.map.addControl(new LegendControl());

  state.map.on("click", (e) => {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;

    if (!isInStudyArea(lat, lon)) {
      setErrorState("Click inside the blue usable area.");
      return;
    }

    state.selectedBBox = snapToGridCell(lat, lon);
    drawSelectionOnMap();
    updateSelectionUI();
  });
}

async function runPrediction() {
  const t1 = document.getElementById("t1")?.value;
  const t2 = document.getElementById("t2")?.value;
  const doML = document.getElementById("mlToggle")?.checked ?? true;

  const dateErr = validateDates(t1, t2);
  if (dateErr) {
    setErrorState(dateErr);
    return;
  }

  const bboxErr = validateBBoxForRun(state.selectedBBox);
  if (bboxErr) {
    setErrorState(bboxErr);
    return;
  }

  setRunningState(true, doML ? "Running… (ML enabled)" : "Running… (baseline)");

  try {
    const out = await HabitatCommon.postJSON("/api/project_bbox", {
      t1,
      t2,
      bbox: state.selectedBBox,
      do_ml: doML,
      size_px: 1200,
    });

    const urls = out.urls || {};
    state.latestTrueUrl = urls.latest_true || "";
    state.latestNdviUrl = urls.latest_ndvi || "";
    state.riskOverlayUrl = urls.risk_overlay || "";
    state.ndviAvailable = Boolean(state.latestNdviUrl);

    const toggle = document.getElementById("viewToggle");
    const toggleText = document.getElementById("toggleText");
    if (toggle && toggleText) {
      toggle.disabled = !state.ndviAvailable;
      toggle.title = state.ndviAvailable ? "Switch between True Colour and NDVI base" : "NDVI base not available for this response";
      if (!state.ndviAvailable) toggle.checked = false;
      toggleText.textContent = toggle.checked ? "NDVI" : "True Colour";
    }

    applyBaseImageByToggle();
    setImageWithFade(document.getElementById("imgOverlayRisk"), state.riskOverlayUrl);

    if (out.ml?.enabled && out.ml?.metrics) {
      const acc = out.ml.metrics.accuracy ?? out.ml.metrics.test_accuracy ?? null;
      HabitatCommon.setText("mlSummary", acc != null ? `ML accuracy: ${(acc * 100).toFixed(1)}%` : "ML ran (metrics available).");
    } else {
      HabitatCommon.setText("mlSummary", out.ml?.enabled ? "ML ran." : "Baseline threshold mask.");
    }

    HabitatCommon.setLoading(false);
    setReadyState(true, "Done ✅");
  } catch (err) {
    setErrorState(`Error: ${err.message}`);
  }
}

function initEvents() {
  document.getElementById("runBtn")?.addEventListener("click", runPrediction);
  document.getElementById("applyBBoxBtn")?.addEventListener("click", applyManualBBox);

  const toggle = document.getElementById("viewToggle");
  const toggleText = document.getElementById("toggleText");
  if (toggle && toggleText) {
    toggle.addEventListener("change", () => {
      toggleText.textContent = toggle.checked ? "NDVI" : "True Colour";
      applyBaseImageByToggle();
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initEvents();
  updateSelectionUI();
  HabitatCommon.setText("mlSummary", "—");
});
