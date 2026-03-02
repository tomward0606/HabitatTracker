let drawnBBox = null;
let drawnLayer = null;

function fmtBBox(b) {
  return HabitatCommon.formatBBox(b);
}

function initMap() {
  const map = L.map("map", { zoomControl: true }).setView([-4.0, -60.0], 5);

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);

  const drawnItems = new L.FeatureGroup();
  map.addLayer(drawnItems);

  const drawControl = new L.Control.Draw({
    draw: {
      polygon: false,
      polyline: false,
      circle: false,
      circlemarker: false,
      marker: false,
      rectangle: {
        shapeOptions: {}
      }
    },
    edit: {
      featureGroup: drawnItems,
      remove: true
    }
  });
  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, function (e) {
    if (drawnLayer) {
      drawnItems.removeLayer(drawnLayer);
    }

    drawnLayer = e.layer;
    drawnItems.addLayer(drawnLayer);

    const b = drawnLayer.getBounds();
    drawnBBox = {
      min_lon: b.getWest(),
      min_lat: b.getSouth(),
      max_lon: b.getEast(),
      max_lat: b.getNorth()
    };

    HabitatCommon.setText("bboxText", fmtBBox(drawnBBox));
    HabitatCommon.setText("status", "ROI selected. Ready to run.");
  });

  map.on(L.Draw.Event.DELETED, function () {
    drawnBBox = null;
    drawnLayer = null;
    HabitatCommon.setText("bboxText", "—");
    HabitatCommon.setText("status", "Draw an ROI rectangle, then run.");
  });
}

async function runPrediction() {
  const t1 = document.getElementById("t1")?.value;
  const t2 = document.getElementById("t2")?.value;
  const doML = document.getElementById("mlToggle")?.checked ?? true;

  if (!drawnBBox) {
    HabitatCommon.setText("status", "Please draw a rectangle ROI on the map first.");
    return;
  }
  if (!t1 || !t2) {
    HabitatCommon.setText("status", "Pick both dates.");
    return;
  }
  if (t2 <= t1) {
    HabitatCommon.setText("status", "Latest date must be after first date.");
    return;
  }

  HabitatCommon.setLoading(true);
  HabitatCommon.setText("status", doML ? "Running… (ML enabled)" : "Running… (baseline)");

  try {
    const out = await HabitatCommon.postJSON("/api/project_bbox", {
      t1,
      t2,
      bbox: drawnBBox,
      do_ml: doML,
      size_px: 1200
    });

    const u = out.urls || {};
    document.getElementById("imgBase").src = u.latest_true || "";
    document.getElementById("imgOverlay").src = u.risk_overlay || "";

    if (out.ml?.enabled && out.ml?.metrics) {
      const acc = out.ml.metrics.accuracy ?? out.ml.metrics.test_accuracy ?? null;
      HabitatCommon.setText("mlSummary", acc != null ? `ML accuracy: ${(acc * 100).toFixed(1)}%` : "ML ran (metrics available).");
    } else {
      HabitatCommon.setText("mlSummary", out.ml?.enabled ? "ML ran." : "Baseline threshold mask.");
    }

    HabitatCommon.setText("status", "Done ✅");
  } catch (e) {
    HabitatCommon.setText("status", `Error: ${e.message}`);
  } finally {
    HabitatCommon.setLoading(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  document.getElementById("runBtn")?.addEventListener("click", runPrediction);
});
