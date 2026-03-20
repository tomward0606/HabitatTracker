"""
Flask + Google Earth Engine backend for habitat change exploration.

Core responsibilities:
- Build cloud-filtered Sentinel-2 composites for user-selected regions/dates.
- Derive NDVI products and simple rule-based change masks.
- Train lightweight Random Forest models for forecast/risk visual layers.
- Return image thumbnail URLs and summary metrics used by the frontend UI.

Notes on architecture:
- Most heavy computation is deferred to Earth Engine server-side objects.
- .getInfo() is used sparingly to materialize small metric payloads for API responses.
- Endpoints are intentionally pragmatic and visualization-oriented rather than
    publication-grade biophysical modeling pipelines.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import ee
import os
from datetime import datetime, timedelta

app = Flask(__name__)

EE_PROJECT = os.environ.get("EE_PROJECT", "habitat-wip")

# CRS and scale metadata returned to clients for traceability/debugging.
TARGET_CRS = "EPSG:3857"
TARGET_SCALE = 20  # meters per pixel

# Curated preset AOIs. Values are [xmin, ymin, xmax, ymax] in lon/lat.
REGIONS = {
    "amazon_a": [-62.5, -10.5, -61.7, -9.7],
    "amazon_b": [-63.95, -10.9, -63.15, -10.1],
    "amazon_c": [-60.95, -4.3, -60.15, -3.5],
}

def init_ee():
    """Initialize Earth Engine for this process (supports service account on Render)."""
    import os
    try:
        sa_email = os.environ.get("EE_SERVICE_ACCOUNT", "").strip()
        sa_key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

        if sa_email and sa_key_path:
            # Service-account auth (recommended for hosting)
            creds = ee.ServiceAccountCredentials(sa_email, sa_key_path)
            ee.Initialize(credentials=creds, project=EE_PROJECT)
        else:
            # Fallback: Application Default Credentials or local OAuth (dev)
            ee.Initialize(project=EE_PROJECT)

    except Exception as e:
        raise RuntimeError(
            "Earth Engine not initialized.\n"
            "If running on Render, set EE_SERVICE_ACCOUNT and GOOGLE_APPLICATION_CREDENTIALS.\n"
            f"Details: {e}"
        )

def mask_s2_clouds_scl(img):
    """
    Keep likely clear-sky/land pixels based on Sentinel-2 Scene Classification Layer.

    Kept classes:
    - 4: Vegetation
    - 5: Not-vegetated
    - 6: Water
    - 7: Unclassified
    """
    scl = img.select("SCL")
    keep = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    return img.updateMask(keep)

def pick_s2_composite(region_geom, date_str, window_days=60, max_cloud=60, limit_n=40):
    """
    Build a median Sentinel-2 composite centered on a target date.

    Strategy:
    1) Search a symmetric time window around date_str.
    2) Filter by tile-level cloud metadata and per-pixel SCL mask.
    3) Keep only B2/B3/B4/B8 bands used by this app.
    4) Sort by CLOUDY_PIXEL_PERCENTAGE and cap collection size for stability.
    5) Return clipped median composite.

    Raises:
        RuntimeError when no scenes are found for the selected constraints.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region_geom)
           .filterDate(start, end)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
           .map(mask_s2_clouds_scl)
           .select(["B2", "B3", "B4", "B8"]))

    col = col.sort("CLOUDY_PIXEL_PERCENTAGE").limit(limit_n)

    count = int(col.size().getInfo())
    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes found near {date_str}. "
            f"Try a larger date window or higher cloud threshold. "
            f"(window_days={window_days}, max_cloud={max_cloud})"
        )

    return col.median().clip(region_geom)

def image_to_truecolor_thumb_url(img, region_geom, size_px=1200):
    """Render a true-colour PNG thumbnail URL (B4/B3/B2) for display in the UI."""
    vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.1}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return img.visualize(**vis).getThumbURL(params)

def compute_ndvi(img):
    # NDVI = (NIR - Red) / (NIR + Red); Sentinel-2 NIR=B8, Red=B4
    return img.normalizedDifference(["B8", "B4"]).rename("NDVI")

def image_to_ndvi_thumb_url(ndvi_img, region_geom, size_px=1200):
    """Render an NDVI thumbnail URL using a brown→green vegetation palette."""
    vis = {"min": -0.2, "max": 0.8, "palette": ["7a3b00", "e6c35c", "9bd16f", "1a9850"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return ndvi_img.visualize(**vis).getThumbURL(params)

def image_to_ndvi_diff_thumb_url(diff_img, region_geom, size_px=1200):
    """Render NDVI change thumbnail URL where red=decline and green=increase."""
    vis = {"min": -0.5, "max": 0.5, "palette": ["d73027", "f7f7f7", "1a9850"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return diff_img.visualize(**vis).getThumbURL(params)

def image_to_loss_mask_thumb_url(loss_mask_img, region_geom, size_px=1200):
    # 1 = loss (red), 0 = black background
    vis = {"min": 0, "max": 1, "palette": ["000000", "ff2d2d"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return loss_mask_img.visualize(**vis).getThumbURL(params)

def build_forecast_features(ndvi_t0, ndvi_t1, ndvi_t2):
    """
    Returns an EE Image with feature bands used for forecasting NDVI at T3.
    """
    d01 = ndvi_t1.subtract(ndvi_t0).rename("d01")
    d12 = ndvi_t2.subtract(ndvi_t1).rename("d12")
    slope = ndvi_t2.subtract(ndvi_t0).divide(2).rename("slope")
    volatility = d01.abs().add(d12.abs()).rename("volatility")

    features = (ndvi_t0.rename("ndvi_t0")
                .addBands(ndvi_t1.rename("ndvi_t1"))
                .addBands(ndvi_t2.rename("ndvi_t2"))
                .addBands(d01)
                .addBands(d12)
                .addBands(slope)
                .addBands(volatility))
    return features

def train_rf_regressor(region_geom, features_img, target_ndvi_img,
                       scale=60, n_points=2000, n_trees=100, train_frac=0.7, seed=42):
    """
    Train Random Forest regression model to predict target_ndvi_img (NDVI at T3).
    Returns: (model, metrics_dict)
    """
    # Build a tabular training set by sampling feature/target pixels.
    sample_img = features_img.addBands(target_ndvi_img.rename("target"))

    samples = sample_img.sample(
        region=region_geom,
        scale=scale,
        numPixels=int(n_points),
        seed=seed,
        geometries=False
    )

    # Random train/test split for quick quality diagnostics.
    samples = samples.randomColumn("rand", seed)
    train = samples.filter(ee.Filter.lt("rand", train_frac))
    test = samples.filter(ee.Filter.gte("rand", train_frac))

    # Earth Engine RF classifier can operate in regression mode for continuous NDVI.
    reg = ee.Classifier.smileRandomForest(numberOfTrees=int(n_trees)).setOutputMode("REGRESSION")

    reg = reg.train(
        features=train,
        classProperty="target",
        inputProperties=features_img.bandNames()
    )

    test_pred = test.classify(reg)

    # Compute absolute/squared error on held-out samples.
    def add_err(feat):
        pred = ee.Number(feat.get("classification"))
        true = ee.Number(feat.get("target"))
        err = pred.subtract(true)
        return feat.set({
            "abs_err": err.abs(),
            "sq_err": err.pow(2)
        })

    test_err = test_pred.map(add_err)

    mae = ee.Number(test_err.reduceColumns(ee.Reducer.mean(), ["abs_err"]).get("mean"))
    mse = ee.Number(test_err.reduceColumns(ee.Reducer.mean(), ["sq_err"]).get("mean"))
    rmse = mse.sqrt()

    metrics = ee.Dictionary({
        "train_points": train.size(),
        "test_points": test.size(),
        "mae": mae,
        "rmse": rmse
    })

    return reg, metrics

def image_to_ndvi_pred_thumb_url(ndvi_img, region_geom, size_px=1000):
    """Render predicted NDVI using the same visualization scale as observed NDVI."""
    vis = {"min": -0.2, "max": 0.8, "palette": ["7a3b00", "e6c35c", "9bd16f", "1a9850"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return ndvi_img.visualize(**vis).getThumbURL(params)

def image_to_pred_change_thumb_url(diff_img, region_geom, size_px=1000):
    """Render predicted NDVI delta with diverging palette for gain/loss interpretation."""
    vis = {"min": -0.5, "max": 0.5, "palette": ["d73027", "f7f7f7", "1a9850"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return diff_img.visualize(**vis).getThumbURL(params)

def image_to_binary_mask_thumb_url(mask_img, region_geom, size_px=1000):
    """Render a binary mask URL (primarily used for debugging/diagnostic products)."""
    vis = {"min": 0, "max": 1, "palette": ["000000", "ff2d2d"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return mask_img.visualize(**vis).getThumbURL(params)

def image_to_risk_overlay_thumb_url(mask_img, region_geom, size_px=1000):
    """
    Transparent PNG overlay: semi-transparent red where risk==1, transparent elsewhere.

    Smoothing pipeline:
    1. focal_mean (radius 2) → converts binary mask to 0–1 neighbourhood density.
    2. Threshold at 0.35 → retains areas where most neighbours agree on risk.
    3. connectedPixelCount filter → removes isolated speckle < 15 px.
    4. focal_mode pass → softens remaining jagged edges into cleaner blobs.

    This produces a visually smooth, professional overlay that is easy to
    interpret in an academic context.
    """
    risk = mask_img.unmask(0).rename("risk").uint8()

    # Focal mean converts the binary mask into a neighbourhood density (0–1).
    # This acts as a Gaussian-blur equivalent available in Earth Engine.
    smoothed = risk.focal_mean(radius=2, units="pixels")

    # Keep only pixels where >35 % of the 5×5 neighbourhood is marked as risk.
    # This naturally removes isolated single-pixel noise.
    risk_clean = smoothed.gt(0.35).uint8().selfMask()

    # Drop any remaining tiny connected components (speckle suppression).
    cc = risk_clean.connectedPixelCount(maxSize=100, eightConnected=True)
    risk_no_speck = risk_clean.updateMask(cc.gte(15))

    # Final focal_mode pass: softens jagged block boundaries for clean blob edges.
    risk_smooth = risk_no_speck.focal_mode(radius=1, units="pixels")

    # Semi-transparent red (opacity 0.55) so the base imagery remains readable.
    vis = {"min": 0, "max": 1, "palette": ["ff4444"], "opacity": 0.55}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return risk_smooth.selfMask().visualize(**vis).getThumbURL(params)


def train_rf_regression_2time(
    region_geom, ndvi1, ndvi2,
    t_veg=0.3, t_drop=0.1,
    scale=60, n_points=2000,
    n_trees=100, train_frac=0.8, seed=42,
):
    """
    Train a Random Forest regression model on 2-date NDVI data to predict
    future vegetation state.

    Feature engineering (4 bands):
      ndvi_old        — NDVI at T1, the baseline observation
      ndvi_new        — NDVI at T2, the latest observation
      ndvi_diff       — T2 − T1: direction and magnitude of recent change
      ndvi_volatility — |T2 − T1|: absolute change as a stability metric

    Regression target:
      Linear trend extrapolation: ndvi_new + ndvi_diff
      (projects the observed T1→T2 trend forward by one equal time period,
       giving a plausible short-horizon future NDVI estimate)

    Train / test split: configurable (default 80 / 20).

    Risk criteria (applied post-prediction):
      - Current NDVI > t_veg  →  pixel is actually vegetated
      - Predicted ΔNDVI < −t_drop  →  significant predicted decline

    Feature importance proxy:
      Absolute Pearson correlation between each feature and the target,
      computed on the full sample set.  Provides an interpretable univariate
      relevance score suitable for academic reporting.

    Returns
    -------
    ndvi_pred   : ee.Image — predicted future NDVI raster
    risk_mask   : ee.Image — binary risk mask (vegetated + declining)
    metrics     : ee.Dictionary — train_n, test_n, mae, rmse,
                  feature_importance (dict of |r| values)
    """
    # ── Feature engineering ──────────────────────────────────────────────────
    ndvi_diff       = ndvi2.subtract(ndvi1).rename("ndvi_diff")
    ndvi_volatility = ndvi_diff.abs().rename("ndvi_volatility")

    features_img = (
        ndvi1.rename("ndvi_old")
             .addBands(ndvi2.rename("ndvi_new"))
             .addBands(ndvi_diff)
             .addBands(ndvi_volatility)
    )
    feature_names = ["ndvi_old", "ndvi_new", "ndvi_diff", "ndvi_volatility"]

    # Regression target: linear trend extrapolation (best available proxy for
    # future state given only two observed time-steps).
    target_img = ndvi2.add(ndvi_diff).rename("target")

    # ── Sampling ─────────────────────────────────────────────────────────────
    sample_img = features_img.addBands(target_img)
    samples = sample_img.sample(
        region=region_geom,
        scale=scale,
        numPixels=int(n_points),
        seed=seed,
        geometries=False,
    )

    # ── 80 / 20 train / test split ───────────────────────────────────────────
    samples     = samples.randomColumn("rand", seed)
    train_col   = samples.filter(ee.Filter.lt("rand", train_frac))
    test_col    = samples.filter(ee.Filter.gte("rand", train_frac))

    # ── Train RF regressor ───────────────────────────────────────────────────
    reg = (
        ee.Classifier.smileRandomForest(numberOfTrees=int(n_trees))
          .setOutputMode("REGRESSION")
          .train(
              features=train_col,
              classProperty="target",
              inputProperties=feature_names,
          )
    )

    # ── Evaluation on held-out test set ──────────────────────────────────────
    test_pred = test_col.classify(reg)

    def add_errors(feat):
        pred  = ee.Number(feat.get("classification"))
        true_ = ee.Number(feat.get("target"))
        err   = pred.subtract(true_)
        return feat.set({"abs_err": err.abs(), "sq_err": err.pow(2)})

    test_err = test_pred.map(add_errors)

    mae  = ee.Number(
        test_err.reduceColumns(ee.Reducer.mean(), ["abs_err"]).get("mean")
    )
    rmse = ee.Number(
        test_err.reduceColumns(ee.Reducer.mean(), ["sq_err"]).get("mean")
    ).sqrt()

    # ── Feature importance proxy (absolute Pearson r with target) ────────────
    # All four expressions are computed lazily and resolved in one getInfo call.
    def pearson_abs(fname):
        corr = samples.reduceColumns(
            reducer=ee.Reducer.pearsonsCorrelation(),
            selectors=[fname, "target"],
        ).get("correlation")
        return ee.Number(corr).abs()

    feature_importance = ee.Dictionary(
        {name: pearson_abs(name) for name in feature_names}
    )

    metrics = ee.Dictionary({
        "train_n":            train_col.size(),
        "test_n":             test_col.size(),
        "mae":                mae,
        "rmse":               rmse,
        "feature_importance": feature_importance,
    })

    # ── Apply model to predict future NDVI ───────────────────────────────────
    ndvi_pred = features_img.classify(reg).rename("ndvi_pred")

    # Risk mask: currently vegetated AND predicted to decline significantly.
    # Both thresholds are configurable to support sensitivity analysis.
    risk_mask = (
        ndvi2.gt(t_veg)
             .And(ndvi_pred.subtract(ndvi2).lt(-t_drop))
             .rename("risk_mask")
    )

    return ndvi_pred, risk_mask, metrics

def train_rf_on_pseudolabels(region_geom, ndvi_before, ndvi_after, ndvi_diff,
                            t_veg, t_drop,
                            scale=60,
                            points_per_class=600,
                            n_trees=80,
                            train_frac=0.7,
                            seed=42):
    """
    Trains RF on 3-band NDVI features with pseudo-labels from NDVI rules.
    Returns: (rf_pred_image, metrics_dict)
    """

    # Feature stack uses pre, post, and change NDVI bands.
    features = (ndvi_before.rename("ndvi_b")
                .addBands(ndvi_after.rename("ndvi_a"))
                .addBands(ndvi_diff.rename("ndvi_d")))

    # Pseudo-label rule: currently vegetated + significant NDVI drop => "loss".
    veg_mask = ndvi_before.gt(t_veg)
    loss_mask = veg_mask.And(ndvi_diff.lt(-t_drop))
    label_img = loss_mask.rename("label").uint8()

    sample_img = features.addBands(label_img)

    samples = sample_img.stratifiedSample(
        numPoints=int(points_per_class),
        classBand="label",
        region=region_geom,
        scale=scale,
        seed=seed,
        geometries=False
    )

    # Split to estimate model quality (accuracy/confusion matrix) quickly.
    samples = samples.randomColumn("rand", seed)
    train = samples.filter(ee.Filter.lt("rand", train_frac))
    test = samples.filter(ee.Filter.gte("rand", train_frac))

    classifier = ee.Classifier.smileRandomForest(numberOfTrees=int(n_trees)).train(
        features=train,
        classProperty="label",
        inputProperties=["ndvi_b", "ndvi_a", "ndvi_d"]
    )

    rf_pred = features.classify(classifier).rename("rf_pred").uint8()

    test_classified = test.classify(classifier)
    cm = test_classified.errorMatrix("label", "classification")
    accuracy = cm.accuracy()

    metrics = ee.Dictionary({
        "points_total": samples.size(),
        "points_train": train.size(),
        "points_test": test.size(),
        "confusion_matrix": cm.array(),
        "accuracy": accuracy
    })

    return rf_pred, metrics

def image_to_rf_pred_thumb_url(rf_pred_img, region_geom, size_px=1200):
    """Render RF binary prediction where class 1 is emphasized in red."""
    vis = {"min": 0, "max": 1, "palette": ["111827", "ff2d2d"]}
    params = {"region": region_geom, "dimensions": int(size_px), "format": "png"}
    return rf_pred_img.visualize(**vis).getThumbURL(params)

def compute_loss_stats(region_geom, veg_mask, loss_mask, scale):
    """
    Compute area-based summary statistics for vegetation and projected/observed loss.

    Returns Earth Engine Number objects in m² and percent for downstream serialization.
    """
    area = ee.Image.pixelArea().rename("area")

    veg_area = area.updateMask(veg_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region_geom,
        scale=scale,
        maxPixels=1e9
    ).get("area")

    loss_area = area.updateMask(loss_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region_geom,
        scale=scale,
        maxPixels=1e9
    ).get("area")

    veg_area = ee.Number(veg_area)
    loss_area = ee.Number(loss_area)

    percent_loss = ee.Algorithms.If(
        veg_area.gt(0),
        loss_area.divide(veg_area).multiply(100),
        0
    )

    return {
        "veg_area_m2": veg_area,
        "loss_area_m2": loss_area,
        "percent_loss": ee.Number(percent_loss),
    }

@app.route("/")
def root():
    """Redirect default route to the main Explore page."""
    return redirect(url_for("explore"))

@app.route("/explore")
def explore():
    """Render map-driven custom ROI prediction page."""
    return render_template("explore.html")

@app.route("/example")
def example_page():
    """Render preset-region example workflow page."""
    return render_template("example.html", regions=REGIONS)

@app.route("/methodology")
def methodology():
    """Render narrative explanation of data, modeling, and caveats."""
    return render_template("methodology.html")

@app.route("/predict")
def predict_page():
    """Backwards-compatible alias to the Explore page."""
    return redirect(url_for("explore"))

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    Baseline analysis endpoint.

    Produces observed true-colour/NDVI assets, rule-based loss mask, optional
    pseudolabel RF output, and summary statistics for the selected date pair.
    """
    init_ee()
    data = request.get_json(force=True)

    region_key = (data.get("region") or "").strip()
    start_date = (data.get("start_date") or "").strip()
    end_date = (data.get("end_date") or "").strip()

    window_days = int(data.get("window_days") or 60)
    max_cloud = int(data.get("max_cloud") or 30)
    size_px = int(data.get("size_px") or 800)

    # Rule thresholds for vegetation presence and decline magnitude.
    t_veg = float(data.get("t_veg") or 0.5)
    t_drop = float(data.get("t_drop") or 0.2)

    if region_key not in REGIONS:
        return jsonify({"ok": False, "error": "Unknown region"}), 400
    if not start_date or not end_date:
        return jsonify({"ok": False, "error": "Missing start_date or end_date"}), 400
    if end_date <= start_date:
        return jsonify({"ok": False, "error": "end_date must be after start_date"}), 400

    region_geom = ee.Geometry.Rectangle(REGIONS[region_key])

    try:
        img_start = pick_s2_composite(region_geom, start_date, window_days, max_cloud)
        img_end = pick_s2_composite(region_geom, end_date, window_days, max_cloud)

        # Context layers used directly in UI side-by-side comparison.
        true_start_url = image_to_truecolor_thumb_url(img_start, region_geom, size_px)
        true_end_url = image_to_truecolor_thumb_url(img_end, region_geom, size_px)

        # NDVI products for ecological signal and temporal change.
        ndvi_start = compute_ndvi(img_start)
        ndvi_end = compute_ndvi(img_end)
        ndvi_diff = ndvi_end.subtract(ndvi_start).rename("NDVI_DIFF")

        veg_mask = ndvi_start.gt(t_veg)
        loss_mask = veg_mask.And(ndvi_diff.lt(-t_drop))

        ndvi_start_url = image_to_ndvi_thumb_url(ndvi_start, region_geom, size_px)
        ndvi_end_url = image_to_ndvi_thumb_url(ndvi_end, region_geom, size_px)
        ndvi_diff_url = image_to_ndvi_diff_thumb_url(ndvi_diff, region_geom, size_px)
        loss_mask_url = image_to_loss_mask_thumb_url(loss_mask, region_geom, size_px)

        # Optional ML branch (enabled by default) trained on pseudo-labels.
        do_ml = bool(data.get("do_ml") if data.get("do_ml") is not None else True)
        points_per_class = int(data.get("points_per_class") or 600)
        n_trees = int(data.get("n_trees") or 80)

        rf_pred_url = None
        ml_metrics_py = None

        if do_ml:
            rf_pred_img, ml_metrics = train_rf_on_pseudolabels(
                region_geom=region_geom,
                ndvi_before=ndvi_start,
                ndvi_after=ndvi_end,
                ndvi_diff=ndvi_diff,
                t_veg=t_veg,
                t_drop=t_drop,
                scale=60,
                points_per_class=points_per_class,
                n_trees=n_trees,
                train_frac=0.7,
                seed=42
            )

            rf_pred_url = image_to_rf_pred_thumb_url(rf_pred_img, region_geom, size_px=size_px)
            ml_metrics_py = ml_metrics.getInfo()

        stats = compute_loss_stats(region_geom, veg_mask, loss_mask, scale=60)
        stats_py = ee.Dictionary(stats).getInfo()

        return jsonify({
            "ok": True,
            "urls": {
                "true_start": true_start_url,
                "true_end": true_end_url,
                "ndvi_start": ndvi_start_url,
                "ndvi_end": ndvi_end_url,
                "ndvi_diff": ndvi_diff_url,
                "loss_mask": loss_mask_url
            },
            "params": {"t_veg": t_veg, "t_drop": t_drop},
            "stats": stats_py,
            "ml": {
                "enabled": do_ml,
                "rf_pred_url": rf_pred_url,
                "metrics": ml_metrics_py,
                "params": {"points_per_class": points_per_class, "n_trees": n_trees}
            },
            "meta": {
                "region": region_key,
                "window_days": window_days,
                "max_cloud": max_cloud,
                "size_px": size_px,
                "target_crs": TARGET_CRS,
                "target_scale": TARGET_SCALE
            }
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": (
                "Analysis failed. Try a larger window (90–120 days), relax max_cloud, or change dates. "
                f"Details: {e}"
            )
        }), 500

@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    """
    Four-date NDVI forecast endpoint.

    Trains a Random Forest regressor on historical NDVI bands/features and predicts
    NDVI at a future date, then derives projected loss masks from that forecast.
    """
    init_ee()
    data = request.get_json(force=True)

    region_key = (data.get("region") or "").strip()

    t0 = (data.get("t0") or "").strip()
    t1 = (data.get("t1") or "").strip()
    t2 = (data.get("t2") or "").strip()
    t3 = (data.get("t3") or "").strip()

    # Operational defaults chosen for robust sample retrieval across many regions.
    window_days = 60
    max_cloud = 30
    size_px = int(data.get("size_px") or 1000)

    n_points = int(data.get("n_points") or 2000)
    n_trees = int(data.get("n_trees") or 100)
    sample_scale = int(data.get("sample_scale") or 60)

    t_veg = float(data.get("t_veg") or 0.5)
    t_drop = float(data.get("t_drop") or 0.2)

    if region_key not in REGIONS:
        return jsonify({"ok": False, "error": "Unknown region"}), 400
    if not (t0 and t1 and t2 and t3):
        return jsonify({"ok": False, "error": "Missing one or more dates (t0,t1,t2,t3)."}), 400

    region_geom = ee.Geometry.Rectangle(REGIONS[region_key])

    try:
        img0 = pick_s2_composite(region_geom, t0, window_days, max_cloud)
        img1 = pick_s2_composite(region_geom, t1, window_days, max_cloud)
        img2 = pick_s2_composite(region_geom, t2, window_days, max_cloud)
        img3 = pick_s2_composite(region_geom, t3, window_days, max_cloud)

        ndvi0 = compute_ndvi(img0)
        ndvi1 = compute_ndvi(img1)
        ndvi2 = compute_ndvi(img2)
        ndvi3 = compute_ndvi(img3)

        # Feature engineering captures level, trend, and volatility signals.
        features = build_forecast_features(ndvi0, ndvi1, ndvi2)

        model, metrics = train_rf_regressor(
            region_geom=region_geom,
            features_img=features,
            target_ndvi_img=ndvi3,
            scale=sample_scale,
            n_points=n_points,
            n_trees=n_trees,
            train_frac=0.7,
            seed=42
        )

        ndvi3_pred = features.classify(model).rename("ndvi_t3_pred")

        pred_change_23 = ndvi3_pred.subtract(ndvi2).rename("pred_change_23")
        pred_veg_mask = ndvi2.gt(t_veg)
        pred_loss_mask = pred_veg_mask.And(pred_change_23.lt(-t_drop)).rename("pred_loss_mask")

        true_t1_url = image_to_truecolor_thumb_url(img1, region_geom, size_px=size_px)
        true_t2_url = image_to_truecolor_thumb_url(img2, region_geom, size_px=size_px)
        ndvi_t1_url = image_to_ndvi_thumb_url(ndvi1, region_geom, size_px=size_px)
        ndvi_t2_url = image_to_ndvi_thumb_url(ndvi2, region_geom, size_px=size_px)

        url_pred_ndvi3 = image_to_ndvi_pred_thumb_url(ndvi3_pred, region_geom, size_px=size_px)
        url_pred_loss = image_to_binary_mask_thumb_url(pred_loss_mask, region_geom, size_px=size_px)

        metrics_py = metrics.getInfo()

        return jsonify({
            "ok": True,
            "urls": {
                "true_t1": true_t1_url,
                "true_t2": true_t2_url,
                "ndvi_t1": ndvi_t1_url,
                "ndvi_t2": ndvi_t2_url,
                "ndvi_t3_pred": url_pred_ndvi3,
                "pred_loss_mask": url_pred_loss
            },
            "metrics": metrics_py,
            "params": {
                "window_days": window_days,
                "max_cloud": max_cloud,
                "n_points": n_points,
                "n_trees": n_trees,
                "sample_scale": sample_scale,
                "t_veg": t_veg,
                "t_drop": t_drop
            }
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Forecast failed. Try smaller region, fewer points, or larger window. Details: {e}"
        }), 500

@app.route("/api/project", methods=["POST"])
def api_project():
    """
    UI-facing projection endpoint used by the Explore page.

    Input: region + two dates.
    Output: first/latest true colour and NDVI, risk overlay, summary stats, and
    optional RF pseudolabel metrics.
    """
    init_ee()
    data = request.get_json(force=True)

    region_key = (data.get("region") or "").strip()
    t1 = (data.get("t1") or "").strip()
    t2 = (data.get("t2") or "").strip()

    # Keep endpoint simple with fixed defaults; advanced controls can be exposed later.
    window_days = 60
    max_cloud = 30
    size_px = int(data.get("size_px") or 1000)

    # Baseline thresholds used in both rule and pseudo-label generation.
    t_veg = 0.5
    t_drop = 0.2

    if region_key not in REGIONS:
        return jsonify({"ok": False, "error": "Unknown region"}), 400
    if not t1 or not t2:
        return jsonify({"ok": False, "error": "Missing t1 or t2"}), 400
    if t2 <= t1:
        return jsonify({"ok": False, "error": "t2 must be after t1"}), 400

    region_geom = ee.Geometry.Rectangle(REGIONS[region_key])

    try:
        img1 = pick_s2_composite(region_geom, t1, window_days, max_cloud)
        img2 = pick_s2_composite(region_geom, t2, window_days, max_cloud)

        ndvi1 = compute_ndvi(img1)
        ndvi2 = compute_ndvi(img2)
        ndvi_diff = ndvi2.subtract(ndvi1).rename("ndvi_diff")

        # Projection horizon multiplier relative to observed T1->T2 gap.
        factor = float(data.get("factor") or 1.0)
        do_ml = bool(data.get("do_ml") if data.get("do_ml") is not None else True)

        # Fixed ML settings (UI stays clean)
        points_per_class = 600
        n_trees = 80

        # Linear trend projection based on recent observed NDVI difference.
        projected_change = ndvi_diff.multiply(factor).rename("projected_change")
        ndvi_future = ndvi2.add(projected_change).rename("ndvi_future")

        veg_now = ndvi2.gt(t_veg)
        loss_risk = veg_now.And(projected_change.lt(-t_drop)).rename("loss_risk")

        # Overlay source can be RF-driven (default) or pure rule-based fallback.
        if do_ml:
            rf_pred_img, ml_metrics = train_rf_on_pseudolabels(
                region_geom=region_geom,
                ndvi_before=ndvi1,
                ndvi_after=ndvi2,
                ndvi_diff=ndvi_diff,
                t_veg=t_veg,
                t_drop=t_drop,
                scale=60,
                points_per_class=points_per_class,
                n_trees=n_trees,
                train_frac=0.7,
                seed=42
            )

            # Constrain predictions to currently vegetated pixels for cleaner maps.
            rf_loss_mask = ndvi2.gt(t_veg).And(rf_pred_img.eq(1)).rename("rf_loss_mask")

            overlay_mask = rf_loss_mask
            ml_metrics_py = ml_metrics.getInfo()
        else:
            overlay_mask = loss_risk
            ml_metrics_py = None

        dt1 = datetime.strptime(t1, "%Y-%m-%d")
        dt2 = datetime.strptime(t2, "%Y-%m-%d")
        gap_years = max((dt2 - dt1).days / 365.25, 0.0)
        horizon_years = gap_years * factor

        # Frontend contract: fixed URL keys consumed by static/app.js.
        urls = {
            "first_true": image_to_truecolor_thumb_url(img1, region_geom, size_px),
            "latest_true": image_to_truecolor_thumb_url(img2, region_geom, size_px),
            "first_ndvi": image_to_ndvi_thumb_url(ndvi1, region_geom, size_px),
            "latest_ndvi": image_to_ndvi_thumb_url(ndvi2, region_geom, size_px),
            "risk_overlay": image_to_risk_overlay_thumb_url(overlay_mask, region_geom, size_px),
        }

        stats = compute_loss_stats(region_geom, veg_now, overlay_mask, scale=60)
        stats_py = ee.Dictionary(stats).getInfo()

        return jsonify({
            "ok": True,
            "urls": urls,
            "stats": stats_py,
            "meta": {
                "gap_years": gap_years,
                "horizon_years": horizon_years
            },
            "ml": {
                "enabled": do_ml,
                "type": "rf_pseudolabel_classifier",
                "metrics": ml_metrics_py,
                "params": {
                    "points_per_class": points_per_class,
                    "n_trees": n_trees
                }
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "error": f"Projection failed: {e}"}), 500

@app.route("/api/project_bbox", methods=["POST"])
def api_project_bbox():
    """
    Prediction endpoint that accepts a user-drawn ROI bbox.

    ML pipeline (when do_ml=True):
    1. Build NDVI composites for the two selected dates.
    2. Engineer 4 features: ndvi_old, ndvi_new, ndvi_diff, ndvi_volatility.
    3. Create a regression target via linear-trend extrapolation.
    4. Sample 2000 pixels and split 80/20 for train/test.
    5. Train a Random Forest regressor (100 trees).
    6. Evaluate with RMSE and MAE on the held-out test set.
    7. Compute per-feature Pearson |r| as an importance proxy.
    8. Apply risk mask: vegetated pixels with a significant predicted drop.
    9. Return smoothed overlay + metrics to the frontend.
    """
    init_ee()
    data = request.get_json(force=True)

    t1 = (data.get("t1") or "").strip()
    t2 = (data.get("t2") or "").strip()

    bbox = data.get("bbox") or {}
    try:
        min_lon = float(bbox["min_lon"])
        min_lat = float(bbox["min_lat"])
        max_lon = float(bbox["max_lon"])
        max_lat = float(bbox["max_lat"])
    except Exception:
        return jsonify({"ok": False, "error": "Invalid bbox. Expected min_lon/min_lat/max_lon/max_lat."}), 400

    if max_lon <= min_lon or max_lat <= min_lat:
        return jsonify({"ok": False, "error": "Invalid bbox ordering."}), 400
    if not (
        -180 <= min_lon <= 180 and -180 <= max_lon <= 180 and
        -90  <= min_lat <= 90  and -90  <= max_lat <= 90
    ):
        return jsonify({"ok": False, "error": "BBox out of range."}), 400

    if not t1 or not t2:
        return jsonify({"ok": False, "error": "Missing t1 or t2"}), 400
    if t2 <= t1:
        return jsonify({"ok": False, "error": "t2 must be after t1"}), 400

    do_ml      = bool(data.get("do_ml", True))
    window_days = 60
    max_cloud   = 30
    size_px     = int(data.get("size_px") or 1200)

    # Configurable risk thresholds (sensible academic defaults).
    t_veg  = float(data.get("t_veg",  0.3))   # minimum NDVI to count as vegetated
    t_drop = float(data.get("t_drop", 0.1))   # minimum predicted drop to flag as risk

    region_geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    try:
        # ── Data acquisition ─────────────────────────────────────────────────
        img1  = pick_s2_composite(region_geom, t1, window_days, max_cloud)
        img2  = pick_s2_composite(region_geom, t2, window_days, max_cloud)
        ndvi1 = compute_ndvi(img1)
        ndvi2 = compute_ndvi(img2)

        # Rule-based baseline: used when ML is disabled or as a fallback.
        ndvi_diff_raw = ndvi2.subtract(ndvi1)
        baseline_loss = (ndvi2.gt(t_veg)
                              .And(ndvi_diff_raw.lt(-t_drop))
                              .rename("baseline_loss"))

        ml_block = {"enabled": do_ml, "type": None, "metrics": None, "params": None}

        if do_ml:
            # ── Regression-based ML pipeline ─────────────────────────────────
            ndvi_pred, risk_mask, ml_metrics = train_rf_regression_2time(
                region_geom=region_geom,
                ndvi1=ndvi1,
                ndvi2=ndvi2,
                t_veg=t_veg,
                t_drop=t_drop,
                scale=60,
                n_points=2000,
                n_trees=100,
                train_frac=0.8,
                seed=42,
            )

            overlay_mask           = risk_mask
            ml_block["type"]       = "rf_regression"
            ml_block["params"]     = {"n_points": 2000, "n_trees": 100, "t_veg": t_veg, "t_drop": t_drop}
            # Resolve all EE lazy values in one network round trip.
            ml_block["metrics"]    = ml_metrics.getInfo()
        else:
            overlay_mask = baseline_loss

        urls = {
            "latest_true":  image_to_truecolor_thumb_url(img2, region_geom, size_px),
            "latest_ndvi":  image_to_ndvi_thumb_url(ndvi2, region_geom, size_px),
            "risk_overlay": image_to_risk_overlay_thumb_url(overlay_mask, region_geom, size_px),
        }

        return jsonify({"ok": True, "urls": urls, "ml": ml_block})

    except Exception as e:
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500

if __name__ == "__main__":
    # Disable reloader to avoid repeated EE initialization in development.
    app.run(debug=True, use_reloader=False)
