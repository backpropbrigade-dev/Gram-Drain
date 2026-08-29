"""
pipeline_engine.py — GramDrain Full LiDAR Processing Backend

Usage:
    from pipeline_engine import run_pipeline
    run_pipeline(las_path, epsg, village_name, out_dir, log_callback)
"""

import os, gc, heapq, warnings, shutil
warnings.filterwarnings('ignore')

import numpy as np
from scipy.ndimage import (
    median_filter, gaussian_filter, uniform_filter,
    minimum_filter, laplace,
    label as scipy_label, center_of_mass,
    binary_opening, binary_closing,
)

NODATA    = -9999.0
MAX_PTS   = 5_000_000
TMP_DIR   = os.path.join(os.path.dirname(__file__), "temp")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_libs():
    """Lazy-import heavy libs; raises ImportError with friendly message."""
    missing = []
    for lib in ["laspy", "rasterio", "geopandas", "xgboost", "sklearn", "CSF", "networkx", "pyproj"]:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        raise ImportError(
            f"Missing Python packages: {missing}. "
            "Install them with:\n  pip install laspy[lazrs] rasterio geopandas xgboost "
            "scikit-learn cloth-simulation-filter networkx pyproj whitebox"
        )


def _log(cb, msg: str):
    """Send a progress message to the Streamlit callback (or print)."""
    if cb:
        cb(msg)
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC SCALING
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dynamic_params(total_pts, xmin, xmax, ymin, ymax, is_geographic):
    MAX_GRID_AXIS    = 4_000
    MAX_DICT_ENTRIES = 33_000_000
    MAX_RAW_INFLIGHT = 330_000_000

    x_range = float(xmax - xmin)
    y_range = float(ymax - ymin)

    if is_geographic:
        lat_mid   = (float(ymin) + float(ymax)) / 2.0
        x_range_m = x_range * 111_000.0 * np.cos(np.radians(lat_mid))
        y_range_m = y_range * 111_000.0
    else:
        x_range_m = x_range
        y_range_m = y_range

    min_voxel_m = max(x_range_m, y_range_m) / MAX_GRID_AXIS
    CANDIDATES_M = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50,
                    0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00]
    voxel_m = next((v for v in CANDIDATES_M if v >= min_voxel_m), max(CANDIDATES_M))

    if is_geographic:
        lat_mid   = (float(ymin) + float(ymax)) / 2.0
        VOXEL_RES = voxel_m / (111_000.0 * np.cos(np.radians(lat_mid)))
    else:
        VOXEL_RES = voxel_m

    grid_cols = int(x_range / VOXEL_RES) + 2
    grid_rows = int(y_range / VOXEL_RES) + 2
    worst_case_cells = grid_cols * grid_rows
    stride_from_dict = max(1, int(np.ceil(worst_case_cells / MAX_DICT_ENTRIES)))
    stride_from_pts  = max(1, int(np.ceil(total_pts / MAX_RAW_INFLIGHT)))
    STRIDE = min(max(stride_from_dict, stride_from_pts), 50)

    return VOXEL_RES, STRIDE, grid_cols, grid_rows, voxel_m


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — GROUND CLASSIFICATION (chunked, memory-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _save_ground_las(ground_pts, out_path, log):
    import laspy
    header              = laspy.LasHeader(point_format=2, version="1.2")
    header.scales       = np.array([0.001, 0.001, 0.001])
    header.offsets      = ground_pts.min(axis=0)
    glas                = laspy.LasData(header)
    glas.x              = ground_pts[:, 0]
    glas.y              = ground_pts[:, 1]
    glas.z              = ground_pts[:, 2]
    glas.classification = np.full(len(ground_pts), 2, dtype=np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    glas.write(out_path)
    _log(log, f"  ✅ Ground LAS saved → {out_path}")
    del glas


def classify_ground(las_path, epsg, out_path, log=None):
    import laspy, CSF as _CSF

    _log(log, "📌 Step 1/9 — Reading LAS header …")
    with laspy.open(las_path) as f:
        hdr       = f.header
        total_pts = int(hdr.point_count)
        xmin_h    = float(hdr.mins[0]);  xmax_h = float(hdr.maxs[0])
        ymin_h    = float(hdr.mins[1]);  ymax_h = float(hdr.maxs[1])

    coord_range   = max(xmax_h - xmin_h, ymax_h - ymin_h)
    is_geographic = coord_range < 2.0
    _log(log, f"  Points: {total_pts:,}  |  Coord type: {'GEOGRAPHIC' if is_geographic else 'PROJECTED'}")

    VOXEL_RES, STRIDE, grid_cols, grid_rows, voxel_m = _compute_dynamic_params(
        total_pts, xmin_h, xmax_h, ymin_h, ymax_h, is_geographic)
    _log(log, f"  Voxel: {voxel_m} m  |  Grid: {grid_cols}×{grid_rows}  |  Stride: 1/{STRIDE}")

    INF32 = np.float32(np.inf)
    min_z = np.full((grid_rows, grid_cols), INF32,  dtype=np.float32)
    min_x = np.zeros((grid_rows, grid_cols),        dtype=np.float64)
    min_y = np.zeros((grid_rows, grid_cols),        dtype=np.float64)

    processed = 0
    chunk_num  = 0
    _log(log, "  Accumulating min-z grid …")
    with laspy.open(las_path) as f:
        for chunk in f.chunk_iterator(2_000_000):
            n_chunk = len(chunk.x)
            if STRIDE > 1:
                keep  = np.arange(0, n_chunk, STRIDE)
                x = np.asarray(chunk.x)[keep].astype(np.float64)
                y = np.asarray(chunk.y)[keep].astype(np.float64)
                z = np.asarray(chunk.z)[keep].astype(np.float32)
                del keep
            else:
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float32)

            xi = np.clip(((x - xmin_h) / VOXEL_RES).astype(np.int32), 0, grid_cols - 1)
            yi = np.clip(((y - ymin_h) / VOXEL_RES).astype(np.int32), 0, grid_rows - 1)
            flat  = yi.astype(np.int64) * grid_cols + xi.astype(np.int64)
            order = np.lexsort((z, flat))
            flat_s = flat[order]; x_s = x[order]; y_s = y[order]; z_s = z[order]
            _, first_idx = np.unique(flat_s, return_index=True)
            flat_u = flat_s[first_idx]; x_u = x_s[first_idx]
            y_u    = y_s[first_idx];    z_u = z_s[first_idx]
            del flat_s, x_s, y_s, z_s, order
            ri = (flat_u // grid_cols).astype(np.int32)
            ci = (flat_u %  grid_cols).astype(np.int32)
            del flat_u, flat, xi, yi
            mask    = z_u < min_z[ri, ci]
            min_z[ri[mask], ci[mask]] = z_u[mask]
            min_x[ri[mask], ci[mask]] = x_u[mask]
            min_y[ri[mask], ci[mask]] = y_u[mask]
            processed += n_chunk; chunk_num += 1
            del x, y, z, x_u, y_u, z_u, ri, ci, mask
            gc.collect()
            if chunk_num % 20 == 0:
                filled = int(np.sum(np.isfinite(min_z)))
                _log(log, f"  … {processed:,}/{total_pts:,} pts | {filled:,} cells filled")

    valid_mask = np.isfinite(min_z)
    pts_ds     = np.column_stack([
        min_x[valid_mask], min_y[valid_mask],
        min_z[valid_mask].astype(np.float64)])
    del min_z, min_x, min_y, valid_mask; gc.collect()
    _log(log, f"  Grid pts extracted: {len(pts_ds):,}")

    if len(pts_ds) > MAX_PTS:
        idx    = np.random.choice(len(pts_ds), MAX_PTS, replace=False)
        pts_ds = pts_ds[idx]; del idx; gc.collect()
        _log(log, f"  Safety cap applied: {MAX_PTS:,} pts")

    if is_geographic:
        from pyproj import Transformer
        _log(log, f"  Reprojecting GEOGRAPHIC → EPSG:{epsg} …")
        t             = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        gx, gy        = t.transform(pts_ds[:, 0], pts_ds[:, 1])
        pts_ds[:, 0]  = gx; pts_ds[:, 1] = gy
        del gx, gy; gc.collect()

    _log(log, "  Running CSF ground classification …")
    csf_obj = _CSF.CSF()
    csf_obj.params.bSloopSmooth     = True
    csf_obj.params.cloth_resolution = 0.5
    csf_obj.params.rigidness        = 3
    csf_obj.params.time_step        = 0.65
    csf_obj.params.class_threshold  = 0.5
    csf_obj.params.interations      = 500
    csf_obj.setPointCloud(pts_ds)
    gnd_idx = _CSF.VecInt(); off_idx = _CSF.VecInt()
    csf_obj.do_filtering(gnd_idx, off_idx)
    gnd        = np.array(gnd_idx)
    ground_pts = pts_ds[gnd].copy()
    _log(log, f"  Ground: {len(gnd):,} ({100*len(gnd)/len(pts_ds):.1f}%)")
    del pts_ds, gnd, off_idx, gnd_idx, csf_obj; gc.collect()

    _save_ground_las(ground_pts, out_path, log)
    return ground_pts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DTM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dtm(ground_pts, epsg, res, out_path, tmp_dir, log=None):
    import rasterio
    from rasterio.transform import from_origin
    _log(log, "📌 Step 2/9 — DTM Generation …")
    x, y, z    = ground_pts[:,0], ground_pts[:,1], ground_pts[:,2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    cols = max(int((xmax - xmin) / res) + 1, 1)
    rows = max(int((ymax - ymin) / res) + 1, 1)
    _log(log, f"  Grid: {cols}×{rows} @ {res}m resolution")

    dtm = np.full((rows, cols), np.nan, dtype=np.float32)
    xi  = np.clip(((x - xmin) / res).astype(np.int32), 0, cols - 1)
    yi  = np.clip(((ymax - y) / res).astype(np.int32), 0, rows - 1)
    for i in range(len(x)):
        r, c = yi[i], xi[i]
        if np.isnan(dtm[r, c]) or z[i] < dtm[r, c]:
            dtm[r, c] = z[i]

    dtm_f = np.where(np.isnan(dtm), 0.0, dtm).astype(np.float32)
    wmask = (~np.isnan(dtm)).astype(np.float32)
    denom = np.maximum(uniform_filter(wmask, size=5), 1e-6)
    dtm   = (uniform_filter(dtm_f, size=5) / denom).astype(np.float32)
    dtm   = median_filter(dtm, size=3).astype(np.float32)
    dtm   = gaussian_filter(dtm.astype(np.float64), sigma=1).astype(np.float32)
    dtm[~np.isfinite(dtm)] = NODATA
    del dtm_f, wmask, denom; gc.collect()

    valid = dtm[dtm != NODATA]
    _log(log, f"  Z: {valid.min():.2f} → {valid.max():.2f} m")

    transform = from_origin(xmin, ymax, res, res)
    profile   = {
        'driver': 'GTiff', 'height': rows, 'width': cols,
        'count': 1, 'dtype': 'float32',
        'crs': f'EPSG:{epsg}', 'transform': transform,
        'nodata': NODATA, 'compress': 'lzw',
        'tiled': True, 'blockxsize': 256, 'blockysize': 256, 'predictor': 2
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(dtm, 1)
    _log(log, f"  ✅ DTM COG saved → {out_path}")
    return dtm, transform, profile


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TERRAIN DERIVATIVES (Slope, Flow Accumulation, TWI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope(name, dtm_arr, profile, tmp_dir, log=None):
    _log(log, "📌 Step 3/9 — Computing slope …")
    try:
        import whitebox
        wbt = whitebox.WhiteboxTools(); wbt.verbose = False
        import rasterio
        dtm_wbt    = os.path.join(tmp_dir, f"{name}_dtm_wbt.tif")
        slope_path = os.path.join(tmp_dir, f"{name}_slope.tif")
        valid_mean = float(np.nanmean(dtm_arr[dtm_arr != NODATA]))
        dtm_filled = dtm_arr.copy()
        dtm_filled[dtm_filled == NODATA] = valid_mean
        p = profile.copy(); p['nodata'] = None
        with rasterio.open(dtm_wbt, 'w', **p) as dst:
            dst.write(dtm_filled.astype(np.float32), 1)
        del dtm_filled
        wbt.slope(dtm_wbt, slope_path, zfactor=1.0)
        with rasterio.open(slope_path) as src:
            slope = src.read(1).astype(np.float32)
            nd    = src.nodata
            if nd is not None:
                slope[slope == nd] = np.nan
        os.remove(slope_path)
        os.remove(dtm_wbt)
        _log(log, f"  Slope via WhiteboxTools: {np.nanmin(slope):.2f}° → {np.nanmax(slope):.2f}°")
    except Exception as e:
        _log(log, f"  WhiteboxTools unavailable ({e}) — falling back to numpy gradient slope")
        dy, dx = np.gradient(np.nan_to_num(dtm_arr.copy(), nan=0))
        slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    return slope


def compute_flow_accumulation(dtm_arr, log=None):
    _log(log, "  Computing flow accumulation …")
    dtm_c = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    dtm_f = np.nan_to_num(dtm_c, nan=float(np.nanmax(dtm_c)))
    r5  = dtm_f - minimum_filter(dtm_f, size=5)
    r21 = dtm_f - minimum_filter(dtm_f, size=21)
    r51 = dtm_f - minimum_filter(dtm_f, size=51)
    accum = (0.2 * np.exp(-r5  / (r5.std()  + 1e-6))
           + 0.3 * np.exp(-r21 / (r21.std() + 1e-6))
           + 0.5 * np.exp(-r51 / (r51.std() + 1e-6))).astype(np.float32)
    accum = gaussian_filter(accum, sigma=3).astype(np.float32)
    accum[dtm_arr == NODATA] = 0.0
    del r5, r21, r51, dtm_f; gc.collect()
    return accum


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FEATURE MATRIX + XGBOOST
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(dtm_arr, slope, accum, twi):
    dtm_c = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    dtm_n = np.nan_to_num(dtm_c, nan=0)
    sl_n  = np.nan_to_num(slope,  nan=0)
    ac_n  = np.nan_to_num(accum,  nan=0)
    tw_n  = np.nan_to_num(twi,    nan=0)
    curv  = laplace(dtm_n).astype(np.float32)
    fi    = np.argmin(np.nan_to_num(dtm_c, nan=9999))
    rc, cc = np.unravel_index(fi, dtm_c.shape)
    dist   = np.sqrt(
        (np.arange(dtm_c.shape[0]).reshape(-1, 1) - rc) ** 2 +
        (np.arange(dtm_c.shape[1]).reshape(1, -1) - cc) ** 2
    ).astype(np.float32)
    mean_elev_5  = uniform_filter(dtm_n, size=5).astype(np.float32)
    mean_slope_5 = uniform_filter(sl_n,  size=5).astype(np.float32)
    mean_twi_5   = uniform_filter(tw_n,  size=5).astype(np.float32)
    std_elev_5   = np.sqrt(np.maximum(
        uniform_filter(dtm_n**2, size=5) - mean_elev_5**2, 0
    )).astype(np.float32)
    mean_accum_11 = uniform_filter(ac_n,  size=11).astype(np.float32)
    min_elev_11   = minimum_filter(dtm_n, size=11).astype(np.float32)
    range_elev_11 = (mean_elev_5 - min_elev_11).astype(np.float32)
    feats = np.stack([
        dtm_n, sl_n, ac_n, tw_n, curv, dist,
        mean_elev_5, std_elev_5, mean_slope_5, mean_twi_5,
        mean_accum_11, range_elev_11
    ], axis=-1)
    del dtm_n, sl_n, ac_n, tw_n, curv, dist
    del mean_elev_5, std_elev_5, mean_slope_5, mean_twi_5
    del mean_accum_11, range_elev_11, min_elev_11; gc.collect()
    return feats


def train_and_predict(feats, dtm_arr, log=None):
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    _log(log, "📌 Step 4/9 — Training XGBoost on 12-feature matrix …")

    def norm(a):
        a = np.nan_to_num(a).astype(np.float32)
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    dtm_c = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    slope_arr = feats[:, :, 1]; accum_arr = feats[:, :, 2]; twi_arr = feats[:, :, 3]
    risk_rule = norm(0.40*norm(twi_arr) + 0.35*norm(accum_arr)
                   + 0.15*(1-norm(slope_arr)) + 0.10*(1-norm(dtm_c)))
    rule_hot  = np.zeros_like(risk_rule, dtype=np.uint8)
    rule_hot[risk_rule > 0.92]                           = 2
    rule_hot[(risk_rule >= 0.78) & (risk_rule <= 0.92)] = 1
    rule_hot[dtm_arr == NODATA] = 0
    _log(log, f"  Rule labels — High: {(rule_hot==2).sum():,} | Medium: {(rule_hot==1).sum():,}")

    rows, cols, nf = feats.shape
    X = feats.reshape(-1, nf)
    y = rule_hot.flatten()
    MAX_TRAIN = 200_000
    if len(X) > MAX_TRAIN:
        idx = np.random.choice(len(X), MAX_TRAIN, replace=False)
        Xt, yt = X[idx], y[idx]
    else:
        Xt, yt = X, y
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(Xt)
    model  = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, eval_metric='mlogloss',
        n_jobs=-1, verbosity=0, scale_pos_weight=3)
    model.fit(Xs, yt)
    _log(log, f"  XGBoost trained on {len(Xt):,} samples | {nf} features")

    preds = np.zeros(len(X), dtype=np.uint8)
    conf  = np.zeros(len(X), dtype=np.float32)
    CHUNK = 500_000
    for s in range(0, len(X), CHUNK):
        e          = min(s + CHUNK, len(X))
        Xc         = scaler.transform(X[s:e])
        p          = model.predict_proba(Xc)
        preds[s:e] = model.predict(Xc).astype(np.uint8)
        conf[s:e]  = p[:, 2] if p.shape[1] > 2 else p[:, 1]
    hotspot_ml = preds.reshape(rows, cols)
    confidence = conf.reshape(rows, cols)
    _log(log, f"  Raw ML — High: {(hotspot_ml==2).sum():,} | Medium: {(hotspot_ml==1).sum():,}")
    return hotspot_ml, confidence, rule_hot


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MORPHOLOGICAL CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def morphological_cleanup(hotspot_ml, dtm_arr, log=None):
    _log(log, "📌 Step 5/9 — Morphological post-processing …")
    struct_open  = np.ones((3, 3), dtype=bool)
    struct_close = np.ones((5, 5), dtype=bool)
    outside      = (dtm_arr == NODATA)
    cleaned      = np.zeros_like(hotspot_ml, dtype=np.uint8)
    for cls in [2, 1]:
        mask = (hotspot_ml == cls)
        mask = binary_opening(mask,  structure=struct_open)
        mask = binary_closing(mask,  structure=struct_close)
        mask[outside] = False
        cleaned[mask] = cls
    removed = int((hotspot_ml > 0).sum()) - int((cleaned > 0).sum())
    _log(log, f"  Noise removed: {removed:,} px | Final High: {(cleaned==2).sum():,} | Medium: {(cleaned==1).sum():,}")
    ml_count = int((cleaned == 2).sum()) + int((cleaned == 1).sum())
    return cleaned, ml_count


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — COG RASTER EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_cog(arr, out_path, profile, dtype='float32', log=None):
    import rasterio
    p = profile.copy()
    p.update({'dtype': dtype, 'nodata': 0 if dtype == 'uint8' else NODATA})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **p) as dst:
        dst.write(arr.astype(dtype), 1)
    _log(log, f"  ✅ Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — DRAINAGE NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def design_drainage(name, hotspot_final, stream_raster,
                    dtm_arr, accum, transform, crs, log=None):
    import geopandas as gpd
    from shapely.geometry import LineString, shape
    from rasterio.features import shapes
    _log(log, "📌 Step 7/9 — Gravity-fed Dijkstra drainage routing …")

    dtm_f = np.nan_to_num(dtm_arr.copy().astype(np.float32),
                           nan=float(np.nanmax(dtm_arr[dtm_arr != NODATA])))

    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-9)

    cost = ((1 - norm(accum)) * 60 + norm(dtm_f) * 40 + 1).astype(np.float32)
    cost[dtm_arr == NODATA] = 9999.0

    labeled, n_clust = scipy_label((hotspot_final == 2).astype(np.uint8))
    _log(log, f"  Hotspot clusters: {n_clust}")

    stream_cells = np.argwhere(stream_raster > 0)
    if len(stream_cells) == 0:
        _log(log, "  No stream cells — skipping drain routing")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    targets = set(map(tuple, stream_cells.tolist()))

    def px_to_coord(r, c):
        return (transform.c + c * transform.a + 0.5 * transform.a,
                transform.f + r * transform.e + 0.5 * transform.e)

    def dijkstra(cost_arr, start, targets):
        rows_d, cols_d = cost_arr.shape
        dist  = np.full((rows_d, cols_d), np.inf, dtype=np.float32)
        prev  = {}
        dist[start] = 0
        heap = [(0.0, start)]
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in targets:
                path = []
                while u in prev:
                    path.append(u); u = prev[u]
                path.append(start)
                return list(reversed(path))
            if d > dist[u]: continue
            r_d, c_d = u
            for dr, dc in dirs:
                nr, nc = r_d + dr, c_d + dc
                if 0 <= nr < rows_d and 0 <= nc < cols_d:
                    nd = d + cost_arr[nr, nc] * (1.414 if dr and dc else 1.0)
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        prev[(nr, nc)] = u
                        heapq.heappush(heap, (nd, (nr, nc)))
        return []

    drain_lines  = []
    total_length = 0.0
    for cid in range(1, min(n_clust + 1, 51)):
        mask = (labeled == cid)
        if mask.sum() < 25: continue
        cy, cx = center_of_mass(mask)
        path   = dijkstra(cost, (int(cy), int(cx)), targets)
        if len(path) < 2: continue
        coords = [px_to_coord(r, c) for r, c in path]
        line   = LineString(coords)
        total_length += line.length
        drain_lines.append({'geometry': line, 'cluster_id': int(cid),
                            'hotspot_area': int(mask.sum()),
                            'length_m': round(line.length, 2),
                            'village': name, 'type': 'proposed_drain'})

    drains_gdf = gpd.GeoDataFrame(drain_lines, crs=crs)
    _log(log, f"  Proposed drains: {len(drains_gdf)} | Length: {total_length/1000:.2f} km")
    return drains_gdf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — GEOPACKAGE (hotspots + streams + drains)
# ─────────────────────────────────────────────────────────────────────────────

def export_geopackage(name, hotspot_final, stream_raster, slope,
                      drains_gdf, transform, crs, out_path, log=None):
    import geopandas as gpd
    from shapely.geometry import shape
    from rasterio.features import shapes
    _log(log, "📌 Step 8/9 — Building GeoPackage …")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Hotspots
    records = []
    for code, label in {1: 'Medium', 2: 'High'}.items():
        mask = (hotspot_final == code).astype(np.uint8)
        for geom, val in shapes(mask, transform=transform):
            if val == 1:
                g = shape(geom)
                records.append({'geometry': g, 'risk_level': label,
                                'risk_code': code, 'area_m2': round(g.area, 2),
                                'village': name})
    hotspots_gdf = gpd.GeoDataFrame(records, crs=crs)

    # Streams
    slope_clean = np.nan_to_num(slope, nan=0)
    stream_phys = ((stream_raster > 0) & (slope_clean > 0.01)).astype(np.uint8)
    stream_geoms = [shape(g) for g, v in shapes(stream_phys, transform=transform) if v == 1]
    streams_gdf = gpd.GeoDataFrame({'geometry': stream_geoms,
                                     'feature_type': 'drainage_channel',
                                     'village': name}, crs=crs)

    streams_gdf.to_file(out_path, layer='streams',  driver='GPKG')
    hotspots_gdf.to_file(out_path, layer='hotspots', driver='GPKG')
    if len(drains_gdf) > 0:
        drains_gdf.to_file(out_path, layer='proposed_drains', driver='GPKG')
    _log(log, f"  ✅ GeoPackage saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — VILLAGE REPORT PNG
# ─────────────────────────────────────────────────────────────────────────────

def generate_report_png(name, dtmarr, hotspotfinal, drainsgdf, transform,
                        outpath, log=None,
                        rulehot=None, slope=None, accum=None, twi=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    NODATA = -9999.0
    titlecolor = 'white'

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    def clean(arr, nd=NODATA):
        a = arr.copy().astype(np.float32)
        a[a == nd] = np.nan
        return a

    # --- Panel 1: DTM ---
    ax1 = fig.add_subplot(gs[0, :2])
    dtmv = clean(dtmarr)
    valid = dtmv[~np.isnan(dtmv)]
    im1 = ax1.imshow(dtmv, cmap='terrain', interpolation='bilinear',
                     vmin=np.percentile(valid, 1), vmax=np.percentile(valid, 99))
    plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.8)
    ax1.set_title(f'{name} — Digital Terrain Model', color=titlecolor, fontweight='bold')
    ax1.axis('off')

    # --- Panel 2: Risk + Drains ---
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor('white')
    outside = np.isnan(dtmv)
    cls = np.clip(np.round(hotspotfinal.astype(np.float32)), 0, 2).astype(np.uint8)
    cls[outside] = 255
    COLORMAP = {255: (0,0,0,0), 0: (0.82,0.94,0.78,1), 1: (1,0.55,0,1), 2: (0.8,0,0,1)}
    rgba = np.zeros((*cls.shape, 4), dtype=np.float32)
    for val, color in COLORMAP.items():
        rgba[cls == val] = color
    ax2.imshow(rgba, interpolation='nearest')
    if len(drainsgdf) > 0:
        xmint, ymaxt = transform.c, transform.f
        resx, resy = transform.a, abs(transform.e)
        for _, rowd in drainsgdf.iterrows():
            coords = list(rowd.geometry.coords)
            pxc = [(c[0] - xmint) / resx for c in coords]
            pxr = [(ymaxt - c[1]) / resy for c in coords]
            ax2.plot(pxc, pxr, color='cyan', linewidth=1.5, linestyle='--', alpha=0.9, zorder=5)
    ax2.legend(handles=[
        mpatches.Patch(color='#D1F0C7', label='Safe / Dry Ground'),
        mpatches.Patch(color='#FF8C00', label='Medium Risk'),
        mpatches.Patch(color='#CC0000', label='High Risk'),
        Line2D([0], [0], color='cyan', linestyle='--', linewidth=2, label='Proposed Drain'),
    ], loc='lower right', fontsize=8, facecolor='#333333', labelcolor='white')
    ax2.set_title(f'{name} — Waterlogging Risk + Proposed Drains', color=titlecolor, fontweight='bold')
    ax2.axis('off')

    # --- Panel 3: Confusion Matrix ---
    ax3 = fig.add_subplot(gs[1, :2])
    if rulehot is not None:
        from sklearn.metrics import confusion_matrix
        ytrue = rulehot.flatten()
        ypred = hotspotfinal.flatten()
        MAXCM = 500000
        if len(ytrue) > MAXCM:
            idx = np.random.choice(len(ytrue), MAXCM, replace=False)
            ytrue, ypred = ytrue[idx], ypred[idx]
        cm = confusion_matrix(ytrue, ypred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        im3 = ax3.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
        ax3.set_xticklabels(['No Risk','Medium','High'], color=titlecolor)
        ax3.set_yticklabels(['No Risk','Medium','High'], color=titlecolor)
        ax3.set_xlabel('ML Predicted', color=titlecolor)
        ax3.set_ylabel('Rule-Based True', color=titlecolor)
        ax3.set_title('Confusion Matrix (ML vs Rule)', color=titlecolor, fontweight='bold')
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center', va='center',
                         color='black', fontsize=10)
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        ax3.tick_params(colors=titlecolor)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'Confusion matrix\nnot available', transform=ax3.transAxes,
                 ha='center', va='center', color='gray', fontsize=10)

    # --- Panel 4: Classification Report Table ---
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')
    if rulehot is not None:
        from sklearn.metrics import classification_report
        report = classification_report(ytrue, ypred, labels=[0,1,2],
                                       target_names=['No Risk','Medium','High'],
                                       output_dict=True, zero_division=0)
        rows_r = ['No Risk','Medium','High','accuracy','macro avg','weighted avg']
        tabled = []
        for r in rows_r:
            if r not in report: continue
            rd = report[r]
            if isinstance(rd, dict):
                tabled.append([r,
                    f"{rd.get('precision',0):.3f}",
                    f"{rd.get('recall',0):.3f}",
                    f"{rd.get('f1-score',0):.3f}",
                    f"{int(rd.get('support',0))}"])
            else:
                tabled.append([r, '', '', f'{rd:.3f}', ''])
        tbl = ax4.table(cellText=tabled,
                        colLabels=['Class','Precision','Recall','F1','Support'],
                        loc='center', cellLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
        for (rowi, coli), cell in tbl.get_celld().items():
            cell.set_facecolor('#2d2d44' if rowi % 2 == 0 else '#1a1a2e')
            cell.set_text_props(color='white')
            cell.set_edgecolor('#555555')
    ax4.set_title('ML Classification Report', color=titlecolor, fontweight='bold')

    # --- Panel 5: Stats Summary ---
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.set_facecolor('#1a1a2e')
    nhigh = int((hotspotfinal == 2).sum())
    nmed  = int((hotspotfinal == 1).sum())
    area_km2 = round(dtmarr[dtmarr != NODATA].size / 1e6, 3)
    total_pts = int(dtmarr[dtmarr != NODATA].size)
    drain_km = round(sum(g.length for g in drainsgdf.geometry) / 1000, 2) if len(drainsgdf) > 0 else 0.0
    summary = (f"Village: {name}  |  Total LiDAR Points: {total_pts:,}  |  "
               f"Survey Area: {area_km2:.3f} km²  |  High Risk Cells: {nhigh:,}  |  "
               f"Medium Risk Cells: {nmed:,}  |  Proposed Drain Length: {drain_km} km  |  "
               f"Proposed Drains: {len(drainsgdf)}")
    ax5.text(0.5, 0.5, summary, transform=ax5.transAxes,
             ha='center', va='center', fontsize=10, color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44', edgecolor='#4488ff'))

    fig.suptitle(f'GramDrain — Village Analysis Report: {name}',
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    _log(log, f' ✅ Report PNG saved → {outpath}')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(las_path: str, epsg: int, village_name: str,
                 out_dir: str, log_callback=None):
    """
    Execute the complete GramDrain pipeline on a single .las/.laz file.

    Parameters
    ----------
    las_path     : full path to the .las or .laz file on disk
    epsg         : integer EPSG code for output projection
    village_name : clean label string (e.g. "KHAPRETA_510206")
    out_dir      : base output directory (subfolders created automatically)
    log_callback : callable(str) — receives progress lines for the Streamlit UI

    Returns
    -------
    dict of output file paths
    """
    _ensure_libs()
    os.makedirs(TMP_DIR, exist_ok=True)

    folders = {
        'ground':          os.path.join(out_dir, 'ground'),
        'dtm':             os.path.join(out_dir, 'dtm'),
        'rasters':         os.path.join(out_dir, 'rasters'),
        'gpkg':            os.path.join(out_dir, 'gpkg'),
        'village_reports': os.path.join(out_dir, 'village_reports'),
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)

    outputs = {
        'ground':     os.path.join(folders['ground'],          f"{village_name}_ground.las"),
        'dtm':        os.path.join(folders['dtm'],             f"{village_name}_dtm_cog.tif"),
        'slope':      os.path.join(folders['rasters'],         f"{village_name}_slope_cog.tif"),
        'accum':      os.path.join(folders['rasters'],         f"{village_name}_accum_cog.tif"),
        'twi':        os.path.join(folders['rasters'],         f"{village_name}_twi_cog.tif"),
        'risk':       os.path.join(folders['rasters'],         f"{village_name}_risk_cog.tif"),
        'confidence': os.path.join(folders['rasters'],         f"{village_name}_confidence_cog.tif"),
        'gpkg':       os.path.join(folders['gpkg'],            f"{village_name}_drainage.gpkg"),
        'report':     os.path.join(folders['village_reports'], f"{village_name}_report.png"),
    }

    _log(log_callback, f"🚀 Starting GramDrain pipeline for: {village_name}")
    _log(log_callback, f"   Input  : {las_path}")
    _log(log_callback, f"   EPSG   : {epsg}")
    _log(log_callback, f"   Output : {out_dir}")

    # ── Step 1: Ground classification ────────────────────────────────────────
    ground_pts = classify_ground(las_path, epsg, outputs['ground'], log=log_callback)

    # ── Step 2: DTM ──────────────────────────────────────────────────────────
    dtm_arr, transform, profile = generate_dtm(
        ground_pts, epsg, res=1.0, out_path=outputs['dtm'],
        tmp_dir=TMP_DIR, log=log_callback)
    del ground_pts; gc.collect()

    # ── Step 3: Derivatives ──────────────────────────────────────────────────
    slope = compute_slope(village_name, dtm_arr, profile, TMP_DIR, log=log_callback)
    accum = compute_flow_accumulation(dtm_arr, log=log_callback)
    dtm_c     = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    slope_rad = np.deg2rad(np.clip(np.nan_to_num(slope), 0.01, 89))
    twi       = np.log((accum * 100 + 1) / (np.tan(slope_rad) + 1e-6)).astype(np.float32)
    twi       = np.nan_to_num(twi)
    stream_thresh = float(np.percentile(accum[accum > 0], 97))
    stream_raster = (accum >= stream_thresh).astype(np.uint8)
    stream_raster[dtm_arr == NODATA] = 0
    _log(log_callback, f"📌 Step 3/9 — TWI + stream network computed | Stream cells: {stream_raster.sum():,}")

    # ── Step 4: XGBoost ──────────────────────────────────────────────────────
    feats = build_feature_matrix(dtm_arr, slope, accum, twi)
    hotspot_ml, confidence, rule_hot = train_and_predict(feats, dtm_arr, log=log_callback)
    del feats; gc.collect()

    # ── Step 5: Morphological cleanup ────────────────────────────────────────
    hotspot_final, ml_count = morphological_cleanup(hotspot_ml, dtm_arr, log=log_callback)
    if ml_count < 10:
        _log(log_callback, "  ⚠️ ML underperformed — using rule-based fallback")
        hotspot_final = rule_hot

    # ── Step 6: Save rasters ─────────────────────────────────────────────────
    _log(log_callback, "📌 Step 6/9 — Saving COG rasters …")
    risk_export = hotspot_final.copy().astype(np.float32)
    risk_export[dtm_arr == NODATA] = NODATA
    save_cog(risk_export,  outputs['risk'],       profile, log=log_callback)
    save_cog(confidence,   outputs['confidence'], profile, log=log_callback)
    save_cog(slope,        outputs['slope'],      profile, log=log_callback)
    save_cog(twi,          outputs['twi'],        profile, log=log_callback)
    save_cog(accum,        outputs['accum'],      profile, log=log_callback)

    # ── Step 7: Drainage routing ─────────────────────────────────────────────
    crs        = f"EPSG:{epsg}"
    drains_gdf = design_drainage(
        village_name, hotspot_final, stream_raster,
        dtm_arr, accum, transform, crs, log=log_callback)

    # ── Step 8: GeoPackage ───────────────────────────────────────────────────
    export_geopackage(village_name, hotspot_final, stream_raster, slope,
                      drains_gdf, transform, crs, outputs['gpkg'], log=log_callback)

    # ── Step 9: Report PNG ───────────────────────────────────────────────────
    generate_report_png(village_name, dtm_arr, hotspot_final,
                        drains_gdf, transform, outputs['report'], log=log_callback,
                        rulehot=rule_hot, slope=slope, accum=accum, twi=twi)

    _log(log_callback, "\n🎉 Pipeline complete! All 9 output files generated.")
    
    outputs['metrics'] = {
        'dtm_cells': int(dtm_arr.size),
        'dtm_rows': dtm_arr.shape[0],
        'dtm_cols': dtm_arr.shape[1],
        'slope_min': float(np.nanmin(slope)),
        'slope_max': float(np.nanmax(slope)),
        'twi_min': float(np.nanmin(twi)),
        'twi_max': float(np.nanmax(twi)),
        'final_high': int((hotspot_final == 2).sum()),
        'final_med': int((hotspot_final == 1).sum()),
        'rule_high': int((rule_hot == 2).sum()),
        'rule_med': int((rule_hot == 1).sum()),
        'ml_high_raw': int((hotspot_ml == 2).sum()) if 'hotspot_ml' in locals() else int((rule_hot == 2).sum()),
        'ml_med_raw': int((hotspot_ml == 1).sum()) if 'hotspot_ml' in locals() else int((rule_hot == 1).sum()),
        'stream_cells': int(stream_raster.sum()),
        'noise_removed': int((hotspot_ml > 0).sum() - (hotspot_final > 0).sum()) if 'hotspot_ml' in locals() else 0,
        'drains': len(drains_gdf),
        'drain_km': round(sum([g.length for g in drains_gdf.geometry]) / 1000, 2) if len(drains_gdf) > 0 else 0,
    }
    
    return outputs
