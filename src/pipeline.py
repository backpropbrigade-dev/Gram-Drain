# Full pipeline — dynamic scaling, 12 features, morphology, gravity routing, per-village reports, ML validation

import warnings
warnings.filterwarnings('ignore')
from rasterio.transform import from_origin
from rasterio.features   import shapes
from scipy.ndimage import (median_filter, gaussian_filter, uniform_filter,
                            minimum_filter, laplace,
                            label as scipy_label, center_of_mass,
                            binary_opening, binary_closing, generic_filter)
from sklearn.metrics import (confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import json, textwrap

def _save_ground_las(name, ground_pts):
    """Write classified ground points to a LAS 1.2 file (class 2)."""
    header              = laspy.LasHeader(point_format=2, version="1.2")
    header.scales       = np.array([0.001, 0.001, 0.001])
    header.offsets      = ground_pts.min(axis=0)
    glas                = laspy.LasData(header)
    glas.x              = ground_pts[:, 0]
    glas.y              = ground_pts[:, 1]
    glas.z              = ground_pts[:, 2]
    glas.classification = np.full(len(ground_pts), 2, dtype=np.uint8)
    glas.write(f'{OUT_BASE}/ground/{name}_ground.las')
    print(f"  Saved LAS      : {OUT_BASE}/ground/{name}_ground.las")
    del glas

# ─────────────────────────────────────────────────────────────────────────
# GROUND CLASSIFICATION — REGULAR
# ─────────────────────────────────────────────────────────────────────────
def classify_ground_regular(name, cfg):
    print(f"\n[{name}] Ground Classification (Regular)")
    las = laspy.read(cfg['raw'])
    x, y, z   = np.array(las.x), np.array(las.y), np.array(las.z)
    total_pts = len(x)
    print(f"  Raw points : {total_pts:,}")
    del las; gc.collect()

    if len(x) > MAX_PTS:
        xi    = ((x - x.min()) / 0.2).astype(np.int32)
        yi    = ((y - y.min()) / 0.2).astype(np.int32)
        keys  = xi.astype(np.int64) * 100_000 + yi.astype(np.int64)
        order = np.argsort(keys)
        keys, x, y, z = keys[order], x[order], y[order], z[order]
        _, first = np.unique(keys, return_index=True)
        x, y, z  = x[first], y[first], z[first]
        if len(x) > MAX_PTS:
            idx = np.random.choice(len(x), MAX_PTS, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
        print(f"  Downsampled: {len(x):,}")

    points = np.vstack((x, y, z)).T.astype(np.float64)
    del x, y, z; gc.collect()

    csf_obj = CSF.CSF()
    csf_obj.params.bSloopSmooth     = True
    csf_obj.params.cloth_resolution = 0.5
    csf_obj.params.rigidness        = 3
    csf_obj.params.time_step        = 0.65
    csf_obj.params.class_threshold  = 0.5
    csf_obj.params.interations      = 500
    csf_obj.setPointCloud(points)
    gnd_idx, off_idx = CSF.VecInt(), CSF.VecInt()
    csf_obj.do_filtering(gnd_idx, off_idx)

    gnd        = np.array(gnd_idx)
    ground_pts = points[gnd]
    print(f"  Ground     : {len(gnd):,} ({100*len(gnd)/len(points):.1f}%)")
    del points, gnd, off_idx; gc.collect()

    _save_ground_las(name, ground_pts)
    VILLAGES[name]['total_pts'] = total_pts
    return ground_pts


# ─────────────────────────────────────────────────────────────────────────
# DYNAMIC MEMORY-SCALING PARAMETER CALCULATOR
# ─────────────────────────────────────────────────────────────────────────
def _compute_dynamic_params(total_pts, xmin, xmax, ymin, ymax, is_geographic):
    """
    Computes VOXEL_RES and STRIDE such that:
      Constraint 1: voxel grid  ≤ 4000 × 4000 = 16M cells  (dict RAM)
      Constraint 2: cell_min occupancy ≤ 33M entries        (heap RAM)
      Constraint 3: stride limits raw throughput for 1B+    (GC pressure)
    """
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
    voxel_m = next((v for v in CANDIDATES_M if v >= min_voxel_m),
                   max(CANDIDATES_M))

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

    info = {
        'voxel_m'       : voxel_m,
        'VOXEL_RES'     : VOXEL_RES,
        'grid_cols'     : grid_cols,
        'grid_rows'     : grid_rows,
        'worst_cells_M' : round(worst_case_cells / 1e6, 2),
        'STRIDE'        : STRIDE,
        'effective_pts' : total_pts // STRIDE,
    }
    return VOXEL_RES, STRIDE, grid_cols, grid_rows, info


# ─────────────────────────────────────────────────────────────────────────
# GROUND CLASSIFICATION — CHUNKED + DYNAMIC SCALING (large files)
# ─────────────────────────────────────────────────────────────────────────
def classify_ground_chunked(name, cfg):
    print(f"\n[{name}] Ground Classification (Chunked + NumPy Grid Accumulator)")

    with laspy.open(cfg['raw']) as f:
        hdr       = f.header
        total_pts = int(hdr.point_count)
        xmin_h    = float(hdr.mins[0]); xmax_h = float(hdr.maxs[0])
        ymin_h    = float(hdr.mins[1]); ymax_h = float(hdr.maxs[1])

    coord_range   = max(xmax_h - xmin_h, ymax_h - ymin_h)
    is_geographic = coord_range < 2.0

    # _compute_dynamic_params is unchanged — still calculates VOXEL_RES + STRIDE
    VOXEL_RES, STRIDE, grid_cols, grid_rows, info = _compute_dynamic_params(
        total_pts, xmin_h, xmax_h, ymin_h, ymax_h, is_geographic
    )

    print(f"  Total pts      : {total_pts:,}")
    print(f"  Coord type     : {'GEOGRAPHIC' if is_geographic else 'PROJECTED'}")
    print(f"  Voxel res      : {info['voxel_m']:.3f} m  ({VOXEL_RES:.8f} native)")
    print(f"  Voxel grid     : {grid_cols} x {grid_rows} "
          f"= {info['worst_cells_M']}M cells")
    print(f"  Stride         : 1/{STRIDE}  "
          f"(effective pts: {info['effective_pts']:,})")

    # ── Pre-allocate NumPy grid arrays ────────────────────────────────────
    # Memory: z=float32 (4B) + x,y=float64 (8B each) = 20B/cell
    # Worst case 4000×4000 = 16M cells → 305 MB   (dict = 4.9 GB for same data)
    grid_mem_mb = (grid_rows * grid_cols * 20) / (1024 ** 2)
    print(f"  Grid RAM       : {grid_mem_mb:.1f} MB  "
          f"({grid_rows}r × {grid_cols}c × 20 B/cell)")

    INF32 = np.float32(np.inf)
    min_z = np.full((grid_rows, grid_cols), INF32,  dtype=np.float32)
    min_x = np.zeros((grid_rows, grid_cols),        dtype=np.float64)
    min_y = np.zeros((grid_rows, grid_cols),        dtype=np.float64)

    # ── Chunked min-z accumulation ────────────────────────────────────────
    processed = 0
    chunk_num = 0

    with laspy.open(cfg['raw']) as f:
        for chunk in f.chunk_iterator(2_000_000):
            n_chunk = len(chunk.x)

            # Stride decimation — keeps deterministic every Nth point
            if STRIDE > 1:
                keep = np.arange(0, n_chunk, STRIDE)
                x = np.asarray(chunk.x)[keep].astype(np.float64)
                y = np.asarray(chunk.y)[keep].astype(np.float64)
                z = np.asarray(chunk.z)[keep].astype(np.float32)
                del keep
            else:
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float32)

            # Grid cell indices
            xi = np.clip(
                ((x - xmin_h) / VOXEL_RES).astype(np.int32), 0, grid_cols - 1
            )
            yi = np.clip(
                ((y - ymin_h) / VOXEL_RES).astype(np.int32), 0, grid_rows - 1
            )

            # ── Vectorized intra-chunk deduplication ──────────────────────
            # Flat row-major index: cell (yi, xi) → yi*grid_cols + xi
            # np.lexsort sorts by flat key first, then z (ascending)
            # → first occurrence per key after sort = minimum z in this chunk
            flat = yi.astype(np.int64) * grid_cols + xi.astype(np.int64)
            order = np.lexsort((z, flat))    # primary: flat, secondary: z asc
            flat_s = flat[order]
            x_s    = x[order]
            y_s    = y[order]
            z_s    = z[order]

            _, first_idx = np.unique(flat_s, return_index=True)
            flat_u = flat_s[first_idx]   # one entry per unique cell in chunk
            x_u    = x_s[first_idx]
            y_u    = y_s[first_idx]
            z_u    = z_s[first_idx]      # minimum z per cell in this chunk

            del flat_s, x_s, y_s, z_s, order
            # ── end dedup ─────────────────────────────────────────────────

            # Recover (row, col) from flat index
            ri = (flat_u // grid_cols).astype(np.int32)
            ci = (flat_u %  grid_cols).astype(np.int32)
            del flat_u, flat, xi, yi

            # Vectorized grid update — only write where this chunk beats current min
            update_mask = z_u < min_z[ri, ci]
            ri_u = ri[update_mask]
            ci_u = ci[update_mask]

            min_z[ri_u, ci_u] = z_u[update_mask]
            min_x[ri_u, ci_u] = x_u[update_mask]
            min_y[ri_u, ci_u] = y_u[update_mask]

            processed += n_chunk
            chunk_num += 1

            del x, y, z, x_u, y_u, z_u, ri, ci, update_mask, ri_u, ci_u
            gc.collect()   # every chunk — grid arrays stay, temporaries are freed

            if chunk_num % 20 == 0:
                filled = int(np.sum(np.isfinite(min_z)))
                print(f"  Processed {processed:,} / {total_pts:,} pts  "
                      f"| grid cells filled: {filled:,}")

    # ── Extract populated cells into point array ──────────────────────────
    # This is O(grid_rows*grid_cols) numpy boolean mask — no Python-level loop
    valid_mask = np.isfinite(min_z)
    n_valid    = int(valid_mask.sum())
    print(f"  Grid cells     : {n_valid:,} filled / "
          f"{grid_rows * grid_cols:,} total")

    pts_ds = np.column_stack([
        min_x[valid_mask],
        min_y[valid_mask],
        min_z[valid_mask].astype(np.float64)
    ])                              # shape (n_valid, 3), contiguous float64

    # Immediately free the large grid arrays — pts_ds is all we need
    del min_z, min_x, min_y, valid_mask
    gc.collect()

    # Safety cap (should never trigger if _compute_dynamic_params is correct)
    if len(pts_ds) > MAX_PTS:
        idx    = np.random.choice(len(pts_ds), MAX_PTS, replace=False)
        pts_ds = pts_ds[idx]
        del idx; gc.collect()
        print(f"  Safety cap     : {MAX_PTS:,} pts applied")

    print(f"  Post-cap pts   : {len(pts_ds):,}")

    # ── Reproject geographic → projected ──────────────────────────────────
    if is_geographic:
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:4326", f"EPSG:{cfg['epsg']}",
                                  always_xy=True)
        gx, gy      = t.transform(pts_ds[:, 0], pts_ds[:, 1])
        pts_ds[:, 0] = gx
        pts_ds[:, 1] = gy
        del gx, gy; gc.collect()
        print(f"  Reprojected    : EPSG:{cfg['epsg']}")

    # ── CSF ground classification ─────────────────────────────────────────
    csf_obj = CSF.CSF()
    csf_obj.params.bSloopSmooth     = True
    csf_obj.params.cloth_resolution = 0.5
    csf_obj.params.rigidness        = 3
    csf_obj.params.time_step        = 0.65
    csf_obj.params.class_threshold  = 0.5
    csf_obj.params.interations      = 500
    csf_obj.setPointCloud(pts_ds)

    gnd_idx = CSF.VecInt(); off_idx = CSF.VecInt()
    csf_obj.do_filtering(gnd_idx, off_idx)

    gnd        = np.array(gnd_idx)
    ground_pts = pts_ds[gnd].copy()
    print(f"  Ground pts     : {len(gnd):,} ({100*len(gnd)/len(pts_ds):.1f}%)")
    print(f"  Off-ground     : {len(np.array(off_idx)):,}")

    del pts_ds, gnd, off_idx, gnd_idx, csf_obj; gc.collect()

    _save_ground_las(name, ground_pts)
    VILLAGES[name]['total_pts'] = total_pts
    return ground_pts


# ─────────────────────────────────────────────────────────────────────────
# DTM
# ─────────────────────────────────────────────────────────────────────────
def generate_dtm(name, cfg, ground_pts):
    print(f"\n[{name}] DTM Generation")
    x, y, z    = ground_pts[:,0], ground_pts[:,1], ground_pts[:,2]
    res        = cfg['res']
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    cols = int((xmax - xmin) / res) + 1
    rows = int((ymax - ymin) / res) + 1
    print(f"  Grid       : {cols} x {rows} @ {res}m")

    VILLAGES[name]['area_m2'] = round((xmax-xmin) * (ymax-ymin), 1)

    dtm = np.full((rows, cols), np.nan, dtype=np.float32)
    xi  = np.clip(((x - xmin) / res).astype(np.int32), 0, cols-1)
    yi  = np.clip(((ymax - y) / res).astype(np.int32), 0, rows-1)
    for i in range(len(x)):
        r, c = yi[i], xi[i]
        if np.isnan(dtm[r,c]) or z[i] < dtm[r,c]:
            dtm[r,c] = z[i]

    dtm_f = np.where(np.isnan(dtm), 0.0, dtm).astype(np.float32)
    wmask = (~np.isnan(dtm)).astype(np.float32)
    denom = np.maximum(uniform_filter(wmask, size=5), 1e-6)
    dtm   = (uniform_filter(dtm_f, size=5) / denom).astype(np.float32)
    dtm   = median_filter(dtm, size=3).astype(np.float32)
    dtm   = gaussian_filter(dtm.astype(np.float64), sigma=1).astype(np.float32)
    dtm[~np.isfinite(dtm)] = NODATA

    valid = dtm[dtm != NODATA]
    print(f"  Z range    : {valid.min():.2f}m -> {valid.max():.2f}m")
    del dtm_f, wmask, denom; gc.collect()

    transform = from_origin(xmin, ymax, res, res)
    profile   = {
        'driver': 'GTiff', 'height': rows, 'width': cols,
        'count': 1, 'dtype': 'float32',
        'crs': f'EPSG:{cfg["epsg"]}', 'transform': transform,
        'nodata': NODATA, 'compress': 'lzw',
        'tiled': True, 'blockxsize': 256, 'blockysize': 256, 'predictor': 2
    }
    local = f'/tmp/mopr/{name}_dtm.tif'
    out   = f'{OUT_BASE}/dtm/{name}_dtm_cog.tif'
    with rasterio.open(local, 'w', **profile) as dst:
        dst.write(dtm, 1)
    shutil.copy2(local, out)
    print(f"  OGC COG    : {out}")
    return local, dtm, transform, profile


# ─────────────────────────────────────────────────────────────────────────
# SLOPE + FLOW ACCUMULATION
# ─────────────────────────────────────────────────────────────────────────
def compute_slope(name, dtm_arr, profile):
    import whitebox
    wbt = whitebox.WhiteboxTools(); wbt.verbose = False
    dtm_wbt    = f'/tmp/mopr/{name}_dtm_wbt.tif'
    valid_mean = float(np.nanmean(dtm_arr[dtm_arr != NODATA]))
    dtm_filled = dtm_arr.copy()
    dtm_filled[dtm_filled == NODATA] = valid_mean
    p = profile.copy(); p['nodata'] = None
    with rasterio.open(dtm_wbt, 'w', **p) as dst:
        dst.write(dtm_filled.astype(np.float32), 1)
    del dtm_filled

    slope_path = f'/tmp/mopr/{name}_slope.tif'
    try:
        wbt.slope(dtm_wbt, slope_path, zfactor=1.0)
        with rasterio.open(slope_path) as src:
            slope = src.read(1).astype(np.float32)
            nd    = src.nodata
            if nd is not None: slope[slope == nd] = np.nan
        os.remove(slope_path)
    except Exception as e:
        print(f"  WBT slope fallback ({e})")
        dy, dx = np.gradient(np.nan_to_num(dtm_arr.copy(), nan=0))
        slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)
    os.remove(dtm_wbt)
    return slope


def numpy_flow_accumulation(dtm_arr):
    dtm_c = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    dtm_f = np.nan_to_num(dtm_c, nan=float(np.nanmax(dtm_c)))

    r5  = dtm_f - minimum_filter(dtm_f, size=5)
    r21 = dtm_f - minimum_filter(dtm_f, size=21)
    r51 = dtm_f - minimum_filter(dtm_f, size=51)

    accum = (0.2*np.exp(-r5  / (r5.std()  + 1e-6))
           + 0.3*np.exp(-r21 / (r21.std() + 1e-6))
           + 0.5*np.exp(-r51 / (r51.std() + 1e-6))).astype(np.float32)
    accum = gaussian_filter(accum, sigma=3).astype(np.float32)
    accum[dtm_arr == NODATA] = 0.0
    del r5, r21, r51, dtm_f; gc.collect()
    return accum


# ─────────────────────────────────────────────────────────────────────────
# 12-FEATURE MATRIX WITH NEIGHBORHOOD STATISTICS
# ─────────────────────────────────────────────────────────────────────────
def build_feature_matrix(dtm_arr, slope, accum, twi):
    """
    12 features per pixel:
      Point      : elevation, slope, accum, twi, curvature, dist_outlet
      5x5 window : mean_elev, std_elev, mean_slope, mean_twi
      11x11 window: mean_accum, range_elev
    """
    dtm_c = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    dtm_n = np.nan_to_num(dtm_c, nan=0)
    sl_n  = np.nan_to_num(slope,  nan=0)
    ac_n  = np.nan_to_num(accum,  nan=0)
    tw_n  = np.nan_to_num(twi,    nan=0)

    curv = laplace(dtm_n).astype(np.float32)
    fi   = np.argmin(np.nan_to_num(dtm_c, nan=9999))
    rc, cc = np.unravel_index(fi, dtm_c.shape)
    dist = np.sqrt(
        (np.arange(dtm_c.shape[0]).reshape(-1,1) - rc)**2 +
        (np.arange(dtm_c.shape[1]).reshape(1,-1) - cc)**2
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
        mean_elev_5, std_elev_5,
        mean_slope_5, mean_twi_5,
        mean_accum_11, range_elev_11
    ], axis=-1)

    del dtm_n, sl_n, ac_n, tw_n, curv, dist
    del mean_elev_5, std_elev_5, mean_slope_5, mean_twi_5
    del mean_accum_11, range_elev_11, min_elev_11
    gc.collect()
    return feats


# ─────────────────────────────────────────────────────────────────────────
# XGBOOST TRAIN / PREDICT
# ─────────────────────────────────────────────────────────────────────────
def train_xgboost(feats, rule_hotspot):
    rows, cols, nf = feats.shape
    X = feats.reshape(-1, nf)
    y = rule_hotspot.flatten()
    MAX_TRAIN = 200_000
    if len(X) > MAX_TRAIN:
        idx = np.random.choice(len(X), MAX_TRAIN, replace=False)
        X, y = X[idx], y[idx]
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, eval_metric='mlogloss',
        n_jobs=-1, verbosity=0, scale_pos_weight=3
    )
    model.fit(Xs, y)
    vals, cnts = np.unique(y, return_counts=True)
    print(f"  Train labels   : { {int(v):int(c) for v,c in zip(vals,cnts)} }")
    print(f"  XGBoost trained: {len(X):,} px | {nf} features")
    return model, scaler


def predict_ml(feats, model, scaler):
    rows, cols, nf = feats.shape
    X     = feats.reshape(-1, nf)
    preds = np.zeros(len(X), dtype=np.uint8)
    conf  = np.zeros(len(X), dtype=np.float32)
    CHUNK = 500_000
    for s in range(0, len(X), CHUNK):
        e         = min(s+CHUNK, len(X))
        Xc        = scaler.transform(X[s:e])
        p         = model.predict_proba(Xc)
        preds[s:e] = model.predict(Xc).astype(np.uint8)
        conf[s:e]  = p[:,2] if p.shape[1] > 2 else p[:,1]
    return preds.reshape(rows,cols), conf.reshape(rows,cols)


# ─────────────────────────────────────────────────────────────────────────
# MORPHOLOGICAL POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────
def morphological_cleanup(hotspot_ml, dtm_arr):
    """
    binary_opening (3x3): removes isolated salt-and-pepper noise pixels.
    binary_closing (5x5): fills small holes within hotspot clusters.
    Applied per class (High then Medium) to preserve class labels.
    """
    struct_open  = np.ones((3,3), dtype=bool)
    struct_close = np.ones((5,5), dtype=bool)
    outside      = (dtm_arr == NODATA)
    cleaned      = np.zeros_like(hotspot_ml, dtype=np.uint8)

    for cls in [2, 1]:
        mask = (hotspot_ml == cls)
        mask = binary_opening(mask,  structure=struct_open)
        mask = binary_closing(mask,  structure=struct_close)
        mask[outside] = False
        cleaned[mask] = cls

    removed = int((hotspot_ml > 0).sum()) - int((cleaned > 0).sum())
    print(f"  Morphology: removed {removed:,} noise pixels")
    print(f"  Final High : {(cleaned==2).sum():,} | Medium: {(cleaned==1).sum():,}")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────
# RISK → RGBA
# ─────────────────────────────────────────────────────────────────────────
def risk_to_rgba(risk_arr, dtm_arr):
    """
    Convert risk raster to RGBA image with correct transparency.

    NODATA (-9999) = outside village boundary       → fully transparent
    0              = safe/dry ground inside village → light green
    1              = medium waterlogging risk       → orange
    2              = high waterlogging risk         → red
    """
    outside = (risk_arr < -100) | (dtm_arr == NODATA)
    cls     = np.clip(np.round(risk_arr), 0, 2).astype(np.uint8)
    cls[outside] = 255   # sentinel for transparent

    COLOR = {
        255: (0.00, 0.00, 0.00, 0.00),   # transparent — outside village
          0: (0.82, 0.94, 0.78, 1.00),   # light green — safe/dry ground
          1: (1.00, 0.55, 0.00, 1.00),   # orange      — medium risk
          2: (0.80, 0.00, 0.00, 1.00),   # red         — high risk
    }
    rgba = np.zeros((*cls.shape, 4), dtype=np.float32)
    for val, color in COLOR.items():
        rgba[cls == val] = color

    return rgba, cls


# ─────────────────────────────────────────────────────────────────────────
# GRAVITY-CORRECT DIJKSTRA DRAINAGE ROUTING
# ─────────────────────────────────────────────────────────────────────────
def design_drainage_network(name, hotspot_final, stream_raster,
                             dtm_arr, accum, transform, crs):
    """
    cost = (1 - norm_accum)*60 + norm_elev*40 + 1

    Low cost along natural valleys (high accum + low elevation) ensures
    all proposed drains are gravity-fed and follow existing drainage axes.
    NODATA cells = 9999 (impenetrable) — drains never exit village boundary.
    """
    dtm_c = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    dtm_f = np.nan_to_num(dtm_c, nan=float(np.nanmax(dtm_c)))

    def norm(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-9)

    cost = ((1 - norm(accum)) * 60 + norm(dtm_f) * 40 + 1).astype(np.float32)
    cost[dtm_arr == NODATA] = 9999.0

    labeled, n_clust = scipy_label((hotspot_final == 2).astype(np.uint8))
    print(f"  Hotspot clusters : {n_clust}")

    stream_cells = np.argwhere(stream_raster > 0)
    if len(stream_cells) == 0:
        print("  No stream cells — skipping drain routing")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    targets = set(map(tuple, stream_cells.tolist()))

    def px_to_coord(r, c):
        return (transform.c + c*transform.a + 0.5*transform.a,
                transform.f + r*transform.e + 0.5*transform.e)

    def dijkstra(cost_arr, start, targets):
        rows, cols = cost_arr.shape
        dist  = np.full((rows,cols), np.inf, dtype=np.float32)
        prev  = {}
        dist[start] = 0
        heap = [(0.0, start)]
        dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in targets:
                path = []
                while u in prev: path.append(u); u = prev[u]
                path.append(start)
                return list(reversed(path))
            if d > dist[u]: continue
            r, c = u
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nd = d + cost_arr[nr,nc]*(1.414 if dr and dc else 1.0)
                    if nd < dist[nr,nc]:
                        dist[nr,nc] = nd
                        prev[(nr,nc)] = u
                        heapq.heappush(heap, (nd,(nr,nc)))
        return []

    drain_lines  = []
    total_length = 0.0
    for cid in range(1, min(n_clust+1, 51)):
        mask = (labeled == cid)
        if mask.sum() < 25: continue
        cy, cx = center_of_mass(mask)
        path   = dijkstra(cost, (int(cy),int(cx)), targets)
        if len(path) < 2: continue
        coords = [px_to_coord(r,c) for r,c in path]
        line   = LineString(coords)
        total_length += line.length
        drain_lines.append({'geometry': line, 'cluster_id': int(cid),
                            'hotspot_area': int(mask.sum()),
                            'length_m': round(line.length,2),
                            'village': name, 'type': 'proposed_drain'})

    drains_gdf = gpd.GeoDataFrame(drain_lines, crs=crs)
    print(f"  Proposed drains  : {len(drains_gdf)} | "
          f"Total length: {total_length/1000:.2f} km")
    VILLAGES[name]['drain_length_km'] = round(total_length/1000, 2)
    return drains_gdf


# ─────────────────────────────────────────────────────────────────────────
# SAVE COG HELPER
# ─────────────────────────────────────────────────────────────────────────
def save_cog(name, arr, tag, profile, dtype='float32'):
    p = profile.copy()
    p.update({'dtype': dtype,
              'nodata': 0 if dtype == 'uint8' else NODATA})
    with rasterio.open(f'{OUT_BASE}/rasters/{name}_{tag}_cog.tif',
                       'w', **p) as dst:
        dst.write(arr.astype(dtype), 1)


# ─────────────────────────────────────────────────────────────────────────
# PER-VILLAGE REPORT WITH ML VALIDATION
# ─────────────────────────────────────────────────────────────────────────
def generate_village_report(name, cfg, dtm_arr, slope, accum, twi,
                             rule_hot, hotspot_final, drains_gdf,
                             streams_gdf, hotspots_gdf, transform):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#1a1a2e')
    title_color = 'white'
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    def clean(arr, nd=NODATA):
        a = arr.copy().astype(np.float32)
        a[a == nd] = np.nan
        return a

    # ── Panel 1: DTM ──────────────────────────────────────────────────────
    ax1   = fig.add_subplot(gs[0, :2])
    dtm_v = clean(dtm_arr)
    valid = dtm_v[~np.isnan(dtm_v)]
    im1   = ax1.imshow(dtm_v, cmap='terrain', interpolation='bilinear',
                       vmin=np.percentile(valid,1), vmax=np.percentile(valid,99))
    plt.colorbar(im1, ax=ax1, label='Elevation (m)', shrink=0.8)
    ax1.set_title(f'{name} — Digital Terrain Model',
                  color=title_color, fontweight='bold')
    ax1.axis('off')

    # ── Panel 2: Risk + Drains (4-class RGBA, transparent NODATA) ────────
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.set_facecolor('white')
    rgba, cls = risk_to_rgba(hotspot_final.astype(np.float32), dtm_arr)
    ax2.imshow(rgba, interpolation='nearest')

    if len(drains_gdf) > 0:
        xmin_t = transform.c; ymax_t = transform.f
        res_x  = transform.a; res_y  = abs(transform.e)
        for _, row_d in drains_gdf.iterrows():
            coords = list(row_d.geometry.coords)
            px_c   = [(c[0] - xmin_t) / res_x for c in coords]
            px_r   = [(ymax_t - c[1]) / res_y  for c in coords]
            ax2.plot(px_c, px_r, color='cyan', linewidth=1.5,
                     linestyle='--', alpha=0.9, zorder=5)

    ax2.legend(handles=[
        mpatches.Patch(color='#D1F0C7', label='Safe / Dry Ground'),
        mpatches.Patch(color='#FF8C00', label='Medium Risk'),
        mpatches.Patch(color='#CC0000', label='High Risk'),
        Line2D([0],[0], color='cyan', linestyle='--',
               linewidth=2, label='Proposed Drain'),
    ], loc='lower right', fontsize=8, facecolor='#333333', labelcolor='white')
    ax2.set_title(f'{name} — Waterlogging Risk + Proposed Drains',
                  color=title_color, fontweight='bold')
    ax2.axis('off')

    # ── Panel 3: Confusion Matrix ─────────────────────────────────────────
    ax3    = fig.add_subplot(gs[1, :2])
    y_true = rule_hot.flatten()
    y_pred = hotspot_final.flatten()
    MAX_CM = 500_000
    if len(y_true) > MAX_CM:
        idx    = np.random.choice(len(y_true), MAX_CM, replace=False)
        y_true = y_true[idx]; y_pred = y_pred[idx]

    cm      = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im3     = ax3.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax3.set_xticks([0,1,2]); ax3.set_yticks([0,1,2])
    ax3.set_xticklabels(['No Risk','Medium','High'], color=title_color)
    ax3.set_yticklabels(['No Risk','Medium','High'], color=title_color)
    ax3.set_xlabel('ML Predicted',     color=title_color)
    ax3.set_ylabel('Rule-Based True',  color=title_color)
    ax3.set_title('Confusion Matrix (ML vs Rule)',
                  color=title_color, fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax3.text(j, i, f'{cm_norm[i,j]:.2f}', ha='center',
                     va='center', color='black', fontsize=10)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    ax3.tick_params(colors=title_color)

    # ── Panel 4: Classification Report Table ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')
    report = classification_report(
        y_true, y_pred, labels=[0,1,2],
        target_names=['No Risk','Medium','High'],
        output_dict=True, zero_division=0)

    rows_r  = ['No Risk','Medium','High','accuracy','macro avg','weighted avg']
    table_d = []
    for r in rows_r:
        if r not in report: continue
        rd = report[r]
        if isinstance(rd, dict):
            table_d.append([r,
                f"{rd.get('precision',0):.3f}",
                f"{rd.get('recall',0):.3f}",
                f"{rd.get('f1-score',0):.3f}",
                f"{int(rd.get('support',0)):,}"])
        else:
            table_d.append([r, '', '', f"{rd:.3f}", ''])

    tbl = ax4.table(cellText=table_d,
                    colLabels=['Class','Precision','Recall','F1','Support'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_facecolor('#2d2d44' if row_i % 2 == 0 else '#1a1a2e')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#555555')
    ax4.set_title('ML Classification Report',
                  color=title_color, fontweight='bold')

    # ── Panel 5: Stats Summary Bar ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    ax5.set_facecolor('#1a1a2e')
    n_high    = int((hotspot_final==2).sum())
    n_med     = int((hotspot_final==1).sum())
    area_km2  = round(VILLAGES[name].get('area_m2',  0) / 1e6, 3)
    total_pts = VILLAGES[name].get('total_pts', 0)
    drain_km  = VILLAGES[name].get('drain_length_km', 0.0)

    summary_text = (
        f"Village: {name}   |   Total LiDAR Points: {total_pts:,}   |   "
        f"Survey Area: {area_km2} km²   |   High Risk Cells: {n_high:,}   |   "
        f"Medium Risk Cells: {n_med:,}   |   "
        f"Proposed Drain Length: {drain_km} km   |   "
        f"Proposed Drains: {len(drains_gdf)}"
    )
    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
             ha='center', va='center', fontsize=10, color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2d2d44',
                       edgecolor='#4488ff'))

    fig.suptitle(f'GramDrain — Village Analysis Report: {name}',
                 fontsize=16, fontweight='bold', color='white', y=0.98)
    out = f'{OUT_BASE}/village_reports/{name}_report.png'
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Report saved : {out}")
    return report


# ─────────────────────────────────────────────────────────────────────────
# FULL VILLAGE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────
def process_village(name, cfg, ground_pts,
                    xgb_model=None, xgb_scaler=None):

    # 1. DTM
    dtm_entry = generate_dtm(name, cfg, ground_pts)
    dtm_path, dtm_arr, transform, profile = dtm_entry
    del ground_pts; gc.collect()

    # 2. Slope + Flow Accumulation
    print(f"\n[{name}] Terrain Derivatives")
    slope = compute_slope(name, dtm_arr, profile)
    accum = numpy_flow_accumulation(dtm_arr)

    stream_thresh = float(np.percentile(accum[accum > 0], 97))
    stream_raster = (accum >= stream_thresh).astype(np.uint8)
    stream_raster[dtm_arr == NODATA] = 0
    print(f"  Slope range  : {np.nanmin(slope):.2f} -> {np.nanmax(slope):.2f} deg")
    print(f"  Stream cells : {stream_raster.sum():,}")

    # 3. TWI
    dtm_c     = dtm_arr.copy().astype(np.float32)
    dtm_c[dtm_c == NODATA] = np.nan
    slope_rad = np.deg2rad(np.clip(np.nan_to_num(slope), 0.01, 89))
    twi       = np.log((accum*100 + 1) / (np.tan(slope_rad) + 1e-6)).astype(np.float32)
    twi       = np.nan_to_num(twi)
    gc.collect()

    # 4. Rule-based labels (training targets for XGBoost)
    def norm(a):
        a = np.nan_to_num(a).astype(np.float32)
        return (a - a.min()) / (a.max() - a.min() + 1e-9)

    risk_rule = norm(0.40*norm(twi) + 0.35*norm(accum)
                   + 0.15*(1-norm(slope)) + 0.10*(1-norm(dtm_c)))
    rule_hot  = np.zeros_like(risk_rule, dtype=np.uint8)
    rule_hot[risk_rule > 0.92]                          = 2
    rule_hot[(risk_rule >= 0.78) & (risk_rule <= 0.92)] = 1
    rule_hot[dtm_arr == NODATA] = 0
    print(f"\n[{name}] Rule labels  — High: {(rule_hot==2).sum():,} "
          f"| Medium: {(rule_hot==1).sum():,}")

    # 5. XGBoost (per-village retrain with 12-feature matrix)
    print(f"\n[{name}] XGBoost")
    feats = build_feature_matrix(dtm_arr, slope, accum, twi)
    xgb_model, xgb_scaler = train_xgboost(feats, rule_hot)
    hotspot_ml, confidence = predict_ml(feats, xgb_model, xgb_scaler)
    print(f"  Raw ML — High: {(hotspot_ml==2).sum():,} "
          f"| Medium: {(hotspot_ml==1).sum():,}")
    del feats; gc.collect()

    # 6. Morphological cleanup
    print(f"\n[{name}] Morphological Post-processing")
    hotspot_ml = morphological_cleanup(hotspot_ml, dtm_arr)

    # 7. Fallback to rule-based if ML completely underperforms
    ml_count = int((hotspot_ml==2).sum()) + int((hotspot_ml==1).sum())
    if ml_count < 10:
        print(f"  ML underperformed — using rule_hot fallback")
        hotspot_final = rule_hot
    else:
        hotspot_final = hotspot_ml

    # 8. Save rasters (risk has NODATA=-9999 outside village)
    risk_export = hotspot_final.copy().astype(np.float32)
    risk_export[dtm_arr == NODATA] = NODATA
    save_cog(name, risk_export,  'risk',       profile)
    save_cog(name, confidence,   'confidence', profile)
    save_cog(name, slope,        'slope',      profile)
    save_cog(name, twi,          'twi',        profile)
    save_cog(name, accum,        'accum',      profile)
    del risk_export; gc.collect()

    # 9. Drainage network (gravity-correct Dijkstra)
    print(f"\n[{name}] Drainage Network Design")
    crs        = f'EPSG:{cfg["epsg"]}'
    drains_gdf = design_drainage_network(
        name, hotspot_final, stream_raster,
        dtm_arr, accum, transform, crs)

    # 10. Vectorize hotspots + streams
    records = []
    for code, label in {1:'Medium', 2:'High'}.items():
        mask = (hotspot_final == code).astype(np.uint8)
        for geom, val in shapes(mask, transform=transform):
            if val == 1:
                g = shape(geom)
                records.append({'geometry': g, 'risk_level': label,
                                'risk_code': code,
                                'area_m2': round(g.area,2),
                                'village': name})
    hotspots_gdf = gpd.GeoDataFrame(records, crs=crs)

    slope_clean = np.nan_to_num(slope, nan=0)
    stream_phys = ((stream_raster > 0) & (slope_clean > 0.01)).astype(np.uint8)
    stream_geoms = [shape(g) for g,v
                    in shapes(stream_phys, transform=transform) if v == 1]
    streams_gdf = gpd.GeoDataFrame({
        'geometry':      stream_geoms,
        'feature_type':  'drainage_channel',
        'physics_valid': True,
        'village':       name
    }, crs=crs)

    # 11. Export OGC GeoPackage
    gpkg = f'{OUT_BASE}/gpkg/{name}_drainage.gpkg'
    streams_gdf.to_file(gpkg,  layer='streams',  driver='GPKG')
    hotspots_gdf.to_file(gpkg, layer='hotspots', driver='GPKG')
    if len(drains_gdf) > 0:
        drains_gdf.to_file(gpkg, layer='proposed_drains', driver='GPKG')

    # 12. Per-village PNG report
    print(f"\n[{name}] Generating Village Report")
    generate_village_report(
        name, cfg, dtm_arr, slope, accum, twi,
        rule_hot, hotspot_final, drains_gdf,
        streams_gdf, hotspots_gdf, transform)

    print(f"\n[{name}] ── Output Summary ──────────────────────")
    print(f"  Streams         : {len(streams_gdf):,}")
    print(f"  Hotspots        : {len(hotspots_gdf):,}")
    print(f"  Proposed drains : {len(drains_gdf)}")
    print(f"  GPKG            : {gpkg}")

    del rule_hot, slope, accum, twi, dtm_c, slope_rad; gc.collect()

    return (dtm_entry,
            (slope_clean, confidence, hotspot_final, stream_raster),
            (streams_gdf, hotspots_gdf, drains_gdf),
            (xgb_model, xgb_scaler))


print("All pipeline functions defined")
