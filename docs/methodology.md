# GramDrain - Full Technical Methodology

This document describes every algorithm, parameter, and design decision in the GramDrain pipeline in detail.
For the submission summary, see `submission_form.docx`. For setup instructions, see the root `README.md`.

---

# Stage 1 - LiDAR Ingestion & Memory-Adaptive Ground Classification

# 1a. File I/O and Coordinate Type Detection

Raw point cloud files (`.las` / `.laz`, up to 11.6 GB) are opened in streaming mode using `laspy[lazrs]`
without loading all points into RAM. Only the LAS header is read first to extract:
- `point_count` (9.8M to 1.65B points across 10 villages)
- Bounding box `[xmin, xmax, ymin, ymax]`
- Coordinate type: if `max(x_range, y_range) < 2.0` → **Geographic (WGS84)**; else → **Projected (UTM)**

# 1b. Dynamic Memory Scaling - `_compute_dynamic_params()`

Before reading any points, two critical parameters are derived from the header alone:

**VOXEL_RES** — horizontal voxel size in native coordinate units.
- Convert range to metres: `x_range_m = x_range × 111,000 × cos(lat_mid)` (geographic files)
- `min_voxel_m = max(x_range_m, y_range_m) / 4,000` — keeps grid ≤ 4,000 × 4,000 = 16M cells
- Snap to nearest candidate: `[0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00]` m

**STRIDE** — point sub-sampling ratio:
- `stride_dict = ⌈worst_cells / 33,000,000⌉`
- `stride_pts  = ⌈total_pts  / 330,000,000⌉`
- `STRIDE = min(max(stride_dict, stride_pts), 50)`

# 1c. NumPy Grid Accumulator

Three pre-allocated arrays `(grid_rows × grid_cols)`:
- `min_z: float32` — running minimum elevation per cell
- `min_x, min_y: float64` — corresponding XY coordinates

Per 2M-point chunk:
1. Stride decimation: `keep = np.arange(0, n_chunk, STRIDE)`
2. Cell indices: `xi = clip((x − xmin) / VOXEL_RES, 0, cols−1)`
3. Flat index: `flat = yi × grid_cols + xi`
4. Intra-chunk dedup: `np.lexsort((z, flat))` then `np.unique(flat, return_index=True)` → minimum-z per cell
5. Cross-chunk update: `update_mask = z_u < min_z[ri, ci]` → fancy-index write

Memory: 20 B/cell × 16M cells = **305 MB** (vs ~4.9 GB Python dict equivalent).

Final extraction: `np.column_stack([min_x[valid_mask], min_y[valid_mask], min_z[valid_mask]])` — single-pass, no spike.

# 1d. Reprojection

Geographic files (PIRAYANKUPPAM, THANDALAM) are reprojected WGS84 → UTM using `pyproj.Transformer`
**after** downsampling, not on the raw point cloud.

# 1e. Cloth Simulation Filter (CSF)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `cloth_resolution` | 0.5 m | Matches SVAMITVA drone density |
| `rigidness` | 3 | Flat–moderate Indian rural terrain |
| `time_step` | 0.65 | Stable simulation convergence |
| `class_threshold` | 0.5 m | Ground membership tolerance |
| `iterations` | 500 | Convergence at village scale |
| `bSloopSmooth` | True | Slope transition smoothing |

---

# Stage 2 - DTM Generation

1. **Min-z binning** at 1 m resolution into a `float32` grid
2. **Weighted void fill**: `dtm = uniform_filter(dtm_f, 5) / max(uniform_filter(valid_mask, 5), 1e-6)`
3. **Noise removal**: `median_filter(size=3)`
4. **Smooth interpolation**: `gaussian_filter(sigma=1)`

Output: OGC Cloud-Optimised GeoTIFF — LZW compressed, 256×256 tiled, `predictor=2`, NODATA=−9999.

---

# Stage 3 - Terrain Derivatives

# Slope
Primary: WhiteboxTools `slope()` (Horn's 3×3 finite-difference, degrees).
Fallback: `np.gradient` + `arctan(sqrt(dx² + dy²))`.

# Multi-Scale Flow Accumulation
```
r5  = DTM − minimum_filter(DTM, 5)
r21 = DTM − minimum_filter(DTM, 21)
r51 = DTM − minimum_filter(DTM, 51)
accum = 0.2·exp(−r5/σ5) + 0.3·exp(−r21/σ21) + 0.5·exp(−r51/σ51)
```
Followed by `gaussian_filter(sigma=3)`. Stream channels = top 3rd percentile of accumulation.

# Topographic Wetness Index (TWI)
```
TWI = ln((accum × 100 + 1) / (tan(slope_rad) + 1e-6))
```

---

# Stage 4 - Waterlogging Hotspot Classification

# 12-Feature Matrix

| # | Feature | Window | Physical meaning |
|---|---------|--------|-----------------|
| 1 | Elevation | Point | Absolute height |
| 2 | Slope | Point | Local steepness |
| 3 | Flow accumulation | Point | Upslope drainage |
| 4 | TWI | Point | Wetness potential |
| 5 | Laplacian curvature | Point | Concavity / convexity |
| 6 | Distance to outlet | Point | Drainage topology |
| 7 | Mean elevation | 5×5 | Neighbourhood context |
| 8 | Std elevation | 5×5 | Micro-roughness |
| 9 | Mean slope | 5×5 | Neighbourhood gradient |
| 10 | Mean TWI | 5×5 | Neighbourhood wetness |
| 11 | Mean accumulation | 11×11 | Meso drainage density |
| 12 | Elevation range | 11×11 | Depression depth |

# Rule-Based Labels (pseudo ground-truth)
```
risk_rule = 0.40·norm(TWI) + 0.35·norm(accum) + 0.15·(1−norm(slope)) + 0.10·(1−norm(z))
Class 2 (High)   : risk_rule > 0.92
Class 1 (Medium) : 0.78 ≤ risk_rule ≤ 0.92
Class 0 (Safe)   : < 0.78
```

# XGBoost Hyperparameters
| Parameter | Value |
|-----------|-------|
| `n_estimators` | 150 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `scale_pos_weight` | 3 |
| `eval_metric` | `mlogloss` |
| Training samples | ≤ 200,000 px |
| Inference chunk | 500,000 px |

# Morphological Post-Processing
- `binary_opening(3×3)` — removes isolated noise pixels
- `binary_closing(5×5)` — fills holes within hotspot clusters
- Applied per class (High → Medium) with NODATA masking

---

# Stage 5 - Gravity-Corrected Drainage Network Design

# Cost Surface
```
cost[r,c] = (1 − norm_accum[r,c]) × 60 + norm_elev[r,c] × 40 + 1
NODATA cells → cost = 9,999  (impenetrable boundary)
```

# Dijkstra Routing (8-connectivity)
- Source: centre of mass of each High-risk cluster (≥ 25 px)
- Target: nearest stream-channel cell (top 3rd percentile accumulation)
- Diagonal cost: `1.414 × cost_cell`; Cardinal: `1.0 × cost_cell`
- Up to 50 clusters routed per village

# Vector Output (GeoPackage layers)
| Layer | Geometry | Key fields |
|-------|----------|------------|
| `proposed_drains` | LineString | cluster_id, length_m, hotspot_area |
| `hotspots` | Polygon | risk_level, risk_code, area_m2 |
| `streams` | Polygon | feature_type, physics_valid |
