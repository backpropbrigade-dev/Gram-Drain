# Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│   SVAMITVA LiDAR (.las / .laz)  │  Village config (EPSG, res, path) │
└───────────────────────┬──────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — GROUND CLASSIFICATION                                     │
│                                                                      │
│  ┌─────────────────────┐     ┌────────────────────────────────────┐  │
│  │  Header read only   │────▶│  _compute_dynamic_params()         │  │
│  │  (no full load)     │     │  VOXEL_RES + STRIDE from metadata  │  │
│  └─────────────────────┘     └──────────────┬─────────────────────┘  │
│                                             │                        │
│                              ┌──────────────▼─────────────────────┐  │
│                              │  NumPy Grid Accumulator            │  │
│                              │  min_z[rows,cols] float32          │  │
│                              │  min_x[rows,cols] float64          │  │
│                              │  min_y[rows,cols] float64          │  │
│                              │  Per-chunk: lexsort→unique→mask    │  │
│                              └──────────────┬─────────────────────┘  │
│                                             │                        │
│                              ┌──────────────▼─────────────────────┐  │
│                              │  Cloth Simulation Filter (CSF)     │  │
│                              │  cloth_res=0.5  rigidness=3        │  │
│                              │  threshold=0.5  iters=500          │  │
│                              └──────────────┬─────────────────────┘  │
└─────────────────────────────────────────────┼────────────────────────┘
                                              │ ground_pts (N×3)
                                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — DTM GENERATION  (1 m resolution)                         │
│                                                                      │
│  Min-z bin → weighted void-fill → median_filter → gaussian_filter   │
│  Output: OGC COG GeoTIFF (LZW, tiled 256×256, NODATA=−9999)        │
└─────────────────────────────────────────────┬────────────────────────┘
                                              │ dtm_arr
                                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — TERRAIN DERIVATIVES                                       │
│                                                                      │
│  Slope          WhiteboxTools Horn 3×3 (fallback: np.gradient)      │
│  Flow accum     0.2·r5 + 0.3·r21 + 0.5·r51 (nested min-filters)    │
│  TWI            ln((accum×100+1) / (tan(slope)+1e-6))               │
└─────────────────────────────────────────────┬────────────────────────┘
                                              │ slope, accum, twi
                                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — HOTSPOT CLASSIFICATION                                    │
│                                                                      │
│  12-feature matrix (point + 5×5 + 11×11 windows)                   │
│        │                                                             │
│        ├──▶ Rule labels   0.40·TWI + 0.35·accum + 0.15·slope + ...  │
│        │    Thresholds    >0.92 → High  |  0.78–0.92 → Medium       │
│        │                                                             │
│        └──▶ XGBoost       150 trees  depth=5  scale_pos_weight=3    │
│             Per-village retrain on ≤200k sampled pixels             │
│                                                                      │
│  Morphological cleanup:  opening(3×3)  →  closing(5×5)             │
└─────────────────────────────────────────────┬────────────────────────┘
                                              │ hotspot_final
                                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — DRAINAGE NETWORK DESIGN                                   │
│                                                                      │
│  Cost = (1−accum)×60 + elev×40 + 1   (NODATA=9999)                 │
│  Hotspot clusters  →  scipy_label  →  centre_of_mass               │
│  Dijkstra (8-conn)  →  nearest stream cell                          │
│  Path  →  LineString  →  GeoPackage layer                           │
└─────────────────────────────────────────────┬────────────────────────┘
                                              │
                                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  OUTPUT LAYER                                                        │
│                                                                      │
│  ground/{name}_ground.las           LAS 1.2, class=2                │
│  dtm/{name}_dtm_cog.tif             COG GeoTIFF float32             │
│  rasters/{name}_risk_cog.tif        Risk classes 0/1/2              │
│  rasters/{name}_slope_cog.tif       Slope degrees                   │
│  rasters/{name}_twi_cog.tif         TWI                             │
│  rasters/{name}_confidence_cog.tif  XGBoost P(High)                 │
│  gpkg/{name}_drainage.gpkg          proposed_drains + hotspots      │
│  village_reports/{name}_report.png  DTM + risk + CM + metrics       │
└──────────────────────────────────────────────────────────────────────┘
```

## Memory Profile

| Component | Peak RAM |
|-----------|----------|
| NumPy Grid Accumulator (4000×4000) | 305 MB |
| Python dict equivalent | ~4.9 GB |
| XGBoost training (200k × 12) | ~150 MB |
| DTM (10,000×10,000 float32) | ~400 MB |
| Inference chunk (500k × 12) | ~50 MB |
| **Total safe budget** | **< 12 GB** |

## Village Routing Decision

```python
if total_pts >= 50_000_000:
    classify_ground_chunked()   # NumPy Grid Accumulator
else:
    classify_ground_regular()   # Full load + voxel dedup
```
