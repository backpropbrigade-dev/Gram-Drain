# Understanding the GramDrain Outputs

Every village processed by the pipeline produces a set of files across six folders.
This guide explains what each file contains, what the values mean, and how to open them.

---

# Complete Output Structure

'''
mopr_outputs/
├── ground/
│   └── {VILLAGE}_ground.las
├── dtm/
│   └── {VILLAGE}_dtm_cog.tif
├── rasters/
│   ├── {VILLAGE}_risk_cog.tif
│   ├── {VILLAGE}_slope_cog.tif
│   ├── {VILLAGE}_twi_cog.tif
│   ├── {VILLAGE}_accum_cog.tif
│   └── {VILLAGE}_confidence_cog.tif
├── gpkg/
│   └── {VILLAGE}_drainage.gpkg
└── village_reports/
    └── {VILLAGE}_report.png

"""

Replace `{VILLAGE}` with the village name — e.g. `DEVDI`, `KADAMTALA_RNG`, `THANDALAM`.

---

# 1. Ground-Classified LAS — `ground/{VILLAGE}_ground.las`

**What it is:** Bare-earth points extracted from raw LiDAR by the Cloth Simulation Filter (CSF).

**Format:** LAS 1.2, Point Format 2 | **Classification:** 2 (ASPRS bare-earth) | **Scale:** 0.001 m

**What to look for:**
- Dense, gap-free coverage = good ground extraction
- Large holes = vegetation or building areas (expected and normal)
- Isolated low-flying outliers = rare CSF misclassifications (acceptable)

**How to open:**
- **CloudCompare** → File → Open → colour by Z
- **QGIS Point Cloud plugin** (QGIS 3.26+) → drag and drop

---

# 2. Digital Terrain Model — `dtm/{VILLAGE}_dtm_cog.tif`

**What it is:** Bare-earth elevation rasterised at 1 m × 1 m. Voids are filled with weighted
neighbourhood average, then smoothed with median filter (size 3) and Gaussian filter (σ=1).

**Format:** Cloud-Optimised GeoTIFF (COG), `float32` | **Resolution:** 1 m | **NODATA:** −9999

| Value | Meaning |
|-------|---------|
| −9999 | Outside data extent or unfilled void |
| Positive float | Elevation above datum in metres |

**Open in QGIS:**
1. Drag `.tif` onto canvas
2. Properties → Symbology → Singleband pseudocolor → Colour ramp: `Terrain`

**Open in Python:**
```python
import rasterio
import numpy as np

with rasterio.open("DEVDI_dtm_cog.tif") as src:
    dtm = src.read(1)
    transform = src.transform

valid = dtm[dtm != -9999]
print(f"Elevation range: {valid.min():.2f} m — {valid.max():.2f} m")
```

---

# 3. Waterlogging Risk Raster — `rasters/{VILLAGE}_risk_cog.tif`

**What it is:** The primary planning output. Each pixel holds a waterlogging risk class from the
XGBoost classifier after morphological cleanup.

**Format:** COG, `float32` | **NODATA:** −9999

| Value | Class | Colour | Meaning |
|-------|-------|--------|---------|
| −9999 | Outside boundary | Transparent | No data |
| 0 | Safe / Dry | Light green `#D1F0C7` | No significant risk |
| 1 | Medium risk | Orange `#FF8C00` | May waterlog in heavy rain |
| 2 | High risk | Red `#CC0000` | Strong convergence of low elevation + high TWI + high accumulation |

**Style in QGIS:**
1. Properties → Symbology → Paletted / Unique Values → Classify
2. Assign colours from the table above

**Read in Python:**
```python
import rasterio

with rasterio.open("DEVDI_risk_cog.tif") as src:
    risk = src.read(1)

high_px   = int((risk == 2).sum())
medium_px = int((risk == 1).sum())
print(f"High risk  : {high_px:,} px = {high_px / 1e4:.2f} ha")
print(f"Medium risk: {medium_px:,} px = {medium_px / 1e4:.2f} ha")
```

---

# 4. XGBoost Confidence Raster — `rasters/{VILLAGE}_confidence_cog.tif`

**What it is:** Raw XGBoost probability for class 2 (High risk) — `predict_proba[:, 2]` reshaped to raster.

**Format:** COG, `float32`, range 0.0 – 1.0 | **NODATA:** −9999

| Confidence | Meaning |
|-----------|---------|
| > 0.80 | Strong High-risk prediction — high planning confidence |
| 0.50 – 0.80 | Borderline — cross-check with TWI and slope before acting |
| < 0.50 | Not flagged as High risk |

Use this raster to prioritise drain construction — high-confidence zones warrant immediate
action; borderline zones benefit from field inspection first.

---

# 5. Slope Raster — `rasters/{VILLAGE}_slope_cog.tif`

**What it is:** Terrain slope in degrees via WhiteboxTools Horn 3×3 finite-difference (fallback: `np.gradient`).

**Format:** COG, `float32` | **Units:** Degrees

| Slope (°) | Terrain type |
|----------|-------------|
| 0 – 2 | Flat — highest waterlogging risk |
| 2 – 5 | Gentle — moderate risk |
| 5 – 15 | Moderate — usually drains naturally |
| > 15 | Steep — low waterlogging risk |

---

# 6. TWI Raster — `rasters/{VILLAGE}_twi_cog.tif`

**What it is:** Topographic Wetness Index — dimensionless index combining upslope area and slope.

**Formula:** `TWI = ln((accum × 100 + 1) / (tan(slope_rad) + 1e-6))`

| TWI | Meaning |
|-----|---------|
| > 12 | Very high wetness potential — valley floors and depressions |
| 8 – 12 | Moderate wetness |
| < 8 | Low wetness — well-drained slopes |

High TWI + low slope + high accumulation is the core trigger for XGBoost High-risk classification.

---

# 7. Flow Accumulation Raster — `rasters/{VILLAGE}_accum_cog.tif`

**What it is:** Multi-scale drainage convergence proxy from nested minimum filters at 5, 21, and 51-pixel radii.

**Formula:**
```
accum = 0.2·exp(−r5/σ5) + 0.3·exp(−r21/σ21) + 0.5·exp(−r51/σ51)
```

- Values near **1.0** → valley bottoms, strong convergence
- Values near **0.0** → ridges and hilltops

The **stream channel layer** is the top 3rd percentile of accumulation — these are the Dijkstra
routing targets for proposed drain outfalls.

---

# 8. Drainage GeoPackage — `gpkg/{VILLAGE}_drainage.gpkg`

**What it is:** A single OGC GeoPackage with three vector layers. All layers use the village UTM CRS.

---

# Layer: `proposed_drains`

**Geometry:** LineString — gravity-fed drain alignment from each hotspot centroid to nearest stream cell.

| Field | Type | Description |
|-------|------|-------------|
| `cluster_id` | int | Hotspot cluster index |
| `hotspot_area` | int | High-risk pixels in the source cluster |
| `length_m` | float | Proposed drain length in metres |
| `village` | str | Village name |
| `type` | str | `proposed_drain` |

**Planning tips:**
- Sort by `hotspot_area` descending to prioritise the largest at-risk zones
- `length_m` gives estimated earthwork length for cost estimation
- All alignments are gravity-fed — no pumping required

---

# Layer: `hotspots`

**Geometry:** Polygon — contiguous risk zones after morphological cleanup.

| Field | Type | Description |
|-------|------|-------------|
| `risk_level` | str | `High` or `Medium` |
| `risk_code` | int | 2 = High, 1 = Medium |
| `area_m2` | float | Area in square metres |
| `village` | str | Village name |

Divide `area_m2` by 10,000 to get hectares.

---

# Layer: `streams`

**Geometry:** Polygon — natural drainage channel cells (top 3rd percentile accumulation, slope > 0.01°).

| Field | Type | Description |
|-------|------|-------------|
| `feature_type` | str | `drainage_channel` |
| `physics_valid` | bool | True = slope-validated channel |
| `village` | str | Village name |

These are the outfall targets where proposed drains discharge.

---

# Load in QGIS

1. Layer → Add Layer → Add Vector Layer → browse to `.gpkg`
2. Pick a layer from the dropdown
3. Repeat for all three layers
4. Overlay on the risk raster for a complete planning view

# Load in Python

```python
import geopandas as gpd

gpkg = "DEVDI_drainage.gpkg"

drains   = gpd.read_file(gpkg, layer="proposed_drains")
hotspots = gpd.read_file(gpkg, layer="hotspots")
streams  = gpd.read_file(gpkg, layer="streams")

total_km     = drains["length_m"].sum() / 1000
high_area_ha = hotspots[hotspots.risk_level == "High"]["area_m2"].sum() / 10000

print(f"Total drain length : {total_km:.2f} km")
print(f"High-risk area     : {high_area_ha:.2f} ha")
```

---

# 9. Village Report PNG — `village_reports/{VILLAGE}_report.png`

**What it is:** Auto-generated 6-panel diagnostic figure (20 × 14 in, 120 DPI).

| Panel | Content |
|-------|---------|
| Top-left | DTM — terrain colour ramp with elevation scale bar |
| Top-right | Risk map + proposed drains (RGBA overlay, cyan dashed lines) |
| Mid-left | Confusion matrix — XGBoost vs rule-based labels (normalised) |
| Mid-right | Classification report — Precision, Recall, F1 per class |
| Bottom | Summary — total LiDAR points, survey area, risk counts, drain length |

**Colour coding in the risk panel:**
- Light green — safe / dry
- Orange — medium risk
- Red — high risk
- Cyan dashed — proposed drain alignment

---

# Quick Reference

| I want to… | File to use |
|-----------|------------|
| Find where to build drains | `proposed_drains` layer in `.gpkg` |
| Map all high-risk zones | `hotspots` layer in `.gpkg` or `_risk_cog.tif` |
| Estimate earthwork length | `length_m` in `proposed_drains` |
| Upload to BhuNaksha / QGIS | `_dtm_cog.tif`, `_risk_cog.tif` (COG, directly compatible) |
| Check ML accuracy | Confusion matrix panel in `_report.png` |
| Check model certainty per pixel | `_confidence_cog.tif` |
| Inspect bare-earth points | `_ground.las` in CloudCompare |
| Validate drain logic manually | `_slope_cog.tif` + '_twi_cog.tif' + 'streams' layer |
'''
