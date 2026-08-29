# GramDrain - AI/ML Waterlogging Detection & Gravity-Based Drainage Planning for Gram Panchayats

> **Ministry of Panchayati Raj - Geospatial Intelligence Challenge**

---

# Overview

**GramDrain** is a fully automated, end-to-end geospatial intelligence pipeline that converts raw village-scale LiDAR point clouds (`.las` / `.laz`) into government-ready drainage planning deliverables. The system identifies waterlogging-prone zones and proposes gravity-correct drain alignments — all without manual GIS intervention.

The pipeline processes point clouds ranging from **9.8 million to 1.65 billion points** across **10 Gram Panchayat villages** (SVAMITVA programme data), producing OGC-compliant GeoTIFF and GeoPackage outputs compatible with BhuNaksha, QGIS, and ArcGIS.

---

# The Problem

Recurring waterlogging in rural Gram Panchayats causes crop loss, property damage, and health hazards. Village-level, terrain-informed drainage planning is difficult to scale because:

- Manual identification of low-lying zones is slow and error-prone
- Existing tools require skilled GIS operators and large RAM machines
- Raw SVAMITVA LiDAR files are too large for standard workflows (up to 11.6 GB, 1.65B points)
- No automated pipeline existed for translating LiDAR → actionable drainage maps

---

# Solution — Five-Stage Pipeline

```
LiDAR (.las/.laz)
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Ground            │  CSF + NumPy Grid Accumulator
│  Classification             │  Dynamic memory scaling
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Stage 2: DTM Generation    │  1 m resolution GeoTIFF (COG)
│                             │  Void-fill + Gaussian smooth
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Stage 3: Terrain           │  Slope, multi-scale flow
│  Derivatives                │  accumulation, TWI
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Stage 4: XGBoost           │  12-feature matrix
│  Hotspot Detection          │  Per-village retrain + morphological cleanup
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Stage 5: Drainage          │  Gravity-corrected Dijkstra routing
│  Network Design             │  GeoPackage vector output
└─────────────────────────────┘
```

---

# Key Technical Innovations

# NumPy Grid Accumulator (16× RAM reduction)
The standard approach of storing voxel min-z values in a Python dictionary caused crashes on the 1.65B-point Kadamtala file (~4.9 GB heap). Replaced with three pre-allocated NumPy arrays (`min_z: float32`, `min_x: float64`, `min_y: float64`) of shape `(grid_rows, grid_cols)`:

- **Dict approach**: 320 bytes/entry × 32M entries = **4.9 GB** + 0.5 GB spike on extraction
- **Grid approach**: 20 bytes/cell × 16M cells = **305 MB** flat, freed immediately after extraction

Intra-chunk deduplication is fully vectorised using `np.lexsort + np.unique` — no Python-level loops.

# Dynamic Memory Scaling
`_compute_dynamic_params()` reads only the LAS file header (not the points) and auto-computes:
- `VOXEL_RES` — snapped to the nearest standard voxel size from `[0.05 → 10.00]` metres, ensuring grid ≤ 4,000 × 4,000 cells
- `STRIDE` — sub-sampling ratio derived from both grid density and total point count constraints

This means **zero manual parameter tuning** across files spanning 9.8M to 1.65B points.

# 12-Feature XGBoost Classifier
Per-village retraining with features spanning three spatial scales:

| Scale | Features |
|-------|----------|
| Point | Elevation, slope, flow accumulation, TWI, Laplacian curvature, distance to outlet |
| 5 × 5 window | Mean elevation, std elevation, mean slope, mean TWI |
| 11 × 11 window | Mean flow accumulation, elevation range |

Pseudo-ground-truth labels are generated from a physics-based terrain rule, then used to train a village-specific XGBoost model that generalises beyond simple thresholding.

# Gravity-Correct Dijkstra Routing
Proposed drain alignments follow natural watercourses using a composite cost surface:

'''
cost = (1 − norm_accum) × 60 + norm_elev × 40 + 1
'''

Low-cost paths run along valley floors (high accumulation + low elevation), guaranteeing all proposed drains are **gravity-fed** and never cross village boundaries (NODATA cells = cost 9,999).

---

# Dataset - 10 Villages Processed

| Village | State | Points | Size | Coord System | EPSG |
|---------|-------|--------|------|--------------|------|
| DEVDI | MP | 64,622,538 | 1.87 GB | Projected | 32643 |
| KHAPRETA | MP | 163,743,261 | 1.62 GB | Projected | 32643 |
| PIRAYANKUPPAM | TN | 157,925,322 | 4.74 GB | Geographic | 32644 |
| DHUNDA | Punjab | 172,862,229 | 1.64 GB | Projected | 32643 |
| DHAL | Punjab | 23,431,282 | 680 MB | Projected | 32643 |
| THANDALAM | TN | 188,077,336 | 5.64 GB | Geographic | 32644 |
| REFLIGHT_64334 | — | 57,635,469 | 1.67 GB | Projected | 32643 |
| CHAKHIRASINGH | — | 9,839,175 | 256 MB | Projected | 32643 |
| GANDHINAGAR_DIG | A&N | 287,661,850 | 2.04 GB | Projected | 32646 |
| KADAMTALA_RNG | A&N | 1,650,723,422 | 11.60 GB | Projected | 32646 |

---

# Outputs Per Village

| Output | Format | Description |
|--------|--------|-------------|
| Ground points | `.las` (LAS 1.2, class 2) | Bare-earth extracted points |
| DTM | `.tif` (COG, float32, 1 m) | Digital Terrain Model |
| Risk raster | `.tif` (COG, uint8) | 0=Safe, 1=Medium, 2=High risk |
| Slope raster | `.tif` (COG, float32) | Terrain slope in degrees |
| TWI raster | `.tif` (COG, float32) | Topographic Wetness Index |
| Confidence raster | `.tif` (COG, float32) | XGBoost prediction confidence |
| Drainage GeoPackage | `.gpkg` (3 layers) | Proposed drains + hotspot polygons + stream channels |
| Village report | `.png` (2400 × 1680 @ 120 DPI) | DTM, risk map, confusion matrix, ML classification report table, stats summary bar |

---
Repo structure
> **Note:** Raw LiDAR files (`.las` / `.laz`) and full-resolution raster outputs are stored on Google Drive, not in this repository. See *Running the Pipeline* below.

---

# Running the Pipeline

# Requirements
- Google Colab (High-RAM runtime, ≥ 52 GB recommended)
- Google Drive with LiDAR input files mounted at `/content/drive/MyDrive/hackathon_data/`

## Streamlit Dashboard

A browser-based interactive calculator is included for instant pipeline parameter estimation without running the full pipeline.

### Run locally
```bash
pip install -r requirements.txt
streamlit run lidar_calculator.py
```

Upload any `.las` / `.laz` file → parameters are auto-detected from the 400-byte header → no full file load required. The dashboard also supports triggering the full pipeline and downloading all outputs.

# Setup

Open `notebooks/IITT.ipynb` in Google Colab, then run cells in order:

**Cell 1** — Install system dependencies and Python packages:
```
laspy[lazrs]  rasterio  geopandas  shapely  xgboost  scikit-learn
pyproj  whitebox  cloth-simulation-filter  networkx  matplotlib  tqdm
```

**Cell 2** — Mount Google Drive and verify all imports.

**Cell 3** — Register village configurations (file paths, EPSG codes, resolution).

**Cell 4** — Inspect point counts and coordinate types for all villages.

**Cell 5** — Define all pipeline functions (ground classification, DTM, features, XGBoost, routing).

**Cells 6–11** — Process each village. The auto-detector selects `classify_ground_regular` or `classify_ground_chunked` based on point count:
- `< 50M points` → regular (full load into RAM)
- `≥ 50M points` → chunked NumPy grid accumulator

**Cell 12** — Verify all output files exist and are OGC-compliant.

---

## Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Point cloud I/O | `laspy[lazrs]` | ≥ 2.5.4 |
| Ground classification | `cloth-simulation-filter` (CSF) | ≥ 1.1.7 |
| Raster I/O & COG export | `rasterio` | ≥ 1.4.0 |
| Vector output | `geopandas`, `shapely` | ≥ 1.0.1 |
| ML classification | `xgboost`, `scikit-learn` | ≥ 2.1.3 / ≥ 1.6.1 |
| Terrain derivatives | `scipy.ndimage`, `whitebox` | ≥ 1.15.0 / ≥ 2.3.4 |
| Coordinate reprojection | `pyproj` | ≥ 3.7.0 |
| Drainage routing | `heapq` (Dijkstra) | stdlib |
| Numerical core | `numpy` | ≥ 2.0.0 |
| Visualisation | `matplotlib`, `plotly` | — / ≥ 5.24.0 |
| Graph algorithms | `networkx` | ≥ 3.4.2 |
| Dashboard / UI | `streamlit` | ≥ 1.41.0 |
| Runtime — pipeline | Google Colab High-RAM (≥ 52 GB) | — |
| Runtime — dashboard | Local / Streamlit Cloud | — |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/ARCHITECTURE.md) | System design and pipeline flow |
| [Methodology](docs/methodology.md) | Algorithm details, CSF parameters, XGBoost features |
| [Outputs Guide](docs/outputs_guide.md) | All output formats, file naming, OGC compliance |
| [Dashboard Example — DHAL](docs/DashboardExample_DHAL.pdf) | Sample Streamlit dashboard report for DHAL village |

# CSF Ground Classification Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `cloth_resolution` | 0.5 m | Matches SVAMITVA drone point density |
| `rigidness` | 3 | Flat-to-moderate Indian rural terrain |
| `time_step` | 0.65 | Stable simulation timestep |
| `class_threshold` | 0.5 m | Height tolerance for ground membership |
| `iterations` | 500 | Sufficient convergence for village-scale extent |
| `bSloopSmooth` | True | Post-processing smoothing for slope transitions |

---

# Impact

- **Environmental**: Identifies drainage bottlenecks before monsoon season, reducing flood damage
- **Social**: Enables Gram Panchayats to plan drainage works without GIS expertise
- **Economic**: Reduces waterlogging crop loss; prioritises infrastructure spending
- **Governance**: Produces OGC-standard deliverables directly loadable into BhuNaksha and SVAMITVA portals
- **Scalability**: The dynamic parameter system processes any SVAMITVA village automatically — from single-hectare hamlets to multi-kilometre township extents

---

*Built for the Ministry of Panchayati Raj Geospatial Intelligence Challenge.*
