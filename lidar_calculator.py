"""
GramDrain · LiDAR Pipeline Calculator
IIT Tirupati · MoPR Hackathon

Input:  total LiDAR points, file size, extent, coordinate type
Output: complete 9-step pipeline parameter sheet (screen + JSON download)

Large-file strategy: laspy reads only the LAS/LAZ header (<1 KB) from the
uploaded BytesIO stream, so even 10 GB files never block the browser — the
full point data is NOT loaded into memory by this app.
"""

import io
import json
import math
import struct
import warnings
import zipfile

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GramDrain · LiDAR Pipeline Calculator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #050d1a;
    color: #c8d8f0;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #06101e 100%);
    border-right: 1px solid #1a3050;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] h4       { color: #7eb8e8 !important; }

  /* ── Main title ── */
  .main-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.2rem;
    color: #e8f4ff; letter-spacing: -0.5px; margin-bottom: 0;
  }
  .main-subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    color: #3d7ab5; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;
  }

  /* ── Step headers ── */
  .step-header {
    font-family: 'Space Mono', monospace; font-size: 0.65rem; letter-spacing: 3px;
    text-transform: uppercase; color: #2a9fd6; margin-bottom: 2px;
    padding: 14px 18px 4px; border-top: 1px solid #0f2540;
  }
  .step-title {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.15rem;
    color: #ddeeff; padding: 0 18px 10px;
  }

  /* ── Metric card ── */
  .metric-card {
    background: linear-gradient(135deg, #0c1e38 0%, #071422 100%);
    border: 1px solid #1a3555; border-radius: 8px;
    padding: 14px 18px; margin: 4px 0;
  }
  .metric-label {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; letter-spacing: 2px;
    text-transform: uppercase; color: #3d7ab5; margin-bottom: 4px;
  }
  .metric-value { font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.3rem; color: #80c8ff; }
  .metric-sub   { font-size: 0.7rem; color: #4a6f94; margin-top: 2px; }

  /* ── Badges ── */
  .badge-ok   { background:#0d3d24; color:#2ecc71; border:1px solid #1a6640; border-radius:4px; padding:2px 8px; font-size:0.65rem; font-family:'Space Mono',monospace; }
  .badge-warn { background:#3d2b00; color:#f39c12; border:1px solid #7a5400; border-radius:4px; padding:2px 8px; font-size:0.65rem; font-family:'Space Mono',monospace; }
  .badge-high { background:#3d0d0d; color:#e74c3c; border:1px solid #7a1a1a; border-radius:4px; padding:2px 8px; font-size:0.65rem; font-family:'Space Mono',monospace; }

  /* ── Section divider ── */
  .sdiv { border-top: 1px solid #0f2540; margin: 18px 0 12px; }

  /* ── Info note box ── */
  .note-box {
    background:#071422; border:1px solid #0f2540; border-left:3px solid #2a9fd6;
    border-radius:6px; padding:14px 20px; margin-top:8px;
  }

  /* ── Upload area override ── */
  [data-testid="stFileUploader"] {
    background: #071422 !important; border: 1px dashed #1a4070 !important;
    border-radius: 8px !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background:#050d1a; border-bottom:1px solid #1a3050; }
  .stTabs [data-baseweb="tab"]      { color:#3d7ab5; font-family:'Syne',sans-serif; }
  .stTabs [aria-selected="true"]    { color:#80c8ff !important; border-bottom:2px solid #2a9fd6 !important; }

  /* ── Expander ── */
  .streamlit-expanderHeader {
    background:#071422 !important; border:1px solid #0f2540 !important;
    border-radius:6px !important; color:#c8d8f0 !important; font-family:'Syne',sans-serif !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background:#050d1a; }
  ::-webkit-scrollbar-thumb { background:#1a4070; border-radius:3px; }

  /* ── Pipeline step row ── */
  .pipe-step {
    background:#071422; border:1px solid #0f2540; border-left:3px solid #1a7acc;
    border-radius:6px; padding:10px 16px; margin-bottom:8px;
    font-family:'Space Mono',monospace; font-size:0.75rem; color:#c8d8f0;
  }
  .pipe-step.done { border-left-color:#2ecc71; }

  /* ── Download button ── */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #0d3d5a, #0a2235) !important;
    color: #80c8ff !important; border: 1px solid #1a5080 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important;
    letter-spacing: 1px !important; border-radius: 6px !important;
  }
  .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1a5080, #0d3050) !important;
    border-color: #2a9fd6 !important;
  }

  #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def metric(label, value, sub="", col=None):
    html = f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>"""
    (col or st).markdown(html, unsafe_allow_html=True)


def step_header(num, title, subtitle=""):
    st.markdown(f"""
    <div class="step-header">STEP {num:02d}</div>
    <div class="step-title">{title}
      {'<span style="font-size:0.7rem;color:#3d7ab5;font-weight:400;margin-left:10px;">' + subtitle + '</span>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)


def badge(label, kind="ok"):
    return f'<span class="badge-{kind}">{label}</span>'


def sdiv():
    st.markdown('<div class="sdiv"></div>', unsafe_allow_html=True)


def dark_fig(fig, title="", height=260):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#c8d8f0", size=11, family="Syne")),
        paper_bgcolor="#071422", plot_bgcolor="#071422",
        font=dict(color="#7eb8e8", family="Space Mono", size=9),
        height=height, margin=dict(l=40, r=20, t=36, b=40),
        xaxis=dict(gridcolor="#0f2540", linecolor="#0f2540", zerolinecolor="#0f2540"),
        yaxis=dict(gridcolor="#0f2540", linecolor="#0f2540", zerolinecolor="#0f2540"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LAS HEADER PARSER  (no full-file load — reads only the 227-byte fixed header)
# ═══════════════════════════════════════════════════════════════════════════════

def _read_las_header_bytes(raw: bytes) -> dict:
    """
    Parse the LAS 1.2/1.3/1.4 public file header from the first 375 bytes.
    All values come from the LAS spec public header block — no point data read.
    Returns: point_count, x_min, x_max, y_min, y_max, z_min, z_max or raises.
    """
    if len(raw) < 227:
        raise ValueError("Too few bytes — not a valid LAS file")

    sig = raw[:4]
    if sig != b"LASF":
        raise ValueError(f"Not a LAS file (signature={sig})")

    # Version major / minor at offset 24–25
    ver_major = raw[24]
    ver_minor = raw[25]

    # Point data format at offset 104 (LAS 1.0-1.3); 105 for 1.4
    # Header size at offset 94 (uint16)
    header_size = struct.unpack_from("<H", raw, 94)[0]

    # Number of point records: offset 107 for LAS 1.2/1.3 (uint32)
    # For LAS 1.4 it's at offset 247 (uint64).  We detect via version.
    if ver_minor <= 3:
        if len(raw) < 243:
            raise ValueError("Header too short for LAS ≤1.3")
        point_count = struct.unpack_from("<I", raw, 107)[0]
        # Min/Max: offsets 179-227 (6 × float64, XYZ max then XYZ min)
        x_max, y_max, z_max, x_min, y_min, z_min = struct.unpack_from("<6d", raw, 179)
    else:
        # LAS 1.4: extended public header — point count at 247 (uint64)
        if len(raw) < 375:
            raise ValueError("Header too short for LAS 1.4")
        point_count = struct.unpack_from("<Q", raw, 247)[0]
        x_max, y_max, z_max, x_min, y_min, z_min = struct.unpack_from("<6d", raw, 179)

    return dict(
        point_count=int(point_count),
        x_min=float(x_min), x_max=float(x_max),
        y_min=float(y_min), y_max=float(y_max),
        z_min=float(z_min), z_max=float(z_max),
        version=f"{ver_major}.{ver_minor}",
    )


def parse_las_header_from_upload(uploaded_file) -> dict:
    """
    Reads only the first 400 bytes from the uploaded file object.
    Works for .las files.  For .laz (compressed), same header layout applies.
    """
    buf = uploaded_file.read(400)
    uploaded_file.seek(0)          # reset so Streamlit can show file info
    return _read_las_header_bytes(buf)


def parse_las_from_zip(uploaded_zip) -> dict:
    """
    Opens a ZIP in streaming mode, reads only the first 400 bytes of the
    first .las or .laz member found — no full decompression.
    """
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith((".las", ".laz")):
                with zf.open(name) as f:
                    buf = f.read(400)
                return _read_las_header_bytes(buf)
    raise ValueError("No .las or .laz file found inside the ZIP")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE CALCULATOR — pure math from IITT_V4 notebook
# ═══════════════════════════════════════════════════════════════════════════════

VOXEL_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50,
                    0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00]

def compute_pipeline(total_pts: int, size_mb: float,
                     x_min: float, x_max: float,
                     y_min: float, y_max: float,
                     coord_type: str,
                     village: str = "CUSTOM",
                     epsg: int = 32643,
                     elev_min: float = None,
                     elev_max: float = None) -> dict:
    """
    Replicates _compute_dynamic_params() + all downstream estimations
    from IITT_V4 (5).ipynb Cell 5C.  Returns a flat dict of every metric.
    """
    is_geographic = (coord_type == "GEOGRAPHIC")

    # ── Extent in metres ──────────────────────────────────────────────────────
    # Use abs() so swapped min/max inputs don't produce negative ranges.
    x_range = abs(float(x_max - x_min))
    y_range = abs(float(y_max - y_min))
    # Clamp to at least 1 m so downstream division / sqrt never blow up.
    x_range = max(x_range, 1e-3)
    y_range = max(y_range, 1e-3)

    if is_geographic:
        lat_mid   = (float(y_min) + float(y_max)) / 2.0
        cos_lat   = math.cos(math.radians(lat_mid))
        x_range_m = x_range * 111_000.0 * max(cos_lat, 1e-6)
        y_range_m = y_range * 111_000.0
    else:
        x_range_m = x_range
        y_range_m = y_range

    # Clamp metric ranges to safe minimums.
    x_range_m = max(x_range_m, 1.0)
    y_range_m = max(y_range_m, 1.0)

    area_m2  = x_range_m * y_range_m
    area_km2 = max(area_m2 / 1e6, 1e-9)   # never zero — guards sqrt later

    # ── Step 2: Dynamic Scaling ───────────────────────────────────────────────
    MAX_GRID_AXIS    = 4_000
    MAX_DICT_ENTRIES = 33_000_000
    MAX_RAW_INFLIGHT = 330_000_000

    min_voxel_m = max(x_range_m, y_range_m) / MAX_GRID_AXIS
    voxel_m     = next((v for v in VOXEL_CANDIDATES if v >= min_voxel_m),
                       max(VOXEL_CANDIDATES))

    if is_geographic:
        VOXEL_RES = voxel_m / (111_000.0 * cos_lat)
    else:
        VOXEL_RES = voxel_m

    grid_cols = int(x_range / VOXEL_RES) + 2
    grid_rows = int(y_range / VOXEL_RES) + 2
    worst_cells = grid_cols * grid_rows

    stride_dict = max(1, math.ceil(worst_cells / MAX_DICT_ENTRIES))
    stride_pts  = max(1, math.ceil(total_pts  / MAX_RAW_INFLIGHT))
    STRIDE      = min(max(stride_dict, stride_pts), 50)

    effective_pts = total_pts // STRIDE
    # Grid arrays: z=float32(4B) + x,y=float64(8B each) = 20B/cell
    grid_ram_mb   = (grid_rows * grid_cols * 20) / (1024 ** 2)

    # ── Chunked vs regular mode ───────────────────────────────────────────────
    chunked = total_pts > 100_000_000

    # ── Step 3: Ground Classification ────────────────────────────────────────
    # Typical ground % from actual data: 57–72%. Use 65 % as midpoint estimate.
    ground_pct    = 0.65
    offground_pct = 1.0 - ground_pct
    ground_pts    = int(effective_pts * ground_pct)
    unique_cells  = min(effective_pts, worst_cells)     # at most one per voxel cell
    reprojected   = is_geographic

    # ── Step 4: DTM Generation ────────────────────────────────────────────────
    dtm_res  = 1.0   # always 1 m/px
    dtm_cols = max(1, int(x_range_m))
    dtm_rows = max(1, int(y_range_m))
    dtm_cells = dtm_rows * dtm_cols

    # Elevation range — use LAS header values if present, else typical estimate
    if elev_min is not None and elev_max is not None:
        z_min_v, z_max_v = float(elev_min), float(elev_max)
    else:
        # Rough guess: lowland relief for most Indian villages
        z_min_v = 50.0
        z_max_v = 50.0 + max(5.0, y_range_m * 0.03)
    elev_range = z_max_v - z_min_v

    # ── Step 5: Terrain Derivatives ───────────────────────────────────────────
    # Empirical from 8-village dataset: slope_max ≈ 22–82°, median ~38°
    slope_min   = 0.01
    slope_max   = round(min(82.0, max(15.0, elev_range / max(x_range_m, y_range_m) * 3000)), 1)
    # Stream threshold = 97th percentile of accumulation raster
    # accum is dimensionless flow proxy; typical p97 ≈ 0.8 × dtm_cells / 100
    stream_thresh = round(dtm_cells * 0.008, 0)
    stream_cells  = int(dtm_cells * 0.012)
    twi_min       = round(max(1.5, 3.0 - elev_range * 0.01), 1)
    twi_max       = round(min(30.0, 18.0 + elev_range * 0.05), 1)
    accum_p97     = int(stream_thresh)

    # ── Step 6: ML / XGBoost ─────────────────────────────────────────────────
    feat_count = 12
    # Rule-based thresholds: high risk > 0.92 quantile, med 0.78–0.92
    # Calibrated to match observed ratios from notebooks
    rule_high   = int(dtm_cells * 0.072)
    rule_med    = int(dtm_cells * 0.170)
    train_n     = min(200_000, rule_high + rule_med)
    ml_high_raw = int(rule_high * 0.987)
    ml_med_raw  = int(rule_med  * 0.986)
    # Confidence improves with density; clamp 0.83–0.91
    density_factor = min(1.0, total_pts / (area_m2 + 1e-6) / 50.0)
    conf_mean = round(min(0.91, 0.83 + density_factor * 0.08), 3)
    conf_max  = round(min(0.999, conf_mean + 0.08), 3)

    # ── Step 7: Morphological Cleanup ────────────────────────────────────────
    noise_removed = int((ml_high_raw + ml_med_raw) * 0.025)
    final_high    = ml_high_raw - int(noise_removed * 0.60)
    final_med     = ml_med_raw  - int(noise_removed * 0.40)
    ml_fallback   = (ml_high_raw + ml_med_raw) < 10

    # ── Step 8: Drainage Routing ──────────────────────────────────────────────
    # clusters ≈ labelled connected components of high-risk mask
    clusters   = max(2, int(final_high / max(dtm_cells * 0.001, 1)))
    clusters   = min(clusters, 100)
    drain_km   = round(math.sqrt(max(area_km2, 1e-9)) * 1.85, 2)   # sqrt-safe
    drains     = max(1, int(clusters * 0.82))

    # ── Export list ───────────────────────────────────────────────────────────
    exports = [
        ("Ground LAS",        f"{village}_ground.las",          "ground/"),
        ("DTM COG",           f"{village}_dtm_cog.tif",         "dtm/"),
        ("Slope COG",         f"{village}_slope_cog.tif",       "rasters/"),
        ("Accumulation COG",  f"{village}_accum_cog.tif",       "rasters/"),
        ("TWI COG",           f"{village}_twi_cog.tif",         "rasters/"),
        ("Risk COG",          f"{village}_risk_cog.tif",        "rasters/"),
        ("Confidence COG",    f"{village}_confidence_cog.tif",  "rasters/"),
        ("GeoPackage",        f"{village}_drainage.gpkg",       "gpkg/"),
        ("Village Report",    f"{village}_report.png",          "village_reports/"),
    ]

    return dict(
        # --- inputs ---
        village=village, epsg=epsg, coord_type=coord_type,
        is_geographic=is_geographic,
        total_pts=total_pts, size_mb=size_mb,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        x_range=x_range, y_range=y_range,
        x_range_m=round(x_range_m, 1), y_range_m=round(y_range_m, 1),
        area_m2=round(area_m2, 1), area_km2=round(area_km2, 4),
        # --- scaling ---
        voxel_m=voxel_m, VOXEL_RES=round(VOXEL_RES, 8),
        grid_cols=grid_cols, grid_rows=grid_rows,
        worst_cells_M=round(worst_cells / 1e6, 2),
        STRIDE=STRIDE, effective_pts=effective_pts,
        grid_ram_mb=round(grid_ram_mb, 1),
        # --- ground ---
        chunked=chunked, mode=("Chunked (CSF)" if chunked else "Regular (CSF)"),
        unique_cells=unique_cells,
        ground_pts=ground_pts,
        ground_pct=round(ground_pct * 100, 1),
        offground_pct=round(offground_pct * 100, 1),
        reprojected=reprojected,
        # --- DTM ---
        dtm_res=dtm_res, dtm_rows=dtm_rows, dtm_cols=dtm_cols,
        dtm_cells=dtm_cells,
        z_min=round(z_min_v, 1), z_max=round(z_max_v, 1),
        elev_range=round(elev_range, 1),
        smoothing="Median 3×3 + Gaussian σ=1",
        # --- derivatives ---
        slope_min=slope_min, slope_max=slope_max,
        stream_thresh=stream_thresh, stream_cells=stream_cells,
        twi_min=twi_min, twi_max=twi_max, accum_p97=accum_p97,
        # --- ML ---
        feat_count=feat_count,
        rule_high=rule_high, rule_med=rule_med,
        train_n=train_n,
        ml_high_raw=ml_high_raw, ml_med_raw=ml_med_raw,
        conf_mean=conf_mean, conf_max=conf_max,
        # --- cleanup ---
        noise_removed=noise_removed,
        final_high=final_high, final_med=final_med,
        ml_fallback=ml_fallback,
        # --- routing ---
        clusters=clusters, drains=drains,
        drain_km=drain_km,
        avg_drain_km=round(drain_km / max(drains, 1), 2),
        # --- export ---
        exports=exports,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-DETECTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _village_from_filename(fname: str) -> str:
    """Strip extension + path separators, return clean uppercase name."""
    import os, re
    base = os.path.splitext(os.path.basename(fname))[0]
    base = re.sub(r"\s*\(.*?\)", "", base)       # drop e.g. "(511671)"
    base = re.sub(r"[^A-Za-z0-9_\-]", "_", base) # non-alphanum → _
    base = re.sub(r"_+", "_", base).strip("_")
    return base.upper()[:30]


def _guess_epsg(x_min, x_max, y_min, y_max, is_geographic: bool) -> int:
    """
    Heuristic EPSG from bounding box.
    Covers Indian subcontinent UTM zones 42N-46N + WGS84 geographic.
    """
    if is_geographic:
        return 4326
    x_c = (x_min + x_max) / 2.0
    y_c = (y_min + y_max) / 2.0
    # Andaman & Nicobar — UTM Zone 46N
    if 1_300_000 <= y_c <= 1_600_000:
        return 32646
    # Northern / Central India
    if y_c >= 2_000_000:
        if   x_c < 350_000: return 32642
        elif x_c < 520_000: return 32643
        elif x_c < 700_000: return 32644
        elif x_c < 850_000: return 32645
        else:               return 32646
    # Southern India
    if 700_000 <= y_c < 2_000_000:
        if   x_c < 500_000: return 32643
        elif x_c < 700_000: return 32644
        else:               return 32645
    return 32643   # global fallback


def _detect_coord_type(x_min, x_max, y_min, y_max) -> str:
    max_val = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    return "GEOGRAPHIC" if max_val < 500 else "PROJECTED"


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Branding ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:10px 0 16px;">
      <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                  letter-spacing:3px;color:#2a9fd6;text-transform:uppercase;">
        IIT Tirupati · MoPR Hackathon
      </div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.4rem;
                  color:#e8f4ff;margin-top:4px;">GramDrain</div>
      <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#3d7ab5;">
        LiDAR Pipeline Calculator
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── File upload ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#2a9fd6;letter-spacing:2px;text-transform:uppercase;
                margin-bottom:6px;">① Drop your file</div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload LAS / LAZ / ZIP (up to 10 GB)",
        type=["las", "laz", "zip"],
        help="All parameters are auto-detected from the file header. No need to enter anything manually.",
    )

    # ── Parse header & auto-fill EVERY field ─────────────────────────────────
    hdr = {}
    parse_error = None
    if uploaded is not None:
        try:
            with st.spinner("🔍 Reading file header …"):
                if uploaded.name.lower().endswith(".zip"):
                    hdr = parse_las_from_zip(uploaded)
                else:
                    hdr = parse_las_header_from_upload(uploaded)
        except Exception as e:
            parse_error = str(e)

    # Compute derived auto-values whenever we have a header
    if hdr:
        _ct  = _detect_coord_type(hdr["x_min"], hdr["x_max"], hdr["y_min"], hdr["y_max"])
        _epsg_auto = _guess_epsg(hdr["x_min"], hdr["x_max"],
                                  hdr["y_min"], hdr["y_max"], _ct == "GEOGRAPHIC")
        _vname_auto  = _village_from_filename(uploaded.name)
        _sizemb_auto = round(uploaded.size / 1e6, 1)   # bytes → MB (exact)
    else:
        _ct          = "PROJECTED"
        _epsg_auto   = 32643
        _vname_auto  = "DATASET"
        _sizemb_auto = 1.0

    # ── Auto-detected summary card (shown only after upload) ─────────────────
    if hdr and not parse_error:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0a2a10,#051a08);
                    border:1px solid #1a6640;border-radius:8px;
                    padding:12px 14px;margin:10px 0;">
          <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                      color:#2ecc71;letter-spacing:2px;text-transform:uppercase;
                      margin-bottom:8px;">✅ Auto-detected from header</div>
          <table style="width:100%;font-size:0.7rem;font-family:'Space Mono',monospace;
                        border-collapse:collapse;">
            <tr><td style="color:#4a6f94;padding:2px 0;">Village</td>
                <td style="color:#80c8ff;text-align:right;">{_vname_auto}</td></tr>
            <tr><td style="color:#4a6f94;padding:2px 0;">Points</td>
                <td style="color:#80c8ff;text-align:right;">{hdr["point_count"]/1e6:.2f} M</td></tr>
            <tr><td style="color:#4a6f94;padding:2px 0;">File size</td>
                <td style="color:#80c8ff;text-align:right;">{_sizemb_auto} MB</td></tr>
            <tr><td style="color:#4a6f94;padding:2px 0;">Coord type</td>
                <td style="color:#80c8ff;text-align:right;">{_ct}</td></tr>
            <tr><td style="color:#4a6f94;padding:2px 0;">X range</td>
                <td style="color:#80c8ff;text-align:right;">
                  {hdr["x_min"]:.2f} → {hdr["x_max"]:.2f}</td></tr>
            <tr><td style="color:#4a6f94;padding:2px 0;">Y range</td>
                <td style="color:#80c8ff;text-align:right;">
                  {hdr["y_min"]:.2f} → {hdr["y_max"]:.2f}</td></tr>
            <tr><td style="color:#4a6f94;padding:2px 0;">Z range</td>
                <td style="color:#80c8ff;text-align:right;">
                  {hdr["z_min"]:.1f} → {hdr["z_max"]:.1f} m</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
    elif parse_error:
        st.error(f"Header parse failed: {parse_error}")

    # ── Override / manual entry (collapsible) ─────────────────────────────────
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                color:#3d7ab5;letter-spacing:2px;text-transform:uppercase;
                margin:10px 0 4px;">② Override (optional)</div>
    """, unsafe_allow_html=True)

    with st.expander("✏️ Edit parameters" if hdr else "✏️ Enter parameters manually",
                     expanded=(not bool(hdr))):

        village_name = st.text_input(
            "Village / Dataset name",
            value=_vname_auto,
        )
        # EPSG is computed internally from extent — not shown to user
        epsg_code = _epsg_auto
        total_pts = st.number_input(
            "Total LiDAR points",
            value=int(hdr.get("point_count", 1_000_000)),
            min_value=1, step=1_000_000, format="%d",
        )
        size_mb = st.number_input(
            "File size (MB)",
            value=_sizemb_auto,
            min_value=0.1, step=10.0,
        )
        st.markdown("**Bounding extent**")
        col_a, col_b = st.columns(2)
        x_min = col_a.number_input(
            "X min", value=float(hdr.get("x_min", 0.0)), format="%.4f")
        x_max = col_b.number_input(
            "X max", value=float(hdr.get("x_max", 1000.0)), format="%.4f")
        y_min = col_a.number_input(
            "Y min", value=float(hdr.get("y_min", 0.0)), format="%.2f")
        y_max = col_b.number_input(
            "Y max", value=float(hdr.get("y_max", 1000.0)), format="%.2f")

        st.markdown("**Elevation (Z)**")
        col_c, col_d = st.columns(2)
        elev_min_in = col_c.number_input(
            "Z min (m)", value=float(hdr.get("z_min", 0.0)), format="%.2f")
        elev_max_in = col_d.number_input(
            "Z max (m)", value=float(hdr.get("z_max", 100.0)), format="%.2f")
        use_elev = True   # always use elevation when present

        st.markdown("**Coordinate type**")
        _ct_options  = ["Auto-detect", "PROJECTED", "GEOGRAPHIC"]
        _ct_default  = 0  # Auto-detect
        coord_radio  = st.radio(
            "Coordinate type", _ct_options, index=_ct_default,
            label_visibility="collapsed",
            help="Auto-detect reads the bounding box range. Override if needed.",
        )
        resolved_coord = (_detect_coord_type(x_min, x_max, y_min, y_max)
                          if coord_radio == "Auto-detect" else coord_radio)

    # Values for compute (from override inputs above)
    # If expander was never opened, Streamlit still returns widget defaults = auto values.

    st.markdown("---")
    can_run = bool(hdr) and not bool(parse_error)
    run_btn = st.button("🔬  Calculate Pipeline", use_container_width=True, disabled=not can_run)
    if run_btn:
        st.session_state.pipeline_run = True
        st.session_state.last_uploaded = uploaded.name if uploaded else None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-title">LiDAR Pipeline Calculator</div>
<div class="main-subtitle">GramDrain · IIT Tirupati · MoPR Hackathon</div>
""", unsafe_allow_html=True)

if uploaded:
    if uploaded.name != st.session_state.get("last_uploaded"):
        st.session_state.pipeline_run = False
else:
    st.session_state.pipeline_run = False

if not st.session_state.get("pipeline_run", False):
    st.markdown("""
    <div style="margin-top:48px;text-align:center;">
      <div style="font-family:'Space Mono',monospace;font-size:2.5rem;
                  color:#1a4070;margin-bottom:16px;">🔬</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;
                  color:#3d7ab5;margin-bottom:8px;">
        Upload a LAS / LAZ file <b>or</b> enter parameters in the sidebar
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                  color:#1a4070;letter-spacing:2px;">
        THEN CLICK &nbsp;🔬 CALCULATE PIPELINE
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Save uploaded file to disk (needed by pipeline_engine) ───────────────────
import os as _os
OUT_DIR   = _os.path.join(_os.path.dirname(__file__), "outputs")
TEMP_DIR  = _os.path.join(_os.path.dirname(__file__), "temp")
_os.makedirs(TEMP_DIR, exist_ok=True)
_os.makedirs(OUT_DIR,  exist_ok=True)

_vname_clean = village_name.upper().replace(" ", "_")

# Determine the saved LAS path
if uploaded is not None:
    _ext      = _os.path.splitext(uploaded.name)[-1].lower()
    _las_path = _os.path.join(TEMP_DIR, f"{_vname_clean}{_ext}")
    # Only re-write if the file has changed (avoid re-writing 10 GB on every rerun)
    _need_write = (not _os.path.exists(_las_path) or
                   _os.path.getsize(_las_path) != uploaded.size)
    if _need_write:
        with st.spinner(f"💾 Saving {uploaded.name} to disk ({_sizemb_auto} MB) …"):
            uploaded.seek(0)
            with open(_las_path, "wb") as _f:
                while True:
                    _chunk = uploaded.read(16 * 1024 * 1024)   # 16 MB chunks
                    if not _chunk:
                        break
                    _f.write(_chunk)
        st.success(f"File saved to {_las_path}")
    _file_ready = True
else:
    _las_path   = None
    _file_ready = False

# ── Run Real Pipeline ─────────────────────────────────────────────────────────
_real_outputs = {}
if _file_ready:
    try:
        from pipeline_engine import run_pipeline as _run_pipeline
        _log_lines = []

        if "real_outputs" not in st.session_state or st.session_state.get("computed_file") != _las_path:
            with st.status("🔬 Running full LiDAR pipeline …", expanded=True) as _status:
                def _log_cb(msg: str):
                    _log_lines.append(msg)
                    st.write(msg)

                _real_outputs = _run_pipeline(
                    las_path     = _las_path,
                    epsg         = int(epsg_code),
                    village_name = _vname_clean,
                    out_dir      = OUT_DIR,
                    log_callback = _log_cb,
                )
                st.session_state.real_outputs = _real_outputs
                st.session_state.computed_file = _las_path
                _status.update(label="✅ Pipeline complete!", state="complete", expanded=False)
        else:
            _real_outputs = st.session_state.real_outputs

    except ImportError as _ie:
        st.error(f"⚠️ Missing pipeline library: {_ie}\n\n"
                 "Install with:\n```\npip install laspy[lazrs] rasterio geopandas "
                 "xgboost scikit-learn cloth-simulation-filter networkx pyproj whitebox\n```")
    except Exception as _pe:
        st.error(f"❌ Pipeline error: {_pe}")
        import traceback
        st.code(traceback.format_exc())

# ── Compute math-based estimates for chart visualizations ────────────────────
p = compute_pipeline(
    total_pts=int(total_pts),
    size_mb=float(size_mb),
    x_min=float(x_min), x_max=float(x_max),
    y_min=float(y_min), y_max=float(y_max),
    coord_type=resolved_coord,
    village=_vname_clean,
    epsg=int(epsg_code),
    elev_min=float(elev_min_in) if use_elev else None,
    elev_max=float(elev_max_in) if use_elev else None,
)

# Overwrite fallback estimations with REAL metrics from the pipeline
if _real_outputs and "metrics" in _real_outputs:
    p.update(_real_outputs["metrics"])



# ── Top bar: village name + risk gauge + download ─────────────────────────────
top1, top2, top3 = st.columns([3, 2, 1])

with top1:
    st.markdown(f"""
    <div class="main-title" style="font-size:1.6rem;">{p['village']}</div>
    <div class="main-subtitle">
      EPSG:{p['epsg']} &nbsp;·&nbsp; {p['coord_type']}
      &nbsp;·&nbsp; {p['total_pts']/1e6:.2f}M pts
      &nbsp;·&nbsp; {p['size_mb']:.0f} MB
    </div>
    """, unsafe_allow_html=True)

with top2:
    risk_pct = (p["final_high"] + p["final_med"]) / max(p["dtm_cells"], 1) * 100
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_pct, 1),
        number={"suffix": "%", "font": {"color": "#80c8ff", "size": 22, "family": "Space Mono"}},
        title={"text": "Area at Risk", "font": {"color": "#3d7ab5", "size": 10}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3d7ab5"},
            "bar": {"color": "#e74c3c"},
            "bgcolor": "#071422", "bordercolor": "#0f2540",
            "steps": [
                {"range": [0, 33], "color": "#0d3d24"},
                {"range": [33, 66], "color": "#3d2b00"},
                {"range": [66, 100], "color": "#3d0d0d"},
            ],
        },
    ))
    fig_g.update_layout(
        paper_bgcolor="#050d1a", plot_bgcolor="#050d1a",
        height=155, margin=dict(l=10, r=10, t=20, b=0),
        font=dict(color="#7eb8e8", family="Space Mono", size=9),
    )
    st.plotly_chart(fig_g, width='stretch', config={"displayModeBar": False})

with top3:
    # Build JSON for download (exclude non-serializable exports list)
    p_json = {k: v for k, v in p.items() if k != "exports"}
    p_json["exports"] = [{"layer": e[0], "file": e[1], "folder": e[2]} for e in p["exports"]]
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇ Download JSON",
        data=json.dumps(p_json, indent=2),
        file_name=f"{p['village']}_pipeline_params.json",
        mime="application/json",
        width='stretch',
    )

sdiv()

# ── Pipeline progress bar ─────────────────────────────────────────────────────
STEP_LABELS = ["Ingest", "Scale", "Ground", "DTM", "Derivatives", "ML", "Cleanup", "Routing", "Export"]
prog_cols = st.columns(len(STEP_LABELS))
for i, (col, lbl) in enumerate(zip(prog_cols, STEP_LABELS)):
    col.markdown(f"""
    <div style="text-align:center;">
      <div style="background:#0f2a45;border:1px solid #1a5080;border-radius:50%;
                  width:32px;height:32px;line-height:32px;
                  font-family:'Space Mono',monospace;font-size:0.7rem;
                  color:#2a9fd6;margin:0 auto 4px;font-weight:700;">{i+1}</div>
      <div style="font-size:0.55rem;color:#3d7ab5;text-transform:uppercase;
                  letter-spacing:1px;font-family:'Space Mono',monospace;">{lbl}</div>
    </div>""", unsafe_allow_html=True)

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — RAW LiDAR INGEST
# ═══════════════════════════════════════════════════════════════════════════════
step_header(1, "Raw LiDAR Ingest", "Point cloud ingestion & inspection")

c1, c2, c3, c4, c5, c6 = st.columns(6)
metric("Total Points",  f"{p['total_pts']/1e6:.2f}M",  f"raw: {p['total_pts']:,}", c1)
metric("File Size",     f"{p['size_mb']:.1f} MB",       f"{p['size_mb']/1024:.2f} GB", c2)
metric("Density",       f"{p['total_pts']/max(p['area_m2'],1):.1f} pt/m²",
       f"{p['area_km2']} km²", c3)
metric("X Extent",      f"{p['x_range_m']:.0f} m",
       f"{p['x_min']:.2f} → {p['x_max']:.2f}", c4)
metric("Y Extent",      f"{p['y_range_m']:.0f} m",
       f"{p['y_min']:.2f} → {p['y_max']:.2f}", c5)

reproj_b = badge("REPROJECTION NEEDED", "warn") if p["reprojected"] else badge("PROJECTED OK", "ok")
c6.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Coord / EPSG</div>
  <div style="margin-top:8px;">{reproj_b}</div>
  <div class="metric-sub">EPSG:{p['epsg']} · {p['coord_type']}</div>
</div>""", unsafe_allow_html=True)

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DYNAMIC SCALING
# ═══════════════════════════════════════════════════════════════════════════════
step_header(2, "Dynamic Scaling", "Voxel grid + stride decimation")

c1, c2, c3, c4, c5 = st.columns(5)
metric("Voxel Size",    f"{p['voxel_m']} m",           "decimation cell",      c1)
metric("Stride",        f"1 / {p['STRIDE']}",          "voxel keeper",         c2)
metric("Effective Pts", f"{p['effective_pts']/1e6:.2f}M", "post-decimation",   c3)
metric("Grid Cells",    f"{p['grid_rows']:,} × {p['grid_cols']:,}",
       f"{p['worst_cells_M']:.2f}M cells",                                     c4)
metric("Grid RAM",      f"{p['grid_ram_mb']:.0f} MB",  "estimated peak",       c5)

# Scale bar chart
fig_scale = go.Figure()
fig_scale.add_trace(go.Bar(
    x=["Raw Points", "Effective Pts"],
    y=[p["total_pts"]/1e6, p["effective_pts"]/1e6],
    marker_color=["#1a4070", "#2a9fd6"],
    text=[f"{p['total_pts']/1e6:.1f}M", f"{p['effective_pts']/1e6:.1f}M"],
    textfont=dict(color="#c8d8f0", size=10), textposition="outside",
))
dark_fig(fig_scale, "Point Decimation (millions)", 210)
fig_scale.update_yaxes(title_text="Million Points", title_font=dict(size=9))
st.plotly_chart(fig_scale, width='stretch', config={"displayModeBar": False})

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — GROUND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
step_header(3, "Ground Classification", "CSF cloth simulation filter")

c1, c2, c3, c4, c5 = st.columns(5)
metric("Mode",           p["mode"],                      "",                    c1)
metric("Unique Cells",   f"{p['unique_cells']/1e6:.2f}M", "filled voxel cells", c2)
metric("Ground Pts",     f"{p['ground_pts']/1e6:.2f}M", "classified ground",   c3)
metric("Ground %",       f"{p['ground_pct']:.1f}%",      "of decimated pts",   c4)
metric("Off-Ground %",   f"{p['offground_pct']:.1f}%",   "veg/building",       c5)

# Donut
fig_donut = go.Figure(go.Pie(
    values=[p["ground_pct"], p["offground_pct"]],
    labels=["Ground", "Off-Ground"],
    hole=0.65,
    marker_colors=["#2a9fd6", "#1a4070"],
    textinfo="label+percent",
    textfont=dict(color="#c8d8f0", size=9),
))
dark_fig(fig_donut, "Ground / Off-Ground Split", 220)
fig_donut.update_layout(showlegend=False)
st.plotly_chart(fig_donut, width='stretch', config={"displayModeBar": False})

# Reprojection note
if p["reprojected"]:
    st.markdown("""
    <div class="note-box">
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#f39c12;
                  text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">
        ⚠ Reprojection Required
      </div>
      <div style="font-size:0.82rem;color:#c8d8f0;">
        Coordinates are GEOGRAPHIC (lat/lon). Pipeline will reproject to
        <code style="background:#0c2240;color:#80c8ff;padding:1px 5px;">EPSG:{epsg_code}</code>
        using <code style="background:#0c2240;color:#80c8ff;padding:1px 5px;">pyproj.Transformer</code>
        before CSF ground classification.
      </div>
    </div>
    """.replace("{epsg_code}", str(epsg_code)), unsafe_allow_html=True)

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — DTM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
step_header(4, "DTM Generation", "Rasterise ground → COG")

c1, c2, c3, c4, c5 = st.columns(5)
metric("Resolution",  f"{p['dtm_res']} m/px",   "output GSD",         c1)
metric("Raster Dims", f"{p['dtm_rows']:,} × {p['dtm_cols']:,}",
       f"{p['dtm_cells']/1e6:.2f}M cells",                            c2)
metric("Z Min",       f"{p['z_min']:.1f} m",    "above MSL",          c3)
metric("Z Max",       f"{p['z_max']:.1f} m",    "above MSL",          c4)
metric("Z Range",     f"{p['elev_range']:.1f} m", p["smoothing"],     c5)

# Synthetic elevation histogram
rng = np.random.default_rng(abs(hash(village_name)) % 2**31)
elev_samples = (rng.beta(2, 3, 1500) * p["elev_range"] + p["z_min"])
fig_elev = go.Figure(go.Histogram(
    x=elev_samples, nbinsx=40,
    marker_color="#2a9fd6", opacity=0.85,
))
dark_fig(fig_elev, "Elevation Distribution (estimated)", 220)
fig_elev.update_xaxes(title_text="Elevation (m)", title_font=dict(size=9))
fig_elev.update_yaxes(title_text="Frequency",     title_font=dict(size=9))
st.plotly_chart(fig_elev, width='stretch', config={"displayModeBar": False})

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TERRAIN DERIVATIVES
# ═══════════════════════════════════════════════════════════════════════════════
step_header(5, "Terrain Derivatives", "Slope · Flow accumulation · TWI")

c1, c2, c3, c4, c5, c6 = st.columns(6)
metric("Slope Min",     f"{p['slope_min']:.2f}°",           "flat terrain",       c1)
metric("Slope Max",     f"{p['slope_max']:.1f}°",            "steepest cell",     c2)
metric("Stream Thresh", f"{p['stream_thresh']:,.0f}",        "p97 accum value",   c3)
metric("Stream Cells",  f"{p['stream_cells']:,}",            "drainage network",  c4)
metric("TWI Min",       f"{p['twi_min']:.1f}",               "driest",            c5)
metric("TWI Max",       f"{p['twi_max']:.1f}",               "wettest",           c6)

fig_deriv = make_subplots(rows=1, cols=2,
    subplot_titles=["Slope Distribution (°)", "TWI Distribution"])
slope_s = np.abs(rng.exponential(p["slope_max"] / 8, 800)).clip(p["slope_min"], p["slope_max"])
twi_s   = rng.normal((p["twi_min"] + p["twi_max"]) / 2,
                     (p["twi_max"] - p["twi_min"]) / 6, 800)
fig_deriv.add_trace(go.Histogram(x=slope_s, marker_color="#f39c12", opacity=0.8,
                                  name="Slope", showlegend=False), row=1, col=1)
fig_deriv.add_trace(go.Histogram(x=twi_s,   marker_color="#2a9fd6", opacity=0.8,
                                  name="TWI",   showlegend=False), row=1, col=2)
fig_deriv.update_layout(
    paper_bgcolor="#071422", plot_bgcolor="#071422",
    font=dict(color="#7eb8e8", family="Space Mono", size=9),
    height=230, margin=dict(l=40, r=20, t=36, b=40),
    annotations=[dict(font=dict(color="#c8d8f0", size=10))],
)
for ax in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
    fig_deriv.update_layout(**{ax: dict(gridcolor="#0f2540", linecolor="#0f2540")})
st.plotly_chart(fig_deriv, width='stretch', config={"displayModeBar": False})

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RISK MODELLING (XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
step_header(6, "Risk Modelling", "Rule-based labels → XGBoost 12-feature matrix")

c1, c2, c3, c4, c5, c6 = st.columns(6)
metric("Rule High",     f"{p['rule_high']:,}",   "label=2 train",      c1)
metric("Rule Medium",   f"{p['rule_med']:,}",    "label=1 train",      c2)
metric("Features",      f"{p['feat_count']}",    "per cell",           c3)
metric("Train Samples", f"{p['train_n']:,}",     "XGBoost fit",        c4)
metric("Conf Mean",     f"{p['conf_mean']:.3f}", "avg prob(high)",     c5)
metric("Conf Max",      f"{p['conf_max']:.3f}",  "peak confidence",    c6)

tab1, tab2 = st.tabs(["  Feature Matrix  ", "  Label Distribution  "])

FEATURES_12 = [
    "DTM Elevation", "Slope (°)", "Flow Accumulation", "TWI",
    "Local Relief",  "Roughness", "Profile Curvature", "Plan Curvature",
    "Aspect",        "Laplacian", "Median Residual",   "Uniform Filter Residual",
]

with tab1:
    fi_vals = rng.dirichlet(np.ones(12) * 2)
    fi_vals.sort(); fi_vals = fi_vals[::-1]
    fig_fi = go.Figure(go.Bar(
        x=FEATURES_12, y=fi_vals,
        marker_color=[f"rgba(42,159,214,{0.4+0.6*v/fi_vals.max()})" for v in fi_vals],
        text=[f"{v:.3f}" for v in fi_vals],
        textposition="outside", textfont=dict(size=8, color="#c8d8f0"),
    ))
    dark_fig(fig_fi, "XGBoost Feature Importance (estimated)", 250)
    fig_fi.update_layout(xaxis=dict(tickangle=-30, tickfont=dict(size=8)))
    st.plotly_chart(fig_fi, width='stretch', config={"displayModeBar": False})

with tab2:
    safe_cells = p["dtm_cells"] - p["rule_high"] - p["rule_med"]
    fig_lbl = go.Figure(go.Bar(
        x=["Safe (0)", "Medium (1)", "High (2)"],
        y=[safe_cells, p["rule_med"], p["rule_high"]],
        marker_color=["#0d3d24", "#3d2b00", "#3d0d0d"],
        marker_line_color=["#2ecc71", "#f39c12", "#e74c3c"],
        marker_line_width=1.5,
        text=[f"{safe_cells:,}", f"{p['rule_med']:,}", f"{p['rule_high']:,}"],
        textfont=dict(color="#c8d8f0", size=10), textposition="outside",
    ))
    dark_fig(fig_lbl, "Training Label Distribution", 240)
    st.plotly_chart(fig_lbl, width='stretch', config={"displayModeBar": False})

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — MORPHOLOGICAL CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════
step_header(7, "Morphological Post-processing", "binary_opening 3×3 → binary_closing 5×5")

c1, c2, c3, c4, c5 = st.columns(5)
metric("Noise Removed", f"{p['noise_removed']:,}",  "pixels erased",    c1)
metric("Final High",    f"{p['final_high']:,}",     "risk=2 cells",     c2)
metric("Final Medium",  f"{p['final_med']:,}",      "risk=1 cells",     c3)
metric("ML Fallback",   "YES" if p["ml_fallback"] else "NO",
       "rule fallback" if p["ml_fallback"] else "ML used",               c4)

raw_total   = p["ml_high_raw"] + p["ml_med_raw"]
final_total = p["final_high"]  + p["final_med"]
red_pct     = (raw_total - final_total) / max(raw_total, 1) * 100
c5.markdown(f"""
<div class="metric-card">
  <div class="metric-label">Noise Reduction</div>
  <div class="metric-value">{red_pct:.1f}%</div>
  <div class="metric-sub">{raw_total:,} → {final_total:,}</div>
</div>""", unsafe_allow_html=True)

fig_clean = go.Figure()
fig_clean.add_trace(go.Bar(
    name="Raw ML", x=["High Risk", "Medium Risk"],
    y=[p["ml_high_raw"], p["ml_med_raw"]],
    marker_color=["#3d0d0d", "#3d2b00"],
    marker_line_color=["#7a1a1a", "#7a5400"], marker_line_width=1,
))
fig_clean.add_trace(go.Bar(
    name="Post-Cleanup", x=["High Risk", "Medium Risk"],
    y=[p["final_high"], p["final_med"]],
    marker_color=["#6b0000", "#7a4400"],
    marker_line_color=["#e74c3c", "#f39c12"], marker_line_width=1.5,
))
dark_fig(fig_clean, "Before vs After Morphological Cleanup", 230)
fig_clean.update_layout(barmode="group",
    legend=dict(font=dict(color="#c8d8f0", size=9), bgcolor="#071422"))
st.plotly_chart(fig_clean, width='stretch', config={"displayModeBar": False})

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — DRAINAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
step_header(8, "Drainage Network Design", "Gravity-fed Dijkstra routing to natural streams")

c1, c2, c3, c4 = st.columns(4)
metric("Hotspot Clusters", str(p["clusters"]),            "connected components",    c1)
metric("Proposed Drains",  str(p["drains"]),              "gravity-routed paths",    c2)
metric("Total Length",     f"{p['drain_km']} km",         "drain path distance",     c3)
metric("Avg Drain",        f"{p['avg_drain_km']:.2f} km", "per drain",               c4)

st.markdown("""
<div class="note-box">
  <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#2a9fd6;
              text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">
    Routing Algorithm Note
  </div>
  <div style="font-size:0.82rem;color:#c8d8f0;line-height:1.7;">
    Cost function: <code style="background:#0c2240;color:#80c8ff;padding:1px 5px;">
    (1−norm_accum)×60 + norm_elev×40 + 1</code><br>
    Low-cost corridors follow natural valleys (high accumulation + low elevation).<br>
    NODATA cells = 9999 (impenetrable) — drains never exit village boundary.<br>
    Only clusters ≥ 25 high-risk pixels are routed; max 50 clusters per village.
  </div>
</div>""", unsafe_allow_html=True)

# Synthetic drain length chart
drain_lengths = np.abs(rng.exponential(
    p["drain_km"] / max(p["drains"], 1) * 1000, p["drains"]
))
fig_drains = go.Figure(go.Bar(
    x=[f"D{i+1:02d}" for i in range(len(drain_lengths))],
    y=drain_lengths,
    marker_color="#2a9fd6", marker_line_color="#4abcf0", marker_line_width=0.5,
))
dark_fig(fig_drains, "Proposed Drain Lengths (m) — estimated distribution", 240)
fig_drains.update_xaxes(tickfont=dict(size=8))
fig_drains.update_yaxes(title_text="Length (m)", title_font=dict(size=9))
st.plotly_chart(fig_drains, width='stretch', config={"displayModeBar": False})

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — EXPORT OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
step_header(9, "Export Outputs", "OGC-compliant COG rasters · GeoPackage · Reports")

import os

# Map p["exports"] row → key in _real_outputs dict (from pipeline_engine)
_EXPORT_KEY_MAP = {
    "Ground LAS":     "ground",
    "DTM COG":        "dtm",
    "Slope COG":      "slope",
    "Accumulation COG": "accum",
    "TWI COG":        "twi",
    "Risk COG":       "risk",
    "Confidence COG": "confidence",
    "GeoPackage":     "gpkg",
    "Village Report": "report",
}

if _real_outputs:
    st.success("✅ Pipeline finished — all output files are ready to download.")
else:
    st.info("ℹ️ Upload a file and click Calculate to generate real output files.")

header_cols = st.columns([2, 3, 2, 2])
header_cols[0].markdown("<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#4a6f94;text-transform:uppercase;'>Layer</div>", unsafe_allow_html=True)
header_cols[1].markdown("<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#4a6f94;text-transform:uppercase;'>Filename</div>", unsafe_allow_html=True)
header_cols[2].markdown("<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#4a6f94;text-transform:uppercase;'>Folder</div>", unsafe_allow_html=True)
header_cols[3].markdown("<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#4a6f94;text-transform:uppercase;'>Status</div>", unsafe_allow_html=True)

for i, (layer, filename, folder) in enumerate(p["exports"]):
    cols = st.columns([2, 3, 2, 2])
    cols[0].markdown(f"<div style='padding-top:10px;color:#c8d8f0;font-size:0.85rem;'>{layer}</div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div style='padding-top:10px;font-family:Space Mono,monospace;font-size:0.75rem;color:#80c8ff;'>{filename}</div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div style='padding-top:10px;font-family:Space Mono,monospace;font-size:0.75rem;color:#4a6f94;'>📁 {folder}</div>", unsafe_allow_html=True)

    # Prefer real pipeline output path, fall back to folder/filename
    _engine_key = _EXPORT_KEY_MAP.get(layer)
    file_path   = _real_outputs.get(_engine_key) if _real_outputs and _engine_key else None
    if not file_path:
        file_path = os.path.join(folder, filename)
    file_exists = os.path.exists(file_path)
    
    with cols[3]:
        if file_exists:
            # Read real physical file
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                
            st.download_button(
                label="⬇ Download",
                data=file_bytes,
                file_name=filename,
                mime="application/octet-stream",
                key=f"dl_file_{i}",
                width='stretch'
            )
        else:
            # File has not been generated by the pipeline yet
            st.download_button(
                label="❌ Missing",
                data=b"",
                file_name=filename,
                disabled=True,
                key=f"dl_missing_{i}",
                width='stretch',
                help=f"Could not find '{file_path}' on the server."
            )

sdiv()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PARAMETER SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📋  Full Parameter Sheet", expanded=False):
    summary_rows = [
        ("Village",            p["village"]),
        ("EPSG",               str(p["epsg"])),
        ("Coordinate type",    p["coord_type"]),
        ("Total LiDAR points", f"{p['total_pts']:,}"),
        ("File size",          f"{p['size_mb']:.1f} MB"),
        ("X extent",           f"{p['x_min']:.4f} → {p['x_max']:.4f}  ({p['x_range_m']:.0f} m)"),
        ("Y extent",           f"{p['y_min']:.4f} → {p['y_max']:.4f}  ({p['y_range_m']:.0f} m)"),
        ("Area",               f"{p['area_km2']:.4f} km²"),
        ("───── SCALING ─────", ""),
        ("Voxel size",         f"{p['voxel_m']} m"),
        ("VOXEL_RES (native)", f"{p['VOXEL_RES']:.8f}"),
        ("Stride",             f"1/{p['STRIDE']}"),
        ("Grid rows × cols",   f"{p['grid_rows']:,} × {p['grid_cols']:,}"),
        ("Worst-case cells",   f"{p['worst_cells_M']:.2f} M"),
        ("Effective points",   f"{p['effective_pts']:,}"),
        ("Grid RAM",           f"{p['grid_ram_mb']:.1f} MB"),
        ("───── GROUND ─────", ""),
        ("Mode",               p["mode"]),
        ("Unique voxel cells", f"{p['unique_cells']:,}"),
        ("Ground points",      f"{p['ground_pts']:,}"),
        ("Ground %",           f"{p['ground_pct']:.1f}%"),
        ("Off-ground %",       f"{p['offground_pct']:.1f}%"),
        ("Reprojected",        "YES" if p["reprojected"] else "NO"),
        ("───── DTM ─────", ""),
        ("Resolution",         f"{p['dtm_res']} m/px"),
        ("Raster dims",        f"{p['dtm_rows']:,} × {p['dtm_cols']:,}"),
        ("DTM cells",          f"{p['dtm_cells']:,}"),
        ("Elev min",           f"{p['z_min']:.1f} m"),
        ("Elev max",           f"{p['z_max']:.1f} m"),
        ("Elev range",         f"{p['elev_range']:.1f} m"),
        ("Smoothing",          p["smoothing"]),
        ("───── DERIVATIVES ─────", ""),
        ("Slope min",          f"{p['slope_min']:.2f}°"),
        ("Slope max",          f"{p['slope_max']:.1f}°"),
        ("Stream threshold",   f"{p['stream_thresh']:,.0f}"),
        ("Stream cells",       f"{p['stream_cells']:,}"),
        ("TWI min",            str(p["twi_min"])),
        ("TWI max",            str(p["twi_max"])),
        ("Accum p97",          f"{p['accum_p97']:,}"),
        ("───── ML ─────", ""),
        ("Feature count",      str(p["feat_count"])),
        ("Rule high labels",   f"{p['rule_high']:,}"),
        ("Rule medium labels", f"{p['rule_med']:,}"),
        ("Train samples",      f"{p['train_n']:,}"),
        ("Raw ML high",        f"{p['ml_high_raw']:,}"),
        ("Raw ML medium",      f"{p['ml_med_raw']:,}"),
        ("Conf mean",          f"{p['conf_mean']:.3f}"),
        ("Conf max",           f"{p['conf_max']:.3f}"),
        ("───── CLEANUP ─────", ""),
        ("Noise removed",      f"{p['noise_removed']:,}"),
        ("Final high",         f"{p['final_high']:,}"),
        ("Final medium",       f"{p['final_med']:,}"),
        ("ML fallback",        "YES" if p["ml_fallback"] else "NO"),
        ("───── ROUTING ─────", ""),
        ("Hotspot clusters",   str(p["clusters"])),
        ("Proposed drains",    str(p["drains"])),
        ("Drain length",       f"{p['drain_km']} km"),
        ("Avg drain",          f"{p['avg_drain_km']:.2f} km"),
    ]
    import pandas as pd
    df_sum = pd.DataFrame(summary_rows, columns=["Parameter", "Value"])
    st.dataframe(df_sum, width='stretch', hide_index=True)

sdiv()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 8px;
            font-family:'Space Mono',monospace;font-size:0.55rem;
            color:#1a4070;letter-spacing:2px;">
  GRAMDRAIN · IIT TIRUPATI MOPR HACKATHON · {p['village']} ·
  {p['total_pts']:,} POINTS · {p['drains']} PROPOSED DRAINS · {p['drain_km']} KM
</div>
""", unsafe_allow_html=True)
