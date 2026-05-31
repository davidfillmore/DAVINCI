"""Smoke test: MODIS AOD vs two CAM6 runs (base + new dust).

Reproduces Figure 3.1 from Buchholz et al. — Australian bushfire event,
Dec 21-23 2019. For each day, generates:
  - 3-panel AOD comparison (MODIS / Base / New Dust)
  - 2-panel bias maps (Base-MODIS / NewDust-MODIS)
  - Pixel count map

Usage:
    conda activate davinci
    python analyses/modis-aod/scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from davinci_monet.pairing.grid_binning import (
    bin_swath_to_grid,
    edges_from_centers,
    normalize_grid,
)
from davinci_monet.plots.style import apply_ncar_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODIS_BASE = Path.home() / "Data" / "MODIS"
MODIS_PLATFORMS = [
    ("Terra", "MOD04_L2"),
    ("Aqua", "MYD04_L2"),
]
CAM6_BASE_FILE = (
    Path.home() / "Data" / "CAM6"
    / "FCnudged_f09.mam.BaseMar27.2019_2021.001_AODVIS.nc"
)
CAM6_NEWDUST_FILE = (
    Path.home() / "Data" / "CAM6"
    / "FCnudged_f09.mam.newdustMar282025.2019_2021.001_AODVIS.nc"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODIS_VAR = "AOD_550_Dark_Target_Deep_Blue_Combined"
CAM6_VAR = "AODVIS"

# Days of year and corresponding dates
DAYS = [
    (355, "2019-12-21"),
    (356, "2019-12-22"),
    (357, "2019-12-23"),
]


# ---------------------------------------------------------------------------
# 1. Read MODIS granules for one day
# ---------------------------------------------------------------------------

def read_modis_granules(day_of_year: int) -> OrderedDict:
    """Read MODIS L2 granules for a single day from all platforms (Terra + Aqua)."""
    import monetio.sat._modis_l2_mm as modis_reader

    variable_dict = {
        MODIS_VAR: {"minimum": 0.0, "maximum": 10.0, "scale": 0.001},
    }

    all_granules: OrderedDict = OrderedDict()
    for platform, prefix in MODIS_PLATFORMS:
        modis_dir = MODIS_BASE / platform / "C61" / "2019" / str(day_of_year)
        if not modis_dir.exists():
            print(f"  {platform}: no data at {modis_dir}, skipping")
            continue
        file_pattern = str(modis_dir / f"{prefix}.*.hdf")
        print(f"  Reading {platform} granules from {modis_dir}...")
        t0 = time.time()
        granules = modis_reader.read_mfdataset(file_pattern, variable_dict)
        elapsed = time.time() - t0
        print(f"    {len(granules)} granules in {elapsed:.1f}s")
        # Prefix keys with platform to avoid collisions between Terra/Aqua
        for key, val in granules.items():
            all_granules[f"{platform}_{key}"] = val

    print(f"  Total: {len(all_granules)} granules (Terra + Aqua)")
    return all_granules


# ---------------------------------------------------------------------------
# 2. Read CAM6 models
# ---------------------------------------------------------------------------

def read_cam6_models(dates: list[str]) -> tuple[xr.Dataset, xr.Dataset]:
    """Read both CAM6 runs and subset to the target dates."""
    print(f"\nReading CAM6 base: {CAM6_BASE_FILE.name}")
    ds_base = xr.open_dataset(str(CAM6_BASE_FILE))
    print(f"Reading CAM6 new dust: {CAM6_NEWDUST_FILE.name}")
    ds_newdust = xr.open_dataset(str(CAM6_NEWDUST_FILE))

    target_times = [pd.Timestamp(d) for d in dates]
    ds_base = ds_base.sel(time=target_times, method="nearest")
    ds_newdust = ds_newdust.sel(time=target_times, method="nearest")

    print(f"  CAM6 grid: lat={ds_base.sizes['lat']}, lon={ds_base.sizes['lon']}")
    print(f"  Base AODVIS range: [{float(ds_base[CAM6_VAR].min()):.4f}, {float(ds_base[CAM6_VAR].max()):.4f}]")
    print(f"  NewDust AODVIS range: [{float(ds_newdust[CAM6_VAR].min()):.4f}, {float(ds_newdust[CAM6_VAR].max()):.4f}]")

    return ds_base, ds_newdust


# ---------------------------------------------------------------------------
# 3. Grid MODIS onto model grid for one day
# ---------------------------------------------------------------------------

def grid_modis_day(
    granules: OrderedDict,
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
    date_str: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin MODIS granules onto CAM6 grid for a single day.

    Returns (data_grid, count_grid) each shaped (nlon, nlat).
    """
    nlat = len(lat_centers)
    nlon = len(lon_centers)

    lat_edges = edges_from_centers(lat_centers)
    lon_edges = edges_from_centers(lon_centers)

    t0_epoch = pd.Timestamp(date_str).timestamp()
    t1_epoch = (pd.Timestamp(date_str) + pd.Timedelta(days=1)).timestamp()
    time_edges = np.array([t0_epoch, t1_epoch], dtype=np.float64)

    count_grid = np.zeros((1, nlon, nlat), dtype=np.int32)
    data_grid = np.zeros((1, nlon, nlat), dtype=np.float64)

    print(f"  Binning {len(granules)} granules onto {nlon}x{nlat} grid...")
    t0 = time.time()
    n_valid_total = 0

    for i, (key, granule) in enumerate(granules.items()):
        # Keys are "Platform_YYYYjjjHHMM" — strip platform prefix
        datetime_str = key.split("_", 1)[1] if "_" in key else key
        obs_timestamp = pd.to_datetime(datetime_str, format='%Y%j%H%M').timestamp()

        aod = granule[MODIS_VAR].values
        lat = granule["lat"].values
        lon = granule["lon"].values

        aod_flat = aod.flatten().astype(np.float64)
        lat_flat = lat.flatten().astype(np.float64)
        lon_flat = lon.flatten().astype(np.float64)

        fill_mask = (lat_flat < -900) | (lon_flat < -900)
        aod_flat[fill_mask] = np.nan

        # Shift lon from -180..180 to 0..360 to match model grid
        lon_flat = np.where(lon_flat < 0, lon_flat + 360.0, lon_flat)
        # Wrap pixels past last lon edge back into first bin
        lon_flat = np.where(lon_flat >= lon_edges[-1], lon_flat - 360.0, lon_flat)

        n_valid = np.isfinite(aod_flat).sum()
        n_valid_total += n_valid

        n_obs = len(aod_flat)
        time_flat = np.full(n_obs, obs_timestamp, dtype=np.float64)

        bin_swath_to_grid(
            time_edges, lon_edges, lat_edges,
            time_flat, lon_flat, lat_flat, aod_flat,
            count_grid, data_grid,
        )

        if (i + 1) % 25 == 0:
            print(f"    ... {i + 1}/{len(granules)} granules, {n_valid_total:,} valid pixels")

    elapsed = time.time() - t0
    print(f"  Binning complete: {n_valid_total:,} pixels in {elapsed:.1f}s")

    normalize_grid(count_grid, data_grid)

    # Squeeze out the time dimension → (nlon, nlat)
    return data_grid[0], count_grid[0]


# ---------------------------------------------------------------------------
# 4. Lon shift helper for cartopy display
# ---------------------------------------------------------------------------

def shift_lon_to_180(lon: np.ndarray, *arrays: np.ndarray):
    """Shift lon from 0..360 to -180..180 and reorder arrays accordingly.

    Returns (lon_shifted, *arrays_reordered).
    """
    lon = lon.copy()
    shift_idx = lon >= 180
    lon[shift_idx] -= 360
    sort_idx = np.argsort(lon)
    lon = lon[sort_idx]
    return (lon,) + tuple(a[sort_idx, :] for a in arrays)


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def make_plots(
    date_str: str,
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    obs_data: np.ndarray,
    count_data: np.ndarray,
    base_data: np.ndarray,
    newdust_data: np.ndarray,
) -> None:
    """Generate 2 figures for one day: 2x3 AOD+bias panel, pixel count."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Shift lon for display
    lon, obs_d, base_d, newdust_d, count_d = shift_lon_to_180(
        lon_centers, obs_data, base_data, newdust_data, count_data.astype(np.float64),
    )
    lat = lat_centers

    def add_map_features(ax):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_global()

    # --- Figure A: 2x3 AOD + bias panel ---
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(
        2, 3, figsize=(20, 9),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Top row: AOD via contourf (CERES-SARB pattern: turbo + levels + extend='max')
    # Non-uniform levels — fine resolution at low end, 0.05 steps above
    # (matches CERES-SARB _compute_levels convention)
    aod_levels = np.array([
        0.00, 0.005, 0.01, 0.02, 0.03, 0.04,
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90, 0.95, 1.00,
    ])
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    aod_panels = [
        (axes[0, 0], obs_d, "(a) MODIS Terra+Aqua AOD"),
        (axes[0, 1], base_d, "(b) CAM6 Base AODVIS"),
        (axes[0, 2], newdust_d, "(c) CAM6 New Dust AODVIS"),
    ]
    for ax, data, title in aod_panels:
        add_map_features(ax)
        # Rasterized base layer eliminates white contour-line artifacts in PDFs
        ax.pcolormesh(
            lon, lat, data.T, cmap=plt.cm.turbo,
            vmin=aod_levels[0], vmax=aod_levels[-1],
            transform=ccrs.PlateCarree(), rasterized=True,
        )
        cf_aod = ax.contourf(
            lon_mesh, lat_mesh, data.T, aod_levels,
            cmap=plt.cm.turbo, extend="max",
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(title)

    # Bottom row: bias via contourf
    bias_base = base_d - obs_d
    bias_newdust = newdust_d - obs_d
    bias_model_diff = newdust_d - base_d
    bias_levels = np.arange(-0.30, 0.35, 0.05)
    bias_panels = [
        (axes[1, 0], bias_base, "(d) Base − MODIS"),
        (axes[1, 1], bias_newdust, "(e) New Dust − MODIS"),
        (axes[1, 2], bias_model_diff, "(f) New Dust − Base"),
    ]
    for ax, data, title in bias_panels:
        add_map_features(ax)
        ax.pcolormesh(
            lon, lat, data.T, cmap=plt.cm.RdBu_r,
            vmin=bias_levels[0], vmax=bias_levels[-1],
            transform=ccrs.PlateCarree(), rasterized=True,
        )
        cf_bias = ax.contourf(
            lon_mesh, lat_mesh, data.T, bias_levels,
            cmap=plt.cm.RdBu_r, extend="both",
            transform=ccrs.PlateCarree(),
        )
        ax.set_title(title)

    # Colorbars — one per row
    cb_aod = fig.colorbar(cf_aod, ax=axes[0, :].tolist(), orientation="horizontal",
                          shrink=0.5, label="AOD", pad=0.05, aspect=40)
    cb_aod.set_ticks(np.arange(0, 1.05, 0.20))
    cb_aod.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    cb_bias = fig.colorbar(cf_bias, ax=axes[1, :].tolist(), orientation="horizontal",
                           shrink=0.5, label="AOD Difference", pad=0.05, aspect=40)
    cb_bias.set_ticks(np.arange(-0.30, 0.35, 0.10))
    cb_bias.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    fig.suptitle(f"MODIS vs CAM6 AOD — {date_str}", fontsize=20)
    out_png = OUTPUT_DIR / f"modis_cam6_aod_{date_str}.png"
    out_pdf = OUTPUT_DIR / f"modis_cam6_aod_{date_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)

    # --- Figure C: Pixel count ---
    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    add_map_features(ax)

    count_display = count_d.copy()
    count_display[count_display == 0] = np.nan
    im = ax.pcolormesh(
        lon, lat, count_display.T, cmap="YlOrBr",
        vmin=1, transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"MODIS Pixel Count per Grid Cell — {date_str}")
    fig.colorbar(im, ax=ax, orientation="horizontal",
                 shrink=0.5, label="Pixel Count", pad=0.05, aspect=40)
    plt.tight_layout()
    out_png = OUTPUT_DIR / f"modis_pixel_count_{date_str}.png"
    out_pdf = OUTPUT_DIR / f"modis_pixel_count_{date_str}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    apply_ncar_style()

    dates = [d for _, d in DAYS]

    # Load both CAM6 runs (all 3 days at once)
    ds_base, ds_newdust = read_cam6_models(dates)

    lat_centers = ds_base["lat"].values
    lon_centers = ds_base["lon"].values

    for day_of_year, date_str in DAYS:
        print(f"\n{'='*60}")
        print(f"Processing {date_str} (day {day_of_year})")
        print(f"{'='*60}")

        # Read and bin MODIS for this day
        granules = read_modis_granules(day_of_year)
        obs_data, count_data = grid_modis_day(
            granules, lat_centers, lon_centers, date_str,
        )

        # Extract CAM6 for this day — (nlat, nlon) → transpose to (nlon, nlat)
        base_aod = ds_base[CAM6_VAR].sel(time=date_str, method="nearest").values.T
        newdust_aod = ds_newdust[CAM6_VAR].sel(time=date_str, method="nearest").values.T

        valid = np.isfinite(obs_data)
        print(f"  Grid cells with obs: {valid.sum():,}")
        if valid.any():
            print(f"  Obs AOD range: [{obs_data[valid].min():.4f}, {obs_data[valid].max():.4f}]")
        print(f"  Max pixel count: {count_data.max()}")

        make_plots(
            date_str, lon_centers, lat_centers,
            obs_data, count_data, base_aod, newdust_aod,
        )

    print(f"\nSmoke test complete! {len(DAYS) * 2} figures in {OUTPUT_DIR}")
