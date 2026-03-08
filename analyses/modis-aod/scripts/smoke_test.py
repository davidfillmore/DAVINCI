"""Smoke test: MODIS AOD vs two CAM6 runs (base + new dust).

Reproduces Figure 3.1 from Buchholz et al. — Australian bushfire event,
Dec 21-23 2019. For each day, generates:
  - 3-panel AOD comparison (MODIS / Base / New Dust)
  - 2-panel bias maps (Base-MODIS / NewDust-MODIS)
  - Pixel count map

Usage:
    conda activate davinci-monet
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

MODIS_BASE = Path.home() / "Data" / "MODIS" / "Terra" / "C61" / "2019"
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
    """Read MODIS L2 granules for a single day via monetio."""
    import monetio.sat._modis_l2_mm as modis_reader

    modis_dir = MODIS_BASE / str(day_of_year)
    variable_dict = {
        MODIS_VAR: {"minimum": 0.0, "maximum": 10.0, "scale": 0.001},
    }

    file_pattern = str(modis_dir / "MOD04_L2.*.hdf")
    print(f"\nReading MODIS granules from {modis_dir}...")
    t0 = time.time()
    granules = modis_reader.read_mfdataset(file_pattern, variable_dict)
    elapsed = time.time() - t0
    print(f"  Read {len(granules)} granules in {elapsed:.1f}s")
    return granules


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

    for i, (datetime_str, granule) in enumerate(granules.items()):
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
    """Generate 3 figures for one day."""
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

    # --- Figure A: 3-panel AOD comparison ---
    fig, axes = plt.subplots(
        1, 3, figsize=(20, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    vmax = 0.8
    panels = [
        (axes[0], obs_d, "MODIS Terra AOD"),
        (axes[1], base_d, "CAM6 Base AODVIS"),
        (axes[2], newdust_d, "CAM6 New Dust AODVIS"),
    ]
    for ax, data, title in panels:
        add_map_features(ax)
        im = ax.pcolormesh(
            lon, lat, data.T, cmap="viridis",
            vmin=0, vmax=vmax, transform=ccrs.PlateCarree(),
        )
        ax.set_title(title)

    # Horizontal colorbar beneath all panels
    fig.colorbar(im, ax=axes.tolist(), orientation="horizontal",
                 shrink=0.5, label="AOD", pad=0.05, aspect=40)
    fig.suptitle(f"MODIS vs CAM6 AOD — {date_str} (Terra)", fontsize=14)
    out = OUTPUT_DIR / f"modis_cam6_aod_comparison_{date_str}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)

    # --- Figure B: 2-panel bias maps ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    from matplotlib.colors import TwoSlopeNorm

    bias_base = base_d - obs_d
    bias_newdust = newdust_d - obs_d
    vlim = 0.3
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

    for ax, bias, title in [
        (ax1, bias_base, "Bias (Base − MODIS)"),
        (ax2, bias_newdust, "Bias (New Dust − MODIS)"),
    ]:
        add_map_features(ax)
        im = ax.pcolormesh(
            lon, lat, bias.T, cmap="RdBu_r",
            norm=norm, transform=ccrs.PlateCarree(),
        )
        ax.set_title(title)

    fig.colorbar(im, ax=[ax1, ax2], orientation="horizontal",
                 shrink=0.5, label="AOD Bias", pad=0.05, aspect=40)
    fig.suptitle(f"AOD Bias (Model − MODIS) — {date_str}", fontsize=14)
    out = OUTPUT_DIR / f"modis_cam6_aod_bias_{date_str}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
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
    out = OUTPUT_DIR / f"modis_pixel_count_{date_str}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
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

    print(f"\nSmoke test complete! {len(DAYS) * 3} figures in {OUTPUT_DIR}")
