"""Smoke test: MODIS AOD vs CAM6 AODVIS.

Follows the MELODIES-MONET pattern exactly:
  1. read_mfdataset → OrderedDict of granules keyed by datetime_str
  2. setup grid edges from model lat/lon
  3. loop over granules, call numba update_data_grid per granule
  4. normalize
  5. plot

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

MODIS_DIR = Path.home() / "Data" / "MODIS" / "Terra" / "C61" / "2019" / "355"
CAM6_FILE = (
    Path.home() / "Data" / "CAM6"
    / "FCnudged_f09.mam.BaseMar27.2019_2021.001_AODVIS.nc"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MODIS_VAR = "AOD_550_Dark_Target_Deep_Blue_Combined"
CAM6_VAR = "AODVIS"


# ---------------------------------------------------------------------------
# 1. Read MODIS granules — MELODIES-MONET pattern
# ---------------------------------------------------------------------------

def read_modis_granules() -> OrderedDict:
    """Read MODIS L2 granules via monetio.read_mfdataset.

    Returns OrderedDict keyed by datetime_str, each value an xr.Dataset
    with lat, lon, time coords and AOD variable.
    """
    import monetio.sat._modis_l2_mm as modis_reader

    variable_dict = {
        MODIS_VAR: {"minimum": 0.0, "maximum": 10.0, "scale": 0.001},
    }

    file_pattern = str(MODIS_DIR / "MOD04_L2.*.hdf")
    print(f"Reading MODIS granules from {MODIS_DIR}...")
    t0 = time.time()
    granules = modis_reader.read_mfdataset(file_pattern, variable_dict)
    elapsed = time.time() - t0
    print(f"Read {len(granules)} granules in {elapsed:.1f}s")
    return granules


# ---------------------------------------------------------------------------
# 2. Read CAM6 model
# ---------------------------------------------------------------------------

def read_cam6() -> xr.Dataset:
    """Read CAM6 and subset to the MODIS date."""
    print(f"Reading CAM6: {CAM6_FILE}")
    ds = xr.open_dataset(str(CAM6_FILE))
    ds = ds.sel(time="2019-12-21", method="nearest")
    ds = ds.expand_dims("time")
    print(f"CAM6 grid: lat={ds.sizes['lat']}, lon={ds.sizes['lon']}")
    print(f"CAM6 AODVIS range: [{float(ds[CAM6_VAR].min()):.4f}, {float(ds[CAM6_VAR].max()):.4f}]")
    return ds


# ---------------------------------------------------------------------------
# 3. Grid MODIS onto model grid — MELODIES-MONET pattern
# ---------------------------------------------------------------------------

def grid_modis(granules: OrderedDict, model: xr.Dataset) -> xr.Dataset:
    """Bin MODIS granules onto CAM6 grid.

    Follows MELODIES-MONET driver.py:
      setup_obs_grid → loop update_obs_gridded_data → normalize
    """
    lat_centers = model["lat"].values
    lon_centers = model["lon"].values
    nlat = len(lat_centers)
    nlon = len(lon_centers)

    # Grid edges from model centers
    lat_edges = edges_from_centers(lat_centers)
    lon_edges = edges_from_centers(lon_centers)

    # Single time bin for one day
    t0_epoch = pd.Timestamp("2019-12-21").timestamp()
    t1_epoch = pd.Timestamp("2019-12-22").timestamp()
    time_edges = np.array([t0_epoch, t1_epoch], dtype=np.float64)
    ntime = 1

    # Allocate — same as MELODIES-MONET: (ntime, nlon, nlat)
    count_grid = np.zeros((ntime, nlon, nlat), dtype=np.int32)
    data_grid = np.zeros((ntime, nlon, nlat), dtype=np.float64)

    # Loop over granules — exactly as update_obs_gridded_data does
    print(f"Binning {len(granules)} granules onto {nlon}x{nlat} grid...")
    t0 = time.time()
    n_valid_total = 0

    for i, (datetime_str, granule) in enumerate(granules.items()):
        # Parse granule timestamp — MELODIES-MONET uses format '%Y%j%H%M'
        obs_timestamp = pd.to_datetime(datetime_str, format='%Y%j%H%M').timestamp()

        aod = granule[MODIS_VAR].values
        lat = granule["lat"].values
        lon = granule["lon"].values

        # Flatten and filter fill values (lat/lon = -999 in MODIS)
        aod_flat = aod.flatten().astype(np.float64)
        lat_flat = lat.flatten().astype(np.float64)
        lon_flat = lon.flatten().astype(np.float64)

        # Mask fill values by setting to NaN (binning skips NaN)
        fill_mask = (lat_flat < -900) | (lon_flat < -900)
        aod_flat[fill_mask] = np.nan

        # Shift lon from -180..180 to 0..360 to match model grid
        lon_flat = np.where(lon_flat < 0, lon_flat + 360.0, lon_flat)

        # Wrap pixels past the last lon edge back into the first bin
        # (prevents clamping artifact at the 360/0 boundary)
        lon_flat = np.where(lon_flat >= lon_edges[-1], lon_flat - 360.0, lon_flat)

        n_valid = np.isfinite(aod_flat).sum()
        n_valid_total += n_valid

        # Time array — all pixels get same granule timestamp
        n_obs = len(aod_flat)
        time_flat = np.full(n_obs, obs_timestamp, dtype=np.float64)

        # Bin — numba fast path, accumulates in-place
        bin_swath_to_grid(
            time_edges, lon_edges, lat_edges,
            time_flat, lon_flat, lat_flat, aod_flat,
            count_grid, data_grid,
        )

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(granules)} granules, {n_valid_total:,} valid pixels")

    elapsed = time.time() - t0
    print(f"Binning complete: {n_valid_total:,} pixels in {elapsed:.1f}s")

    # Normalize — same as MELODIES-MONET normalize_data_grid
    normalize_grid(count_grid, data_grid)

    # Build xr.Dataset
    obs_gridded = xr.Dataset(
        {
            f"obs_{MODIS_VAR}": (["time", "lon", "lat"], data_grid.astype(np.float32)),
            "obs_count": (["time", "lon", "lat"], count_grid),
        },
        coords={
            "time": [pd.Timestamp("2019-12-21")],
            "lon": lon_centers,
            "lat": lat_centers,
        },
    )

    # Add model on same grid
    model_aod = model[CAM6_VAR].values  # (1, nlat, nlon)
    # Transpose to (time, lon, lat) to match obs layout
    model_aod_tlonlat = model_aod.transpose(0, 2, 1)  # (1, nlon, nlat)
    obs_gridded[f"model_{CAM6_VAR}"] = (["time", "lon", "lat"], model_aod_tlonlat.astype(np.float32))

    valid = np.isfinite(obs_gridded[f"obs_{MODIS_VAR}"].values)
    print(f"Grid cells with obs: {valid.sum():,} / {count_grid.size:,}")
    print(f"Obs AOD range: [{data_grid[valid].min():.4f}, {data_grid[valid].max():.4f}]")
    print(f"Max pixel count per cell: {count_grid.max()}")

    return obs_gridded


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

def make_plots(paired: xr.Dataset) -> None:
    """Generate spatial distribution and bias plots."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    apply_ncar_style()

    obs_var = f"obs_{MODIS_VAR}"
    model_var = f"model_{CAM6_VAR}"

    obs_data = paired[obs_var].isel(time=0).values
    model_data = paired[model_var].isel(time=0).values
    bias = model_data - obs_data
    lon = paired["lon"].values.copy()
    lat = paired["lat"].values

    # Shift lon from 0..360 to -180..180 for cartopy display
    shift_idx = lon >= 180
    lon[shift_idx] -= 360
    sort_idx = np.argsort(lon)
    lon = lon[sort_idx]
    obs_data = obs_data[sort_idx, :]
    model_data = model_data[sort_idx, :]
    bias = bias[sort_idx, :]

    # --- Figure 1: Obs + Model side by side ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for ax in (ax1, ax2):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_global()

    vmax = 0.8
    im1 = ax1.pcolormesh(
        lon, lat, obs_data.T, cmap="YlOrRd",
        vmin=0, vmax=vmax, transform=ccrs.PlateCarree(),
    )
    ax1.set_title("MODIS Terra AOD (binned to CAM6 grid)")
    fig.colorbar(im1, ax=ax1, shrink=0.7, label="AOD")

    im2 = ax2.pcolormesh(
        lon, lat, model_data.T, cmap="YlOrRd",
        vmin=0, vmax=vmax, transform=ccrs.PlateCarree(),
    )
    ax2.set_title("CAM6 AODVIS")
    fig.colorbar(im2, ax=ax2, shrink=0.7, label="AOD")

    fig.suptitle("MODIS vs CAM6 AOD — 2019-12-21 (Terra)", fontsize=14)
    plt.tight_layout()
    out1 = OUTPUT_DIR / "modis_cam6_aod_comparison.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")
    plt.close(fig)

    # --- Figure 2: Bias ---
    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()

    vlim = 0.3
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    im = ax.pcolormesh(
        lon, lat, bias.T, cmap="RdBu_r",
        norm=norm, transform=ccrs.PlateCarree(),
    )
    ax.set_title("AOD Bias (CAM6 - MODIS) — 2019-12-21")
    fig.colorbar(im, ax=ax, shrink=0.7, label="AOD Bias")
    plt.tight_layout()
    out2 = OUTPUT_DIR / "modis_cam6_aod_bias.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig)

    # --- Figure 3: Pixel count ---
    fig, ax = plt.subplots(
        figsize=(10, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()

    count = paired["obs_count"].isel(time=0).values[sort_idx, :].astype(np.float64)
    count[count == 0] = np.nan  # white for no data
    im = ax.pcolormesh(
        lon, lat, count.T, cmap="YlOrBr",
        vmin=1, transform=ccrs.PlateCarree(),
    )
    ax.set_title("MODIS Pixel Count per Grid Cell — 2019-12-21")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pixel Count")
    plt.tight_layout()
    out3 = OUTPUT_DIR / "modis_pixel_count.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    granules = read_modis_granules()
    model = read_cam6()
    paired = grid_modis(granules, model)
    make_plots(paired)
    print("\nSmoke test complete!")
