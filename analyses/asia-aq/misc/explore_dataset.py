#!/usr/bin/env python
"""
Explore CESM/CAM ASIA-AQ dataset output.

This script examines the dataset data structure, variables, and coverage
to inform configuration for dataset-dataset comparisons.
"""

import os
from pathlib import Path

import numpy as np
import xarray as xr

# Data location from env var or default to ~/Data/ASIA-AQ
DATA_DIR = Path(os.environ.get("ASIA_AQ_DATA", Path.home() / "Data" / "ASIA-AQ"))
FILE_PATTERN = "f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.*.nc"


def explore_single_file(filepath: Path) -> xr.Dataset:
    """Load and describe a single file."""
    print(f"\n{'='*60}")
    print(f"File: {filepath.name}")
    print("=" * 60)

    ds = xr.open_dataset(filepath)

    print("\n--- Dimensions ---")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")

    print("\n--- Coordinates ---")
    for name, coord in ds.coords.items():
        if coord.size <= 10:
            print(f"  {name}: {coord.values}")
        else:
            print(f"  {name}: [{coord.values[0]:.2f} ... {coord.values[-1]:.2f}] ({coord.size} values)")

    print("\n--- Data Variables ---")
    for name, var in ds.data_vars.items():
        attrs = var.attrs
        long_name = attrs.get("long_name", "")
        units = attrs.get("units", "")
        print(f"  {name:12s} {str(var.dims):35s} [{units:10s}] {long_name}")

    return ds


def explore_grid(ds: xr.Dataset) -> None:
    """Examine the grid structure."""
    print("\n--- Grid Information ---")

    lat = ds.lat.values
    lon = ds.lon.values

    print(f"  Latitude:  {lat.min():.2f} to {lat.max():.2f} ({len(lat)} points)")
    print(f"  Longitude: {lon.min():.2f} to {lon.max():.2f} ({len(lon)} points)")

    # Grid spacing
    dlat = np.diff(lat)
    dlon = np.diff(lon)
    print(f"  Lat spacing: {dlat.mean():.4f} deg (std: {dlat.std():.6f})")
    print(f"  Lon spacing: {dlon.mean():.4f} deg (std: {dlon.std():.6f})")

    # Vertical levels
    if "lev" in ds.dims:
        lev = ds.lev.values
        print(f"\n  Vertical levels: {len(lev)}")
        print(f"  Level values (hybrid sigma): {lev[0]:.4f} to {lev[-1]:.4f}")

        # If we have pressure, show it
        if "PMID" in ds:
            pmid = ds.PMID.isel(time=0)
            # Surface pressure approx
            p_sfc = pmid.isel(lev=-1).mean().values / 100  # hPa
            p_top = pmid.isel(lev=0).mean().values / 100
            print(f"  Pressure range: ~{p_top:.1f} to ~{p_sfc:.1f} hPa")


def explore_time_coverage(data_dir: Path, pattern: str) -> None:
    """Examine temporal coverage across all files."""
    print("\n--- Temporal Coverage ---")

    files = sorted(data_dir.glob(pattern))
    print(f"  Total files: {len(files)}")

    if not files:
        print("  No files found!")
        return

    # Parse times from filenames
    # Format: ...cam.h2i.YYYY-MM-DD-SSSSS.nc
    times = []
    for f in files:
        parts = f.stem.split(".")
        date_part = parts[-1]  # YYYY-MM-DD-SSSSS
        date_str, seconds = date_part.rsplit("-", 1)
        times.append((date_str, int(seconds)))

    # Unique dates
    dates = sorted(set(t[0] for t in times))
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Unique dates: {len(dates)}")

    # Time steps per day
    files_per_day = len(times) // len(dates)
    print(f"  Files per day: {files_per_day} (hourly output)")


def explore_variable_stats(ds: xr.Dataset, variables: list[str]) -> None:
    """Compute basic statistics for key variables."""
    print("\n--- Variable Statistics (single timestep) ---")

    for var in variables:
        if var not in ds:
            print(f"  {var}: not found")
            continue

        data = ds[var]
        units = data.attrs.get("units", "")

        # For 3D variables, look at surface (last level)
        if "lev" in data.dims:
            data = data.isel(lev=-1)

        values = data.values.flatten()
        values = values[~np.isnan(values)]

        if len(values) == 0:
            print(f"  {var}: all NaN")
            continue

        print(f"  {var:12s}: min={values.min():.3e}, max={values.max():.3e}, "
              f"mean={values.mean():.3e} [{units}]")


def main():
    """Main exploration routine."""
    print("CESM/CAM ASIA-AQ Dataset Data Exploration")
    print("=" * 60)

    # Check data directory
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        return

    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    if not files:
        print(f"ERROR: No files matching pattern: {FILE_PATTERN}")
        return

    # Explore first file
    ds = explore_single_file(files[0])

    # Grid information
    explore_grid(ds)

    # Time coverage
    explore_time_coverage(DATA_DIR, FILE_PATTERN)

    # Variable statistics
    key_vars = ["O3", "NO", "NO2", "CO", "CH2O", "PM25", "AODVISdn", "PS", "Z3"]
    explore_variable_stats(ds, key_vars)

    ds.close()

    # Load multiple files to check time axis
    print("\n--- Multi-file Time Axis ---")
    ds_multi = xr.open_mfdataset(
        files[:24],  # First 24 hours
        combine="by_coords",
        parallel=False,
    )
    print(f"  Combined time range: {ds_multi.time.values[0]} to {ds_multi.time.values[-1]}")
    print(f"  Time steps: {len(ds_multi.time)}")
    ds_multi.close()

    print("\n" + "=" * 60)
    print("Exploration complete.")


if __name__ == "__main__":
    main()
