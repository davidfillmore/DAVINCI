#!/usr/bin/env python
"""
Compute NO2 tropospheric column from CESM 3D output.

Integrates NO2 mixing ratio vertically using hybrid pressure coordinates
to produce a 2D NO2 column field comparable to Pandora observations.

Usage:
    python compute_no2_column.py
"""

from glob import glob
from pathlib import Path

import numpy as np
import xarray as xr

# Constants
G = 9.80665  # gravity [m/s²]
M_AIR = 0.0289644  # molar mass of dry air [kg/mol]
P0 = 100000.0  # reference pressure [Pa]
TROP_P_MIN = 20000.0  # tropopause pressure threshold [Pa] (~200 hPa)

# Input/output paths
MODEL_DIR = Path("/glade/derecho/scratch/fillmore/ASIA-AQ/model")
OUTPUT_DIR = Path("/glade/derecho/scratch/fillmore/ASIA-AQ/obs")

# Time range
START_DATE = "2024-02-01"
END_DATE = "2024-02-29"


def compute_pressure_levels(ds: xr.Dataset) -> xr.DataArray:
    """Compute pressure at each level from hybrid coordinates.

    P(k) = hyam(k) * P0 + hybm(k) * PS

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with hyam, hybm, PS variables.

    Returns
    -------
    xr.DataArray
        Pressure at each level [Pa], dims (time, lev, lat, lon).
    """
    hyam = ds["hyam"]  # hybrid A coefficient at midpoints
    hybm = ds["hybm"]  # hybrid B coefficient at midpoints
    ps = ds["PS"]  # surface pressure [Pa]

    # Broadcast to full dimensions
    # hyam/hybm are (lev,), PS is (time, lat, lon)
    pressure = hyam * P0 + hybm * ps

    return pressure


def compute_no2_column(ds: xr.Dataset, trop_p_min: float = TROP_P_MIN) -> xr.DataArray:
    """Compute NO2 tropospheric column from 3D field.

    Column (mol/m²) = sum over levels of: NO2 (mol/mol) * dp / (g * M_air)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with NO2, hyam, hybm, hyai, hybi, PS.
    trop_p_min : float
        Minimum pressure for troposphere [Pa]. Levels with P < trop_p_min
        are considered stratospheric and excluded.

    Returns
    -------
    xr.DataArray
        NO2 tropospheric column [mol/m²], dims (time, lat, lon).
    """
    # Get pressure at level interfaces
    hyai = ds["hyai"]  # hybrid A at interfaces
    hybi = ds["hybi"]  # hybrid B at interfaces
    ps = ds["PS"]

    # Pressure at interfaces: (ilev,) broadcast with (time, lat, lon)
    p_int = hyai * P0 + hybi * ps  # (time, ilev, lat, lon)

    # Pressure difference across each layer (positive = downward)
    # dp[k] = p_int[k+1] - p_int[k]
    dp = p_int.diff(dim="ilev")
    dp = dp.rename({"ilev": "lev"})

    # Get midpoint pressures for troposphere mask
    p_mid = compute_pressure_levels(ds)

    # Mask stratosphere (P < trop_p_min)
    trop_mask = p_mid >= trop_p_min

    # Get NO2 mixing ratio
    no2 = ds["NO2"]  # mol/mol

    # Apply troposphere mask
    no2_trop = no2.where(trop_mask, 0.0)
    dp_trop = dp.where(trop_mask, 0.0)

    # Compute column: sum(NO2 * dp) / (g * M_air)
    # Units: (mol/mol) * (Pa) / (m/s² * kg/mol) = mol/m²
    no2_column = (no2_trop * dp_trop).sum(dim="lev") / (G * M_AIR)

    no2_column.attrs = {
        "long_name": "NO2 tropospheric vertical column",
        "units": "mol/m2",
        "tropopause_pressure": f"{trop_p_min} Pa",
    }

    return no2_column


def main():
    print("=" * 60)
    print("CESM NO2 Tropospheric Column Computation")
    print("=" * 60)
    print()

    # Find model files
    pattern = str(MODEL_DIR / "f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.2024-02-*.nc")
    files = sorted(glob(pattern))

    print(f"Found {len(files)} model files")
    print(f"First: {Path(files[0]).name}")
    print(f"Last:  {Path(files[-1]).name}")
    print()

    # Open dataset
    print("Loading model data...")
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=True,
        chunks={"time": 24},  # chunk by day
    )

    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"Grid: {len(ds.lat)} x {len(ds.lon)}")
    print(f"Levels: {len(ds.lev)}")
    print()

    # Check required variables
    required = ["NO2", "PS", "hyam", "hybm", "hyai", "hybi"]
    missing = [v for v in required if v not in ds]
    if missing:
        print(f"ERROR: Missing variables: {missing}")
        return

    # Compute column
    print("Computing NO2 tropospheric column...")
    print(f"  Tropopause threshold: {TROP_P_MIN/100:.0f} hPa")

    no2_column = compute_no2_column(ds, trop_p_min=TROP_P_MIN)

    # Compute to get actual values (this triggers Dask computation)
    print("  Computing (this may take a while)...")
    no2_column = no2_column.compute()

    print()
    print(f"Column shape: {no2_column.shape}")
    print(f"Column range: {float(no2_column.min()):.2e} to {float(no2_column.max()):.2e} mol/m²")
    print()

    # Create output dataset
    ds_out = xr.Dataset(
        {
            "NO2_column": no2_column,
        },
        coords={
            "time": ds.time,
            "lat": ds.lat,
            "lon": ds.lon,
        },
    )

    ds_out.attrs = {
        "source": "CESM/CAM-chem f.e3b06m.FCnudged.t6s.01x01.01",
        "processing": "compute_no2_column.py",
        "description": "NO2 tropospheric column integrated from 3D field",
        "tropopause_pressure_Pa": TROP_P_MIN,
    }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR / f"cesm_no2_column_{START_DATE.replace('-', '')}_{END_DATE.replace('-', '')}.nc"
    )

    print(f"Saving to {output_file}...")
    ds_out.to_netcdf(output_file)

    print()
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
