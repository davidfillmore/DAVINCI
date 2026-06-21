#!/usr/bin/env python
"""
Preprocess Pandora L2 files for ASIA-AQ analysis.

Reads raw Pandora L2 txt files and outputs a single NetCDF file
with NO2 tropospheric column data filtered by quality and time.

Usage:
    python preprocess_pandora.py
"""

from glob import glob
from pathlib import Path

import xarray as xr

from davinci_monet.datasets.surface.pandora import PandoraReader

# Input/output paths
PANDORA_DIR = Path("/glade/campaign/acom/acom-weather/emmons/ASIAAQ_obs/Pandora")
OUTPUT_DIR = Path("/glade/derecho/scratch/fillmore/ASIA-AQ/obs")

# Time range
START_TIME = "2024-02-01"
END_TIME = "2024-02-29"

# Quality filters
QUALITY_FLAG_MAX = 1  # 0=high, 1=high+medium
SOLAR_ZENITH_MAX = 80.0  # degrees


def main():
    print("=" * 60)
    print("Pandora L2 Preprocessing for ASIA-AQ")
    print("=" * 60)
    print()

    # Find all L2 files
    pattern = str(PANDORA_DIR / "Pandora*_L2_*.txt")
    files = sorted(glob(pattern))

    # Exclude Boulder (not in ASIA-AQ domain)
    files = [f for f in files if "Boulder" not in f]

    print(f"Found {len(files)} Pandora L2 files (excluding Boulder)")
    for f in files:
        print(f"  {Path(f).name}")
    print()

    # Read and process
    print(f"Time range: {START_TIME} to {END_TIME}")
    print(f"Quality filter: flag <= {QUALITY_FLAG_MAX}")
    print(f"Solar zenith filter: <= {SOLAR_ZENITH_MAX}°")
    print()

    reader = PandoraReader()

    try:
        ds = reader.open(
            file_paths=files,
            quality_flag_max=QUALITY_FLAG_MAX,
            solar_zenith_max=SOLAR_ZENITH_MAX,
            start_time=START_TIME,
            end_time=END_TIME,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return

    print("Dataset summary:")
    print(ds)
    print()

    # Show site info
    print("Sites:")
    for i, site in enumerate(ds.site.values):
        lat = float(ds.latitude.values[i])
        lon = float(ds.longitude.values[i])
        n_obs = int(ds.no2_trop_column.sel(site=site).notnull().sum())
        print(f"  {site:20s} ({lat:6.2f}°N, {lon:7.2f}°E) - {n_obs} obs")
    print()

    # Statistics
    no2_col = ds.no2_trop_column.values
    valid = ~xr.DataArray(no2_col).isnull()
    print(f"Total valid observations: {int(valid.sum())}")
    print(
        f"NO2 column range: {float(ds.no2_trop_column.min()):.2e} to {float(ds.no2_trop_column.max()):.2e} mol/m²"
    )
    print()

    # Save to NetCDF
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR
        / f"pandora_no2_column_{START_TIME.replace('-', '')}_{END_TIME.replace('-', '')}.nc"
    )

    # Add processing metadata
    ds.attrs["preprocessing"] = "preprocess_pandora.py"
    ds.attrs["quality_flag_max"] = QUALITY_FLAG_MAX
    ds.attrs["solar_zenith_max"] = SOLAR_ZENITH_MAX
    ds.attrs["start_time"] = START_TIME
    ds.attrs["end_time"] = END_TIME

    ds.to_netcdf(output_file)
    print(f"Saved: {output_file}")
    print()
    print("=" * 60)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
