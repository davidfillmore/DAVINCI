#!/usr/bin/env python
"""
Run CESM model evaluation against AirNow, AERONET, and Pandora observations.

This script uses the DAVINCI-MONET pipeline to:
1. Load CESM model data (with derived NO2 column)
2. Load AirNow, AERONET, and Pandora observations
3. Pair model with observations
4. Generate comparison plots
5. Calculate statistics

Usage:
    python run_evaluation.py

Or use the CLI directly:
    davinci-monet run ../configs/asia-aq.yaml
"""

import os
from glob import glob
from pathlib import Path

import xarray as xr

from davinci_monet.models.cesm import compute_tropospheric_column
from davinci_monet.pipeline.runner import run_analysis

# Data directory from env var or default to ~/Data/ASIA-AQ
ASIA_AQ_DATA = Path(os.environ.get("ASIA_AQ_DATA", Path.home() / "Data" / "ASIA-AQ"))

# Analysis directory (where this script lives)
ASIA_AQ_ANALYSIS = Path(__file__).parent.parent.resolve()

# Set env vars for config file expansion
os.environ.setdefault("ASIA_AQ_DATA", str(ASIA_AQ_DATA))
os.environ.setdefault("ASIA_AQ_ANALYSIS", str(ASIA_AQ_ANALYSIS))


def precompute_no2_column(model_files: list[str], output_path: Path) -> None:
    """Precompute NO2 tropospheric column from CESM model data.

    Parameters
    ----------
    model_files
        List of CESM model file paths.
    output_path
        Path to save the NO2 column dataset.
    """
    print("Computing NO2 tropospheric column from model data...")

    columns = []
    for fpath in sorted(model_files):
        ds = xr.open_dataset(fpath)

        # Compute NO2 column (mol/m2)
        no2_col = compute_tropospheric_column(ds, "NO2", z_dim="lev")
        no2_col.name = "NO2_column"

        columns.append(no2_col)
        ds.close()

    # Concatenate along time
    no2_column_da = xr.concat(columns, dim="time")

    # Create dataset
    ds_out = xr.Dataset({"NO2_column": no2_column_da})
    ds_out.attrs["description"] = "CESM NO2 tropospheric column (vertically integrated)"
    ds_out.attrs["units"] = "mol/m2"

    # Save
    ds_out.to_netcdf(output_path)
    print(f"  Saved NO2 column to: {output_path}")
    print(f"  Shape: {no2_column_da.shape}")
    print(f"  Mean: {float(no2_column_da.mean()):.2e} mol/m2")


def main():
    """Run the ASIA-AQ model evaluation pipeline."""
    # Paths
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "configs" / "asia-aq.yaml"
    data_dir = base_dir / "data"
    no2_column_path = data_dir / "cesm_no2_column_20240201_20240229.nc"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    print("=" * 70)
    print("CESM/CAM-chem ASIA-AQ Model Evaluation")
    print("=" * 70)
    print(f"\nUsing config: {config_path}")
    print()

    # Precompute NO2 column if needed
    if not no2_column_path.exists():
        model_files = glob(str(ASIA_AQ_DATA / "CAM" / "f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.2024-02-*.nc"))
        if model_files:
            precompute_no2_column(model_files, no2_column_path)
        else:
            print("WARNING: No model files found for NO2 column computation")
    else:
        print(f"Using existing NO2 column file: {no2_column_path}")

    print()

    # Run the pipeline
    result = run_analysis(str(config_path), show_plots=True)

    # Report results
    print()
    print("=" * 70)
    if result.success:
        print("Pipeline completed successfully!")
        print(f"Total time: {result.total_duration_seconds:.1f} seconds")
        print(f"Stages completed: {', '.join(result.completed_stages)}")
    else:
        print("Pipeline failed!")
        for failed in result.failed_stages:
            print(f"  {failed.stage_name}: {failed.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
