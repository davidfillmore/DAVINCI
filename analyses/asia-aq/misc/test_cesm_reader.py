#!/usr/bin/env python
"""
Test CESM/CAM reader with ASIA-AQ dataset output.

This script verifies that the DAVINCI-MONET CESM reader correctly
loads and standardizes the ASIA-AQ dataset output.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import DAVINCI modules
from davinci_monet.datasets.cesm import CESMFVReader

# Data paths
DATA_DIR = Path.home() / "Data" / "ASIA-AQ"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# File pattern for first day only (24 files)
FILE_PATTERN = "f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.2024-02-01-*.nc"


def test_cesm_reader() -> None:
    """Test the CESM reader with ASIA-AQ data."""
    print("Testing DAVINCI-MONET CESM Reader")
    print("=" * 60)

    # Get file list
    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    print(f"Found {len(files)} files for 2024-02-01")

    if not files:
        print("ERROR: No files found!")
        return

    # Load data using the CESM finite-volume reader
    print("\nLoading with CESMFVReader().open()...")
    ds = CESMFVReader().open(
        files,
        variables=["O3", "NO2", "CO", "PM25", "AODVISdn", "PS", "Z3", "PMID"],
    )

    print(f"\nDataset loaded:")
    print(f"  Label: cesm_asiaq")
    print(f"  Dataset type: cesm_fv")

    print(f"\n--- Dataset Structure ---")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Coordinates: {list(ds.coords)}")
    print(f"  Variables: {list(ds.data_vars)}")

    # Check time axis
    print(f"\n--- Time Axis ---")
    print(f"  Start: {ds.time.values[0]}")
    print(f"  End: {ds.time.values[-1]}")
    print(f"  Steps: {len(ds.time)}")

    # Check grid (note: DAVINCI-MONET renames dims and coords)
    print(f"\n--- Grid ---")
    # Handle different coordinate naming conventions
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    lat_vals = ds[lat_coord].values
    lon_vals = ds[lon_coord].values
    print(f"  Lat range: {lat_vals.min():.2f} to {lat_vals.max():.2f}")
    print(f"  Lon range: {lon_vals.min():.2f} to {lon_vals.max():.2f}")
    if "z" in ds.dims:
        print(f"  Vertical levels: {len(ds.z)}")

    # Surface values (lowest dataset level = index 0 for CESM)
    # Note: CESM levels go from surface (z=0) to top (z=-1)
    print(f"\n--- Surface Values (level 0) ---")
    for var in ["O3", "NO2", "CO", "PM25"]:
        if var in ds:
            # Get surface level data
            data = ds[var]
            if "z" in data.dims:
                surface = data.isel(z=0, time=12)  # Noon on first day, surface
            else:
                surface = data.isel(time=12)

            vals = surface.values.flatten()
            vals = vals[~np.isnan(vals)]

            # Convert mol/mol to ppb
            ppb = vals * 1e9

            print(f"  {var}: {ppb.min():.1f} - {ppb.max():.1f} ppb " f"(mean: {ppb.mean():.1f})")

    return ds


def plot_surface_snapshot(ds) -> None:
    """Create a quick-look surface plot."""

    # Get coordinate arrays (handle different naming conventions)
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    lat_vals = ds[lat_coord].values
    lon_vals = ds[lon_coord].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CESM/CAM ASIA-AQ Surface Fields\n2024-02-01 12:00 UTC", fontsize=14)

    variables = [
        ("O3", "Ozone", "RdYlBu_r", 20, 80),
        ("NO2", "NO2", "YlOrRd", 0, 10),
        ("CO", "CO", "YlOrBr", 50, 300),
        ("PM25", "PM2.5", "RdPu", 0, 100),
    ]

    for ax, (var, title, cmap, vmin, vmax) in zip(axes.flat, variables):
        if var not in ds:
            ax.text(0.5, 0.5, f"{var} not found", ha="center", va="center", transform=ax.transAxes)
            continue

        data = ds[var]
        if "z" in data.dims:
            surface = data.isel(z=0, time=12)  # z=0 is surface for CESM
        else:
            surface = data.isel(time=12)

        # Convert to ppb (or ug/m3 for PM25)
        values = surface.values * 1e9

        im = ax.pcolormesh(
            lon_vals,
            lat_vals,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title} (ppb)" if var != "PM25" else f"{title} (approx ug/m3)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Mark campaign locations
        locations = {
            "Seoul": (127.0, 37.5),
            "Manila": (121.0, 14.6),
            "Bangkok": (100.5, 13.75),
            "Taipei": (121.5, 25.0),
        }
        for city, (clon, clat) in locations.items():
            if lon_vals.min() < clon < lon_vals.max() and lat_vals.min() < clat < lat_vals.max():
                ax.plot(clon, clat, "ko", markersize=5)
                ax.text(clon + 1, clat, city, fontsize=8)

    plt.tight_layout()

    # Save PNG and PDF
    outfile_png = OUTPUT_DIR / "cesm_surface_snapshot.png"
    outfile_pdf = OUTPUT_DIR / "cesm_surface_snapshot.pdf"
    plt.savefig(outfile_png, dpi=300, bbox_inches="tight")
    plt.savefig(outfile_pdf, bbox_inches="tight")
    print(f"\nSaved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")
    plt.close()


def plot_time_series(ds) -> None:
    """Create time series at key locations."""

    # Get coordinate arrays (handle different naming conventions)
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    lat_vals = ds[lat_coord].values
    lon_vals = ds[lon_coord].values

    # Determine spatial dimension names
    y_dim = "y" if "y" in ds.dims else "lat"
    x_dim = "x" if "x" in ds.dims else "lon"

    # Campaign locations
    locations = {
        "Seoul": (127.0, 37.5),
        "Manila": (121.0, 14.6),
        "Bangkok": (100.5, 13.75),
        "Taipei": (121.5, 25.0),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("CESM/CAM ASIA-AQ Surface O3 Time Series\n2024-02-01", fontsize=14)

    for ax, (city, (target_lon, target_lat)) in zip(axes.flat, locations.items()):
        # Find nearest grid point - handle 2D lat/lon arrays
        if lat_vals.ndim == 2:
            # 2D coordinate arrays: find minimum distance
            dist = np.sqrt((lat_vals - target_lat) ** 2 + (lon_vals - target_lon) ** 2)
            min_idx = np.unravel_index(dist.argmin(), dist.shape)
            lat_idx, lon_idx = int(min_idx[0]), int(min_idx[1])
            actual_lat = lat_vals[lat_idx, lon_idx]
            actual_lon = lon_vals[lat_idx, lon_idx]
        else:
            # 1D coordinate arrays
            lat_idx = int(np.abs(lat_vals - target_lat).argmin())
            lon_idx = int(np.abs(lon_vals - target_lon).argmin())
            actual_lat = lat_vals[lat_idx]
            actual_lon = lon_vals[lon_idx]

        # Extract surface O3 time series using dimension names (z=0 is surface)
        if "z" in ds.dims:
            o3_ts = ds["O3"].isel(z=0, **{y_dim: lat_idx, x_dim: lon_idx}) * 1e9
        else:
            o3_ts = ds["O3"].isel(**{y_dim: lat_idx, x_dim: lon_idx}) * 1e9

        # Plot
        ax.plot(range(len(o3_ts)), o3_ts.values, "b-", linewidth=1.5)
        ax.set_ylabel("O3 (ppb)")
        ax.set_title(f"{city} ({actual_lat:.1f}N, {actual_lon:.1f}E)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 80)

    axes[1, 0].set_xlabel("Hour (UTC)")
    axes[1, 1].set_xlabel("Hour (UTC)")

    plt.tight_layout()

    # Save PNG and PDF
    outfile_png = OUTPUT_DIR / "cesm_o3_timeseries.png"
    outfile_pdf = OUTPUT_DIR / "cesm_o3_timeseries.pdf"
    plt.savefig(outfile_png, dpi=300, bbox_inches="tight")
    plt.savefig(outfile_pdf, bbox_inches="tight")
    print(f"Saved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")
    plt.close()


def main():
    """Main entry point."""
    # Test the reader
    ds = test_cesm_reader()

    if ds is not None:
        # Create plots
        print("\n" + "=" * 60)
        print("Creating diagnostic plots...")
        plot_surface_snapshot(ds)
        plot_time_series(ds)

    print("\n" + "=" * 60)
    print("Test complete.")


if __name__ == "__main__":
    main()
