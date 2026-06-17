#!/usr/bin/env python
"""
Plot CESM dataset data using DAVINCI-MONET spatial plotters.

Since the built-in plotters expect paired data, we create a dataset
with dataset data as both 'geometry' and 'dataset' variables to use show_var="dataset".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from davinci_monet.datasets.cesm import CESMFVReader
from davinci_monet.plots import (
    MapConfig,
    PlotConfig,
    SpatialDistributionPlotter,
)

# Paths
DATA_DIR = Path.home() / "Data" / "ASIA-AQ"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_PATTERN = "f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.2024-02-01-*.nc"


def prepare_dataset_as_paired(ds: xr.Dataset, var: str, scale: float = 1.0) -> xr.Dataset:
    """
    Create a 'paired' dataset from dataset data for use with plotters.

    The plotters expect geometry_var and dataset_var, so we duplicate the dataset
    variable and add proper coordinate names.
    """
    # Extract surface data (z=0 for CESM)
    if "z" in ds[var].dims:
        data = ds[var].isel(z=0) * scale
    else:
        data = ds[var] * scale

    # Time average
    if "time" in data.dims:
        data = data.mean(dim="time")

    # Get coordinates
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"

    # Create paired-like dataset
    paired = xr.Dataset(
        {
            f"geometry_{var.lower()}": data,
            f"dataset_{var.lower()}": data,
        }
    )

    # Add coordinates
    paired = paired.assign_coords(
        {
            "latitude": ds[lat_coord],
            "longitude": ds[lon_coord],
        }
    )

    return paired


def plot_with_davinci_plotter():
    """Use DAVINCI-MONET spatial plotter for dataset visualization."""
    print("Loading CESM data...")
    files = sorted(DATA_DIR.glob(FILE_PATTERN))[:12]  # First 12 hours
    ds = CESMFVReader().open(files, variables=["O3", "NO2", "CO", "PM25"])

    # Configure map
    map_config = MapConfig(
        extent=(90, 140, 0, 45),  # lon_min, lon_max, lat_min, lat_max
        show_coastlines=True,
        show_countries=True,
        show_gridlines=True,
    )

    # Variables to plot with their settings
    variables = [
        ("O3", 1e9, "ppb", "RdYlBu_r", 0, 80),
        ("NO2", 1e9, "ppb", "YlOrRd", 0, 20),
        ("CO", 1e9, "ppb", "YlOrBr", 50, 400),
        ("PM25", 1e9, "μg/m³", "RdPu", 0, 100),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "CESM/CAM ASIA-AQ Surface Fields (DAVINCI-MONET Plotter)\n2024-02-01 00-12 UTC Average",
        fontsize=14,
    )

    for ax, (var, scale, units, cmap, vmin, vmax) in zip(axes.flat, variables):
        print(f"  Plotting {var}...")

        # Prepare paired-like dataset
        paired = prepare_dataset_as_paired(ds, var, scale=scale)

        # Configure plot
        plot_config = PlotConfig(
            title=f"{var} ({units})",
            vmin=vmin,
            vmax=vmax,
        )

        # Create plotter and plot
        plotter = SpatialDistributionPlotter(config=plot_config, map_config=map_config)

        # We need to create a GeoAxes for cartopy
        import cartopy.crs as ccrs

        ax_geo = fig.add_subplot(2, 2, list(axes.flat).index(ax) + 1, projection=ccrs.PlateCarree())
        ax.remove()  # Remove the original non-geo axis

        # Plot using the plotter
        plotter.plot(
            paired,
            geometry_var=f"geometry_{var.lower()}",
            dataset_var=f"dataset_{var.lower()}",
            ax=ax_geo,
            show_var="dataset",
            cmap=cmap,
            plot_type="pcolormesh",
            time_average=False,  # Already averaged
        )

    plt.tight_layout()

    # Save
    outfile_png = OUTPUT_DIR / "cesm_spatial_davinci.png"
    outfile_pdf = OUTPUT_DIR / "cesm_spatial_davinci.pdf"
    plt.savefig(outfile_png, dpi=300, bbox_inches="tight")
    plt.savefig(outfile_pdf, bbox_inches="tight")
    print(f"\nSaved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")
    plt.close()


def plot_single_variable_example():
    """Demonstrate single-panel plot with DAVINCI-MONET."""
    print("\nCreating single-variable example...")

    files = sorted(DATA_DIR.glob(FILE_PATTERN))[:12]
    ds = CESMFVReader().open(files, variables=["O3"])

    # Prepare data
    paired = prepare_dataset_as_paired(ds, "O3", scale=1e9)

    # Configure
    plot_config = PlotConfig(
        title="Surface Ozone - CESM/CAM ASIA-AQ",
        vmin=0,
        vmax=80,
    )

    map_config = MapConfig(
        extent=(90, 140, 0, 45),
        show_coastlines=True,
        show_countries=True,
        show_gridlines=True,
    )

    # Create plotter
    plotter = SpatialDistributionPlotter(config=plot_config, map_config=map_config)

    # Plot
    fig = plotter.plot(
        paired,
        geometry_var="geometry_o3",
        dataset_var="dataset_o3",
        show_var="dataset",
        cmap="RdYlBu_r",
        plot_type="pcolormesh",
    )

    # Save
    outfile_png = OUTPUT_DIR / "cesm_o3_spatial_davinci.png"
    outfile_pdf = OUTPUT_DIR / "cesm_o3_spatial_davinci.pdf"
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    fig.savefig(outfile_pdf, bbox_inches="tight")
    print(f"Saved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    print("DAVINCI-MONET Spatial Plotting Demo")
    print("=" * 50)

    try:
        plot_single_variable_example()
        print("\n" + "=" * 50)
        print("Done!")
    except ImportError as e:
        print(f"Note: Cartopy required for spatial plots: {e}")
        print("Install with: conda install -c conda-forge cartopy")
