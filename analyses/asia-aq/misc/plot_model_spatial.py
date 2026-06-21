#!/usr/bin/env python
"""
Plot CESM model data using DAVINCI spatial plotters.

Uses the single-source ``SpatialPlotter`` (the ``type: spatial`` renderer): it
renders one source's field on a map, dispatching on the source's geometry shape
(grid -> pcolormesh). The renderer slices a 3-D field to the surface
automatically using the CESM vertical convention and time-averages by default,
so no manual level/time reduction is needed here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

from davinci_monet.datasets.cesm import CESMFVReader
from davinci_monet.plots import MapConfig, PlotConfig, SpatialPlotter, build_series

# Paths
DATA_DIR = Path.home() / "Data" / "ASIA-AQ"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FILE_PATTERN = "f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.2024-02-01-*.nc"

# ASIA-AQ map extent (lon_min, lon_max, lat_min, lat_max)
ASIA_AQ_EXTENT = (90, 140, 0, 45)


def model_field(ds: xr.Dataset, var: str, scale: float = 1.0, units: str = "") -> xr.Dataset:
    """Build a single-source GRID dataset for one CESM variable.

    Scales the field (e.g. mol/mol -> ppb) and tags it with the ``grid``
    geometry and a source label so ``SpatialPlotter`` renders it as a filled
    pcolormesh field.
    """
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"

    field = ds[var] * scale
    field.attrs["units"] = units

    out = xr.Dataset({var: field})
    if lat != "latitude" or lon != "longitude":
        out = out.rename({lat: "latitude", lon: "longitude"})
    out.attrs["geometry"] = "grid"
    out.attrs["source_label"] = "CESM"
    return out


def plot_panel():
    """Render a 2x2 panel of CESM surface fields with the spatial plotter."""
    import cartopy.crs as ccrs

    print("Loading CESM data...")
    files = sorted(DATA_DIR.glob(FILE_PATTERN))[:12]  # First 12 hours
    ds = CESMFVReader().open(files, variables=["O3", "NO2", "CO", "PM25"])

    map_config = MapConfig(
        extent=ASIA_AQ_EXTENT,
        show_coastlines=True,
        show_countries=True,
        show_gridlines=True,
    )

    # (variable, scale, units, cmap, vmin, vmax)
    variables = [
        ("O3", 1e9, "ppb", "RdYlBu_r", 0, 80),
        ("NO2", 1e9, "ppb", "YlOrRd", 0, 20),
        ("CO", 1e9, "ppb", "YlOrBr", 50, 400),
        ("PM25", 1e9, "µg/m³", "RdPu", 0, 100),
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(
        "CESM/CAM ASIA-AQ Surface Fields (DAVINCI Plotter)\n2024-02-01 00-12 UTC Average",
        fontsize=14,
    )

    for ax, (var, scale, units, cmap, vmin, vmax) in zip(axes.flat, variables):
        print(f"  Plotting {var}...")
        single = model_field(ds, var, scale=scale, units=units)
        plotter = SpatialPlotter(
            config=PlotConfig(title=f"{var} ({units})"), map_config=map_config
        )
        # SpatialPlotter slices the surface level and time-averages by default.
        plotter.render(build_series(single, var), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.tight_layout()

    outfile_png = OUTPUT_DIR / "cesm_spatial_davinci.png"
    outfile_pdf = OUTPUT_DIR / "cesm_spatial_davinci.pdf"
    plt.savefig(outfile_png, dpi=300, bbox_inches="tight")
    plt.savefig(outfile_pdf, bbox_inches="tight")
    print(f"\nSaved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")
    plt.close()


def plot_single_variable_example():
    """Render a single-panel surface ozone map with the spatial plotter."""
    print("\nCreating single-variable example...")

    files = sorted(DATA_DIR.glob(FILE_PATTERN))[:12]
    ds = CESMFVReader().open(files, variables=["O3"])

    single = model_field(ds, "O3", scale=1e9, units="ppb")

    map_config = MapConfig(
        extent=ASIA_AQ_EXTENT,
        show_coastlines=True,
        show_countries=True,
        show_gridlines=True,
    )
    plotter = SpatialPlotter(
        config=PlotConfig(title="Surface Ozone - CESM/CAM ASIA-AQ"),
        map_config=map_config,
    )
    fig = plotter.render(build_series(single, "O3"), cmap="RdYlBu_r", vmin=0, vmax=80)

    outfile_png = OUTPUT_DIR / "cesm_o3_spatial_davinci.png"
    outfile_pdf = OUTPUT_DIR / "cesm_o3_spatial_davinci.pdf"
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    fig.savefig(outfile_pdf, bbox_inches="tight")
    print(f"Saved: {outfile_png}")
    print(f"Saved: {outfile_pdf}")
    plt.close(fig)


if __name__ == "__main__":
    print("DAVINCI Spatial Plotting Demo")
    print("=" * 50)

    try:
        plot_single_variable_example()
        print("\n" + "=" * 50)
        print("Done!")
    except ImportError as e:
        print(f"Note: Cartopy required for spatial plots: {e}")
        print("Install with: conda install -c conda-forge cartopy")
