#!/usr/bin/env python
"""Scatter Plot Example.

Demonstrates the scatter plotter for dataset vs dataset comparison
with regression line and statistics.

Data: Surface point datasets (O3) and satellite swath data (NO2)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, create_paired_swath_data, save_figure

from davinci_monet.plots import PlotConfig, ScatterPlotter, build_series


def main():
    """Generate scatter plot examples."""
    print("Creating scatter plots...")

    # Example 1: Surface O3
    print("  Surface O3...")
    paired_surface = create_paired_surface_data(n_sites=30, variables=["O3"])

    plotter = ScatterPlotter(PlotConfig(title="Scatter Plot: Surface O3"))
    fig = plotter.render(build_series(paired_surface, "x_o3", "y_o3"))
    save_figure(fig, "03a_scatter_surface")
    plt.close(fig)

    # Example 2: Satellite NO2 (larger dataset with density)
    print("  Satellite NO2...")
    paired_swath = create_paired_swath_data(n_scans=100, n_pixels=60, variables=["NO2"])

    # Flatten swath data for scatter
    x_flat = paired_swath["x_no2"].values.flatten()
    y_flat = paired_swath["y_no2"].values.flatten()

    import numpy as np
    import xarray as xr

    scatter_ds = xr.Dataset(
        {
            "x_no2": (["point"], x_flat),
            "y_no2": (["point"], y_flat),
        }
    )

    plotter = ScatterPlotter(PlotConfig(title="Scatter Plot: Satellite NO2 (Swath)"))
    fig = plotter.render(
        build_series(scatter_ds, "x_no2", "y_no2"),
        show_density=True,
    )
    save_figure(fig, "03b_scatter_satellite")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
