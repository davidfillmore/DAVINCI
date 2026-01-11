#!/usr/bin/env python
"""Scatter Plot Example.

Demonstrates the scatter plotter for model vs observation comparison
with regression line and statistics.

Data: Surface point observations (O3) and satellite swath data (NO2)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_scatter

from _helpers import create_paired_surface_data, create_paired_swath_data, save_figure


def main():
    """Generate scatter plot examples."""
    print("Creating scatter plots...")

    # Example 1: Surface O3
    print("  Surface O3...")
    paired_surface = create_paired_surface_data(n_sites=30, variables=["O3"])

    fig = plot_scatter(
        paired_surface,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Scatter Plot: Surface O3",
    )
    save_figure(fig, "03a_scatter_surface")
    plt.close(fig)

    # Example 2: Satellite NO2 (larger dataset with density)
    print("  Satellite NO2...")
    paired_swath = create_paired_swath_data(n_scans=100, n_pixels=60, variables=["NO2"])

    # Flatten swath data for scatter
    obs_flat = paired_swath["obs_no2"].values.flatten()
    model_flat = paired_swath["model_no2"].values.flatten()

    import xarray as xr
    import numpy as np

    scatter_ds = xr.Dataset({
        "obs_no2": (["point"], obs_flat),
        "model_no2": (["point"], model_flat),
    })

    fig = plot_scatter(
        scatter_ds,
        obs_var="obs_no2",
        model_var="model_no2",
        title="Scatter Plot: Satellite NO2 (Swath)",
        show_density=True,
    )
    save_figure(fig, "03b_scatter_satellite")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
