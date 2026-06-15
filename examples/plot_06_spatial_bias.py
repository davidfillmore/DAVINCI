#!/usr/bin/env python
"""Spatial Bias Map Example.

Demonstrates the spatial_bias plotter for showing geographic distribution
of dataset bias at dataset locations.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import PlotConfig, SpatialBiasPlotter, build_series


def main():
    """Generate spatial bias map example."""
    print("Creating spatial bias map...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=50, variables=["O3"])

    # Time-average for spatial map
    paired_mean = paired.mean(dim="time")

    # Create plot using davinci_monet.plots
    plotter = SpatialBiasPlotter(PlotConfig(title="Spatial Bias: Surface O3"))
    fig = plotter.render(build_series(paired_mean, "x_o3", "y_o3"))

    save_figure(fig, "06_spatial_bias")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
