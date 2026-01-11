#!/usr/bin/env python
"""Spatial Bias Map Example.

Demonstrates the spatial_bias plotter for showing geographic distribution
of model bias at observation locations.

Data: Surface point observations (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_spatial_bias

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate spatial bias map example."""
    print("Creating spatial bias map...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=50, variables=["O3"])

    # Time-average for spatial map
    paired_mean = paired.mean(dim="time")

    # Create plot using davinci_monet.plots
    fig = plot_spatial_bias(
        paired_mean,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Spatial Bias: Surface O3",
    )

    save_figure(fig, "06_spatial_bias")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
