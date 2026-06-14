#!/usr/bin/env python
"""Spatial Distribution Map Example.

Demonstrates the spatial_distribution plotter for showing geographic
distribution of dataset values.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_spatial_distribution

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate spatial distribution map example."""
    print("Creating spatial distribution map...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=50, variables=["O3"])

    # Time-average for spatial map
    paired_mean = paired.mean(dim="time")

    # Create plot using davinci_monet.plots
    fig = plot_spatial_distribution(
        paired_mean,
        geometry_var="geometry_o3",
        dataset_var="dataset_o3",
        title="Spatial Distribution: Surface O3",
    )

    save_figure(fig, "08_spatial_distribution")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
