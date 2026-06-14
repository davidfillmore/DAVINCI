#!/usr/bin/env python
"""Spatial Overlay Map Example.

Demonstrates the spatial_overlay plotter for showing dataset field as
contours with dataset points overlaid.

Data: Gridded (L3) datasets with separate dataset field
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import plot_spatial_overlay
from davinci_monet.tests.synthetic.datasets import create_dataset_dataset
from davinci_monet.tests.synthetic.generators import Domain


def main():
    """Generate spatial overlay map example."""
    print("Creating spatial overlay map...")

    # Create synthetic point datasets
    paired_points = create_paired_surface_data(n_sites=40, variables=["O3"])
    paired_mean = paired_points.mean(dim="time")

    # Create separate gridded dataset field for contouring
    domain = Domain(lat_min=25, lat_max=50, lon_min=-125, lon_max=-65, n_lat=50, n_lon=100)
    dataset_grid = create_dataset_dataset(
        variables=["O3"],
        domain=domain,
        n_levels=0,
        seed=42,
    )
    y_field = dataset_grid["O3"].isel(time=0)

    # Create plot using davinci_monet.plots
    fig = plot_spatial_overlay(
        paired_mean,
        x_var="x_o3",
        y_var="y_o3",
        y_field=y_field,
        title="Spatial Overlay: O3 Dataset + Geometry",
    )

    save_figure(fig, "07_spatial_overlay")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
