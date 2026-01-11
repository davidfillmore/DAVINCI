#!/usr/bin/env python
"""Spatial Overlay Map Example.

Demonstrates the spatial_overlay plotter for showing model field as
contours with observation points overlaid.

Data: Gridded (L3) observations with separate model field
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from davinci_monet.plots import plot_spatial_overlay
from davinci_monet.tests.synthetic.generators import Domain
from davinci_monet.tests.synthetic.models import create_model_dataset

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate spatial overlay map example."""
    print("Creating spatial overlay map...")

    # Create synthetic point observations
    paired_points = create_paired_surface_data(n_sites=40, variables=["O3"])
    paired_mean = paired_points.mean(dim="time")

    # Create separate gridded model field for contouring
    domain = Domain(lat_min=25, lat_max=50, lon_min=-125, lon_max=-65, n_lat=50, n_lon=100)
    model_grid = create_model_dataset(
        variables=["O3"],
        domain=domain,
        n_levels=0,
        seed=42,
    )
    model_field = model_grid["O3"].isel(time=0)

    # Create plot using davinci_monet.plots
    fig = plot_spatial_overlay(
        paired_mean,
        obs_var="obs_o3",
        model_var="model_o3",
        model_field=model_field,
        title="Spatial Overlay: O3 Model + Obs",
    )

    save_figure(fig, "07_spatial_overlay")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
