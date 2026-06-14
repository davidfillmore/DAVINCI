#!/usr/bin/env python
"""Satellite Swath Plot Example.

Demonstrates plotting satellite swath data (e.g., TROPOMI NO2)
using spatial bias and spatial distribution plotters.

Data: Satellite swath datasets (NO2 column density)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_swath_data, save_figure

from davinci_monet.plots import plot_spatial_bias, plot_spatial_distribution


def main():
    """Generate satellite swath plot examples."""
    print("Creating satellite swath plots...")

    # Create synthetic TROPOMI-like swath data
    # bias_range creates east-west gradient from -40% to +40%
    paired = create_paired_swath_data(
        n_scans=100,
        n_pixels=60,
        variables=["NO2"],
        bias_range=0.4,
    )

    # Flatten the swath data for spatial plotting
    # The swath has dims (scanline, pixel) with 2D lat/lon coordinates
    import numpy as np
    import xarray as xr

    x_flat = paired["x_no2"].values.flatten()
    y_flat = paired["y_no2"].values.flatten()
    lat_flat = paired["latitude"].values.flatten()
    lon_flat = paired["longitude"].values.flatten()

    # Remove NaN values
    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    lat_flat = lat_flat[valid]
    lon_flat = lon_flat[valid]

    # Create flattened dataset for spatial plotting
    paired_flat = xr.Dataset(
        {
            "x_no2": ("point", x_flat),
            "y_no2": ("point", y_flat),
        },
        coords={
            "point": np.arange(len(x_flat)),
            "latitude": ("point", lat_flat),
            "longitude": ("point", lon_flat),
        },
        attrs={
            "title": "TROPOMI NO2 Column",
            "units": "mol/m2",
        },
    )

    # 1. Spatial bias plot (colorbar indicates "Dataset - Geometry")
    fig1 = plot_spatial_bias(
        paired_flat,
        x_var="x_no2",
        y_var="y_no2",
        lat_var="latitude",
        lon_var="longitude",
        time_average=False,  # Already 1D
        title="Spatial Bias: TROPOMI L2 NO2",
        marker_size=3,
    )
    save_figure(fig1, "14a_satellite_swath_bias")
    plt.close(fig1)

    # 2. Spatial distribution (datasets)
    fig2 = plot_spatial_distribution(
        paired_flat,
        x_var="x_no2",
        y_var="y_no2",
        lat_var="latitude",
        lon_var="longitude",
        show_var="x",
        time_average=False,
        title="Spatial Distribution: TROPOMI L2 NO2 (Geometry)",
        marker_size=3,
    )
    save_figure(fig2, "14b_satellite_swath_geometry")
    plt.close(fig2)

    # 3. Spatial distribution (dataset)
    fig3 = plot_spatial_distribution(
        paired_flat,
        x_var="x_no2",
        y_var="y_no2",
        lat_var="latitude",
        lon_var="longitude",
        show_var="y",
        time_average=False,
        title="Spatial Distribution: TROPOMI L2 NO2 (Dataset)",
        marker_size=3,
    )
    save_figure(fig3, "14c_satellite_swath_dataset")
    plt.close(fig3)

    print("Done!")


if __name__ == "__main__":
    main()
