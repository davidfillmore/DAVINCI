#!/usr/bin/env python
"""Satellite Swath Plot Example.

Demonstrates plotting satellite swath data (e.g., TROPOMI NO2)
using spatial bias and spatial distribution plotters.

Data: Satellite swath datasets (NO2 column density)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_spatial_bias, plot_spatial_distribution

from _helpers import create_paired_swath_data, save_figure


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

    geometry_flat = paired["geometry_no2"].values.flatten()
    dataset_flat = paired["dataset_no2"].values.flatten()
    lat_flat = paired["latitude"].values.flatten()
    lon_flat = paired["longitude"].values.flatten()

    # Remove NaN values
    valid = np.isfinite(geometry_flat) & np.isfinite(dataset_flat)
    geometry_flat = geometry_flat[valid]
    dataset_flat = dataset_flat[valid]
    lat_flat = lat_flat[valid]
    lon_flat = lon_flat[valid]

    # Create flattened dataset for spatial plotting
    paired_flat = xr.Dataset(
        {
            "geometry_no2": ("point", geometry_flat),
            "dataset_no2": ("point", dataset_flat),
        },
        coords={
            "point": np.arange(len(geometry_flat)),
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
        geometry_var="geometry_no2",
        dataset_var="dataset_no2",
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
        geometry_var="geometry_no2",
        dataset_var="dataset_no2",
        lat_var="latitude",
        lon_var="longitude",
        show_var="geometry",
        time_average=False,
        title="Spatial Distribution: TROPOMI L2 NO2 (Geometry)",
        marker_size=3,
    )
    save_figure(fig2, "14b_satellite_swath_geometry")
    plt.close(fig2)

    # 3. Spatial distribution (dataset)
    fig3 = plot_spatial_distribution(
        paired_flat,
        geometry_var="geometry_no2",
        dataset_var="dataset_no2",
        lat_var="latitude",
        lon_var="longitude",
        show_var="dataset",
        time_average=False,
        title="Spatial Distribution: TROPOMI L2 NO2 (Dataset)",
        marker_size=3,
    )
    save_figure(fig3, "14c_satellite_swath_dataset")
    plt.close(fig3)

    print("Done!")


if __name__ == "__main__":
    main()
