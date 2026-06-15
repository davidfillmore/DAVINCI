#!/usr/bin/env python
"""Satellite Swath Plot Example.

Demonstrates plotting satellite swath data (e.g., TROPOMI NO2)
using the spatial bias plotter and the single-source spatial field plotter.

Data: Satellite swath datasets (NO2 column density)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_swath_data, save_figure

from davinci_monet.plots import PlotConfig, SpatialBiasPlotter, SpatialPlotter, build_series


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
    plotter_bias = SpatialBiasPlotter(PlotConfig(title="Spatial Bias: TROPOMI L2 NO2"))
    fig1 = plotter_bias.render(
        build_series(paired_flat, "x_no2", "y_no2"),
        lat_var="latitude",
        lon_var="longitude",
        time_average=False,  # Already 1D
        marker_size=3,
    )
    save_figure(fig1, "14a_satellite_swath_bias")
    plt.close(fig1)

    # Build single-source point datasets for the single-source spatial field
    # plotter (one field per source).
    def _single_source(var: str, label: str) -> xr.Dataset:
        out = xr.Dataset(
            {"NO2": ("point", paired_flat[var].values)},
            coords={
                "point": paired_flat["point"].values,
                "latitude": ("point", paired_flat["latitude"].values),
                "longitude": ("point", paired_flat["longitude"].values),
            },
            attrs={"geometry": "point", "source_label": label},
        )
        out["NO2"].attrs["units"] = "mol/m2"
        return out

    # 2. Spatial field (geometry / x source)
    plotter_x = SpatialPlotter(PlotConfig(title="Spatial Field: TROPOMI L2 NO2 (Geometry)"))
    fig2 = plotter_x.render(
        build_series(_single_source("x_no2", "geometry"), "NO2"),
        time_average=False,
        marker_size=3,
    )
    save_figure(fig2, "14b_satellite_swath_geometry")
    plt.close(fig2)

    # 3. Spatial field (dataset / y source)
    plotter_y = SpatialPlotter(PlotConfig(title="Spatial Field: TROPOMI L2 NO2 (Dataset)"))
    fig3 = plotter_y.render(
        build_series(_single_source("y_no2", "dataset"), "NO2"),
        time_average=False,
        marker_size=3,
    )
    save_figure(fig3, "14c_satellite_swath_dataset")
    plt.close(fig3)

    print("Done!")


if __name__ == "__main__":
    main()
