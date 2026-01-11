#!/usr/bin/env python
"""Site Time Series Plot Example.

Demonstrates the site_timeseries plotter for multi-panel plots showing
model vs observation time series at individual monitoring sites.

Data: Surface point observations (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_site_timeseries

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate site time series plot example."""
    print("Creating site time series plot...")

    # Create synthetic paired data with 4 sites
    paired = create_paired_surface_data(n_sites=4, variables=["O3"])

    # Create plot using davinci_monet.plots
    # Shows all 4 sites in a 2x2 grid
    fig = plot_site_timeseries(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Site Time Series: O3 at Individual Stations",
        ncols=2,
        min_points=10,
    )

    save_figure(fig, "11_site_timeseries")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
