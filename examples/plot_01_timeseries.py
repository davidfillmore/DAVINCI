#!/usr/bin/env python
"""Time Series Plot Example.

Demonstrates the timeseries plotter for comparing model and observation
time series with uncertainty bands.

Data: Surface point observations (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_timeseries, PlotConfig

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate time series plot example."""
    print("Creating time series plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    fig = plot_timeseries(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Time Series: O3 Model vs Observations",
    )

    save_figure(fig, "01_timeseries")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
