#!/usr/bin/env python
"""Time Series Plot Example.

Demonstrates the timeseries plotter for comparing dataset and dataset
time series with uncertainty bands.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import PlotConfig, TimeSeriesPlotter, build_series


def main():
    """Generate time series plot example."""
    print("Creating time series plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    plotter = TimeSeriesPlotter(PlotConfig(title="Time Series: O3 Y vs X"))
    fig = plotter.render(build_series(paired, "x_o3", "y_o3"))

    save_figure(fig, "01_timeseries")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
