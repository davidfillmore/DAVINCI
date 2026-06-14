#!/usr/bin/env python
"""Flight Time Series Plot Example.

Demonstrates the flight_timeseries plotter for multi-panel plots showing
dataset vs dataset time series for individual aircraft flights.

Data: Aircraft track datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_track_data, save_figure

from davinci_monet.plots import plot_flight_timeseries


def main():
    """Generate flight time series plot example."""
    print("Creating flight time series plot...")

    # Create synthetic paired track data
    paired = create_paired_track_data(n_flights=6, variables=["O3"])

    # Create plot using davinci_monet.plots
    fig = plot_flight_timeseries(
        paired,
        x_var="x_o3",
        y_var="y_o3",
        title="Flight Time Series: O3 Along Aircraft Tracks",
        ncols=3,
        min_points=20,
    )

    save_figure(fig, "12_flight_timeseries")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
