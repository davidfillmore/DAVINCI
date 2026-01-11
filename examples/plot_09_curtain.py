#!/usr/bin/env python
"""Curtain Plot Example.

Demonstrates the curtain plotter for vertical cross-sections showing
model and observation profiles over time.

Data: Aircraft track observations with altitude (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_curtain

from _helpers import create_paired_track_data, save_figure


def main():
    """Generate curtain plot example."""
    print("Creating curtain plot...")

    # Create synthetic paired track data (has altitude coordinate)
    paired = create_paired_track_data(
        n_flights=1,  # Single flight for cleaner curtain
        points_per_flight=500,
        variables=["O3"],
    )

    # Create plot using davinci_monet.plots
    fig = plot_curtain(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Curtain Plot: Aircraft O3",
        alt_var="altitude",
    )

    save_figure(fig, "09_curtain")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
