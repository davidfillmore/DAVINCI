#!/usr/bin/env python
"""Scorecard Plot Example.

Demonstrates the scorecard plotter for multi-metric performance heatmaps.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import plot_scorecard


def main():
    """Generate scorecard plot example."""
    print("Creating scorecard plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    fig = plot_scorecard(
        paired,
        x_var="x_o3",
        y_var="y_o3",
        title="Scorecard: O3 Performance Metrics",
    )

    save_figure(fig, "10_scorecard")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
