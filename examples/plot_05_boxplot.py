#!/usr/bin/env python
"""Box Plot Example.

Demonstrates the boxplot plotter for comparing distributions between
dataset and datasets.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import BoxPlotter, PlotConfig, build_series


def main():
    """Generate box plot example."""
    print("Creating box plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    plotter = BoxPlotter(PlotConfig(title="Box Plot: O3 Distribution"))
    fig = plotter.render(build_series(paired, "x_o3", "y_o3"))

    save_figure(fig, "05_boxplot")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
