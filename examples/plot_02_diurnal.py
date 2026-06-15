#!/usr/bin/env python
"""Diurnal Cycle Plot Example.

Demonstrates the diurnal plotter for comparing mean diurnal patterns
between dataset and datasets.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import DiurnalPlotter, PlotConfig, build_series


def main():
    """Generate diurnal cycle plot example."""
    print("Creating diurnal cycle plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    plotter = DiurnalPlotter(PlotConfig(title="Diurnal Cycle: O3 Y vs X"))
    fig = plotter.render(build_series(paired, "x_o3", "y_o3"))

    save_figure(fig, "02_diurnal")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
