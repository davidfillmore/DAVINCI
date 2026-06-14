#!/usr/bin/env python
"""Diurnal Cycle Plot Example.

Demonstrates the diurnal plotter for comparing mean diurnal patterns
between dataset and datasets.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import plot_diurnal


def main():
    """Generate diurnal cycle plot example."""
    print("Creating diurnal cycle plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    fig = plot_diurnal(
        paired,
        x_var="x_o3",
        y_var="y_o3",
        title="Diurnal Cycle: O3 Dataset vs Datasets",
    )

    save_figure(fig, "02_diurnal")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
