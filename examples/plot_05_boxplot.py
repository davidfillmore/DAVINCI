#!/usr/bin/env python
"""Box Plot Example.

Demonstrates the boxplot plotter for comparing distributions between
model and observations.

Data: Surface point observations (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_boxplot

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate box plot example."""
    print("Creating box plot...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    fig = plot_boxplot(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Box Plot: O3 Distribution",
    )

    save_figure(fig, "05_boxplot")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
