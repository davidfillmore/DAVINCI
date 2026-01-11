#!/usr/bin/env python
"""Taylor Diagram Example.

Demonstrates the Taylor diagram plotter for performance summary showing
correlation, standard deviation ratio, and RMSE.

Data: Surface point observations (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_taylor

from _helpers import create_paired_surface_data, save_figure


def main():
    """Generate Taylor diagram example."""
    print("Creating Taylor diagram...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    # Taylor diagram for single variable comparison
    fig = plot_taylor(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="Taylor Diagram: O3 Performance",
    )

    save_figure(fig, "04_taylor")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
