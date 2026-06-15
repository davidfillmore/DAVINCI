#!/usr/bin/env python
"""Taylor Diagram Example.

Demonstrates the Taylor diagram plotter for performance summary showing
correlation, standard deviation ratio, and RMSE.

Data: Surface point datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_surface_data, save_figure

from davinci_monet.plots import PlotConfig, TaylorPlotter, build_series


def main():
    """Generate Taylor diagram example."""
    print("Creating Taylor diagram...")

    # Create synthetic paired data
    paired = create_paired_surface_data(n_sites=30, variables=["O3"])

    # Create plot using davinci_monet.plots
    # Taylor diagram for single variable comparison
    plotter = TaylorPlotter(PlotConfig(title="Taylor Diagram: O3 Performance"))
    fig = plotter.render(build_series(paired, "x_o3", "y_o3"))

    save_figure(fig, "04_taylor")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
