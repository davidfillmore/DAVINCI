#!/usr/bin/env python
"""3D Track Map Plot Example.

Demonstrates the track_map_3d plotter for 3D visualization of aircraft
flight trajectories colored by dataset, dataset, or bias values.

Data: Aircraft track datasets (O3)
"""

import matplotlib.pyplot as plt
from _helpers import create_paired_track_data, save_figure

from davinci_monet.plots import PlotConfig, TrackMap3DPlotter, build_series


def main():
    """Generate 3D track map plot examples."""
    print("Creating 3D track map plots...")

    # Create synthetic paired track data
    paired = create_paired_track_data(n_flights=6, variables=["O3"])

    # Example 1: Show bias (default)
    print("  Bias view...")
    plotter = TrackMap3DPlotter(PlotConfig(title="3D Track Map: Aircraft O3 Bias"))
    fig = plotter.render(
        build_series(paired, "x_o3", "y_o3"),
        show_var="bias",
    )
    save_figure(fig, "13a_track_map_3d_bias")
    plt.close(fig)

    # Example 2: Show datasets
    print("  Datasets view...")
    plotter = TrackMap3DPlotter(PlotConfig(title="3D Track Map: Aircraft O3 (Geometry)"))
    fig = plotter.render(
        build_series(paired, "x_o3", "y_o3"),
        show_var="x",
    )
    save_figure(fig, "13b_track_map_3d_geometry")
    plt.close(fig)

    # Example 3: Show dataset
    print("  Dataset view...")
    plotter = TrackMap3DPlotter(PlotConfig(title="3D Track Map: Aircraft O3 (Dataset)"))
    fig = plotter.render(
        build_series(paired, "x_o3", "y_o3"),
        show_var="y",
    )
    save_figure(fig, "13c_track_map_3d_dataset")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
