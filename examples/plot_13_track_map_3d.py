#!/usr/bin/env python
"""3D Track Map Plot Example.

Demonstrates the track_map_3d plotter for 3D visualization of aircraft
flight trajectories colored by observation, model, or bias values.

Data: Aircraft track observations (O3)
"""

import matplotlib.pyplot as plt

from davinci_monet.plots import plot_track_map_3d

from _helpers import create_paired_track_data, save_figure


def main():
    """Generate 3D track map plot examples."""
    print("Creating 3D track map plots...")

    # Create synthetic paired track data
    paired = create_paired_track_data(n_flights=6, variables=["O3"])

    # Example 1: Show bias (default)
    print("  Bias view...")
    fig = plot_track_map_3d(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="3D Track Map: Aircraft O3 Bias",
        show_var="bias",
    )
    save_figure(fig, "13a_track_map_3d_bias")
    plt.close(fig)

    # Example 2: Show observations
    print("  Observations view...")
    fig = plot_track_map_3d(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="3D Track Map: Aircraft O3 (Obs)",
        show_var="obs",
    )
    save_figure(fig, "13b_track_map_3d_obs")
    plt.close(fig)

    # Example 3: Show model
    print("  Model view...")
    fig = plot_track_map_3d(
        paired,
        obs_var="obs_o3",
        model_var="model_o3",
        title="3D Track Map: Aircraft O3 (Model)",
        show_var="model",
    )
    save_figure(fig, "13c_track_map_3d_model")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
