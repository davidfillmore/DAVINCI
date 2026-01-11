#!/usr/bin/env python
"""Satellite Gridded (L3) Plot Example.

Demonstrates plotting satellite Level-3 gridded data
using pcolormesh for native grid visualization.

Data: Gridded satellite observations (NO2 column density)
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.plots.base import format_plot_title

from _helpers import create_paired_gridded_data, save_figure


def main():
    """Generate satellite gridded plot examples."""
    print("Creating satellite gridded (L3) plots...")

    # Create synthetic L3 gridded data (like TROPOMI L3 NO2)
    # bias_range creates diagonal pattern from -30% (SW) to +30% (NE)
    paired = create_paired_gridded_data(
        variables=["NO2"],
        bias_range=0.3,
    )

    # Select time mean for plotting
    obs_mean = paired["obs_no2"].mean(dim="time")
    model_mean = paired["model_no2"].mean(dim="time")
    bias_mean = model_mean - obs_mean

    lats = paired["lat"].values
    lons = paired["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Common map setup
    def setup_map(ax):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.add_feature(cfeature.STATES, linewidth=0.2, linestyle=":")
        ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    # 1. Observations (L3 grid)
    fig1, ax1 = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    setup_map(ax1)

    vmin = float(np.nanpercentile(obs_mean.values, 2))
    vmax = float(np.nanpercentile(obs_mean.values, 98))

    pcm1 = ax1.pcolormesh(
        lon_grid, lat_grid, obs_mean.values,
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    cbar1 = fig1.colorbar(pcm1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label(format_plot_title("NO2 Column (mol/m²)"), fontsize=10)
    ax1.set_title(format_plot_title("Spatial Distribution: TROPOMI L3 NO2 (Obs)"), fontsize=12)
    plt.tight_layout()

    save_figure(fig1, "15a_satellite_gridded_obs")
    plt.close(fig1)

    # 2. Model
    fig2, ax2 = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    setup_map(ax2)

    pcm2 = ax2.pcolormesh(
        lon_grid, lat_grid, model_mean.values,
        cmap="YlOrRd",
        vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    cbar2 = fig2.colorbar(pcm2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label(format_plot_title("NO2 Column (mol/m²)"), fontsize=10)
    ax2.set_title(format_plot_title("Spatial Distribution: TROPOMI L3 NO2 (Model)"), fontsize=12)
    plt.tight_layout()

    save_figure(fig2, "15b_satellite_gridded_model")
    plt.close(fig2)

    # 3. Bias (Model - Obs) with diverging colormap
    fig3, ax3 = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    setup_map(ax3)

    # Symmetric limits for bias
    bias_abs_max = float(np.nanmax(np.abs(bias_mean.values)))
    norm = TwoSlopeNorm(vmin=-bias_abs_max, vcenter=0, vmax=bias_abs_max)

    pcm3 = ax3.pcolormesh(
        lon_grid, lat_grid, bias_mean.values,
        cmap="RdBu_r",
        norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    cbar3 = fig3.colorbar(pcm3, ax=ax3, shrink=0.8, pad=0.02)
    cbar3.set_label(format_plot_title("Bias (Model - Obs) (mol/m²)"), fontsize=10)
    ax3.set_title(format_plot_title("Spatial Bias: TROPOMI L3 NO2"), fontsize=12)
    plt.tight_layout()

    save_figure(fig3, "15c_satellite_gridded_bias")
    plt.close(fig3)

    print("Done!")


if __name__ == "__main__":
    main()
