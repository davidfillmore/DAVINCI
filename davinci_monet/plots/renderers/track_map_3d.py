"""3D track map renderer for DAVINCI-MONET.

This module provides 3D visualization of aircraft flight tracks,
showing longitude, latitude, and altitude with color-coded values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    calculate_symmetric_limits,
    format_label_with_units,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("track_map_3d")
class TrackMap3DPlotter(BasePlotter):
    """Plotter for 3D flight track visualization.

    Creates 3D plots showing aircraft trajectory with:
    - X-axis: Longitude
    - Y-axis: Latitude
    - Z-axis: Altitude
    - Color: Variable value (obs, model, or bias)

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TrackMap3DPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_O3",
    ...     model_var="model_O3",
    ...     show_var="bias",
    ... )
    """

    name: str = "track_map_3d"

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        alt_var: str = "altitude",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        show_var: Literal["obs", "model", "bias"] = "bias",
        cmap: str | None = None,
        marker_size: float = 20,
        alpha: float = 0.7,
        elev: float = 25,
        azim: float = -60,
        show_projection: bool = True,
        projection_alpha: float = 0.3,
        alt_scale: float = 0.001,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a 3D track map.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Existing axes (ignored, creates new 3D axes).
        alt_var
            Name of altitude coordinate.
        lat_var
            Name of latitude coordinate.
        lon_var
            Name of longitude coordinate.
        show_var
            Which variable to show: 'obs', 'model', or 'bias'.
        cmap
            Colormap name. Default depends on show_var.
        marker_size
            Size of scatter markers.
        alpha
            Transparency of markers.
        elev
            Elevation angle for 3D view.
        azim
            Azimuth angle for 3D view.
        show_projection
            If True, show 2D projection on the bottom (z=0) plane.
        projection_alpha
            Transparency of projection markers.
        alt_scale
            Scale factor for altitude (e.g., 0.001 to convert m to km).
        **kwargs
            Additional options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Get coordinates
        if lat_var in paired_data.coords:
            lats = paired_data[lat_var].values
        elif lat_var in paired_data.data_vars:
            lats = paired_data[lat_var].values
        else:
            raise ValueError(f"Latitude variable '{lat_var}' not found")

        if lon_var in paired_data.coords:
            lons = paired_data[lon_var].values
        elif lon_var in paired_data.data_vars:
            lons = paired_data[lon_var].values
        else:
            raise ValueError(f"Longitude variable '{lon_var}' not found")

        if alt_var in paired_data.coords:
            alts = paired_data[alt_var].values * alt_scale
        elif alt_var in paired_data.data_vars:
            alts = paired_data[alt_var].values * alt_scale
        else:
            raise ValueError(f"Altitude variable '{alt_var}' not found")

        # Get data values
        obs_vals = paired_data[obs_var].values
        model_vals = paired_data[model_var].values

        # Calculate what to show
        if show_var == "obs":
            values = obs_vals
            default_cmap = "viridis"
            label = get_variable_label(paired_data, obs_var, include_prefix=False)
        elif show_var == "model":
            values = model_vals
            default_cmap = "viridis"
            label = get_variable_label(paired_data, model_var, include_prefix=False)
        else:  # bias
            values = model_vals - obs_vals
            default_cmap = "RdBu_r"
            # Get a nice display name for bias label
            var_label = get_variable_label(paired_data, obs_var, include_prefix=False)
            label = f"{var_label} Bias"

        cmap = cmap or default_cmap

        # Filter valid data
        valid = ~np.isnan(values) & ~np.isnan(lats) & ~np.isnan(lons) & ~np.isnan(alts)
        lats = lats[valid]
        lons = lons[valid]
        alts = alts[valid]
        values = values[valid]

        if len(values) == 0:
            raise ValueError("No valid data points for 3D track plot")

        # Create figure with 3D axes
        fig = plt.figure(figsize=self.config.figure.figsize)
        ax3d = fig.add_subplot(111, projection="3d")

        # Set color limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(values)
        else:
            vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(values)
            vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(values)

        # Plot 3D scatter
        scatter = ax3d.scatter(
            lons, lats, alts,
            c=values,
            cmap=cmap,
            s=marker_size,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            depthshade=True,
        )

        # Add 2D projection on bottom plane
        if show_projection:
            ax3d.scatter(
                lons, lats, np.zeros_like(alts),
                c=values,
                cmap=cmap,
                s=marker_size * 0.3,
                alpha=projection_alpha,
                vmin=vmin,
                vmax=vmax,
            )

        # Set view angle
        ax3d.view_init(elev=elev, azim=azim)

        # Labels
        ax3d.set_xlabel("Longitude (°E)", fontsize=10)
        ax3d.set_ylabel("Latitude (°N)", fontsize=10)
        ax3d.set_zlabel("Altitude (km)", fontsize=10)

        # Colorbar
        units = get_variable_units(paired_data, obs_var)
        cbar_label = format_label_with_units(label, units)
        cbar_label = format_plot_title(cbar_label)  # Apply subscript formatting
        cbar = fig.colorbar(scatter, ax=ax3d, shrink=0.6, pad=0.1)
        cbar.set_label(cbar_label, fontsize=10)

        # Title
        if self.config.title:
            ax3d.set_title(self.config.title, fontsize=12)

        plt.tight_layout()
        return fig


def plot_track_map_3d(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    title: str | None = None,
    show_var: Literal["obs", "model", "bias"] = "bias",
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for 3D track map plots.

    Parameters
    ----------
    paired_data
        Paired dataset with model and observation variables.
    obs_var
        Name of observation variable.
    model_var
        Name of model variable.
    title
        Plot title.
    show_var
        Which variable to show: 'obs', 'model', or 'bias'.
    **kwargs
        Additional options passed to plotter.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    config = PlotConfig(title=title)
    plotter = TrackMap3DPlotter(config)
    return plotter.plot(
        paired_data,
        obs_var,
        model_var,
        show_var=show_var,
        **kwargs,
    )
