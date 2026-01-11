"""Spatial distribution plot renderer for DAVINCI-MONET.

This module provides spatial distribution plotting functionality for
displaying observation or model values on a map without comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.base import (
    PlotConfig,
    calculate_data_limits,
    format_label_with_units,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("spatial_distribution")
class SpatialDistributionPlotter(BaseSpatialPlotter):
    """Plotter for spatial distribution maps.

    Creates maps showing the spatial distribution of values
    using scatter points or gridded pcolormesh.

    Parameters
    ----------
    config
        Plot configuration.
    map_config
        Map-specific configuration.

    Examples
    --------
    >>> plotter = SpatialDistributionPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     show_var="obs",
    ... )
    """

    name: str = "spatial_distribution"

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        show_var: Literal["obs", "model", "both"] = "obs",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        time_average: bool = True,
        cmap: str = "viridis",
        marker_size: float | None = None,
        plot_type: Literal["scatter", "pcolormesh"] = "scatter",
        alpha: float | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial distribution plot.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Optional GeoAxes to plot on.
        show_var
            Which variable to show ('obs', 'model', or 'both').
        lat_var
            Name of latitude coordinate/variable.
        lon_var
            Name of longitude coordinate/variable.
        time_average
            If True, average over time dimension.
        cmap
            Colormap name.
        marker_size
            Override marker size.
        plot_type
            Type of plot ('scatter' or 'pcolormesh').
        alpha
            Override alpha.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            if show_var == "both":
                # Create side-by-side subplots
                fig, axes = plt.subplots(
                    1, 2,
                    figsize=(self.config.figure.figsize[0] * 1.8, self.config.figure.figsize[1]),
                    dpi=self.config.figure.dpi,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax_obs, ax_model = axes
            else:
                fig, ax = self.create_map_figure()
                ax_obs = ax_model = ax
        else:
            fig = ax.get_figure()
            ax_obs = ax_model = ax

        # Get data
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Time average if requested
        if time_average and "time" in obs_data.dims:
            obs_data = obs_data.mean(dim="time")
            model_data = model_data.mean(dim="time")

        # Get coordinates
        if lat_var in paired_data.coords:
            lats = paired_data[lat_var].values
            lons = paired_data[lon_var].values
        else:
            lats = paired_data[lat_var].values
            lons = paired_data[lon_var].values

        # Calculate common limits
        if show_var == "both":
            all_values = np.concatenate([
                obs_data.values.flatten(),
                model_data.values.flatten(),
            ])
        elif show_var == "obs":
            all_values = obs_data.values.flatten()
        else:
            all_values = model_data.values.flatten()

        all_values = all_values[np.isfinite(all_values)]
        vmin, vmax = calculate_data_limits(all_values)

        if self.config.vmin is not None:
            vmin = self.config.vmin
        if self.config.vmax is not None:
            vmax = self.config.vmax

        # Style
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2
        a = alpha if alpha is not None else style.alpha

        # Units and label
        units = get_variable_units(paired_data, obs_var)
        var_label = get_variable_label(paired_data, obs_var)
        cbar_label = format_label_with_units(var_label or obs_var, units)

        # Plot observation
        if show_var in ("obs", "both"):
            target_ax = ax_obs if show_var == "both" else ax
            self.add_map_features(target_ax)

            mappable = self._plot_data(
                target_ax,
                obs_data.values,
                lats,
                lons,
                plot_type,
                cmap,
                vmin,
                vmax,
                ms,
                a,
            )

            if show_var == "both":
                self.add_colorbar(fig, mappable, target_ax, label=cbar_label)
                target_ax.set_title("Observations", fontsize=self.config.text.title_fontsize)

        # Plot model
        if show_var in ("model", "both"):
            target_ax = ax_model if show_var == "both" else ax
            self.add_map_features(target_ax)

            mappable = self._plot_data(
                target_ax,
                model_data.values,
                lats,
                lons,
                plot_type,
                cmap,
                vmin,
                vmax,
                ms,
                a,
            )

            if show_var == "both":
                self.add_colorbar(fig, mappable, target_ax, label=cbar_label)
                target_ax.set_title("Model", fontsize=self.config.text.title_fontsize)

        # Add colorbar and title for single panel
        if show_var != "both":
            self.add_colorbar(fig, mappable, ax, label=cbar_label)
            if self.config.title:
                title = self.config.title
            else:
                # Use base variable name without prefix for cleaner title
                base_label = get_variable_label(paired_data, obs_var, include_prefix=False)
                title = f"{base_label} ({'Observations' if show_var == 'obs' else 'Model'})"
            ax.set_title(format_plot_title(title), fontsize=self.config.text.title_fontsize)

        plt.tight_layout()
        return fig

    def _plot_data(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        plot_type: str,
        cmap: str,
        vmin: float,
        vmax: float,
        marker_size: float,
        alpha: float,
    ) -> Any:
        """Plot data on axes.

        Parameters
        ----------
        ax
            GeoAxes to plot on.
        data
            Data values.
        lats, lons
            Coordinates.
        plot_type
            'scatter' or 'pcolormesh'.
        cmap
            Colormap.
        vmin, vmax
            Value limits.
        marker_size
            Marker size for scatter.
        alpha
            Transparency.

        Returns
        -------
        Mappable
            The plot mappable for colorbar.
        """
        import cartopy.crs as ccrs

        data_flat = data.flatten()
        lats_flat = np.broadcast_to(lats, data.shape).flatten() if lats.ndim < data.ndim else lats.flatten()
        lons_flat = np.broadcast_to(lons, data.shape).flatten() if lons.ndim < data.ndim else lons.flatten()

        # Remove NaN values
        mask = np.isfinite(data_flat)
        data_flat = data_flat[mask]
        lats_flat = lats_flat[mask]
        lons_flat = lons_flat[mask]

        if plot_type == "pcolormesh" and lats.ndim == 2:
            # Gridded data - use pcolormesh
            return ax.pcolormesh(
                lons,
                lats,
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                alpha=alpha,
            )
        else:
            # Point data - use scatter
            return ax.scatter(
                lons_flat,
                lats_flat,
                c=data_flat,
                s=marker_size**2,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
                alpha=alpha,
                edgecolors="none",
            )


def plot_spatial_distribution(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    map_config: MapConfig | dict[str, Any] | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for spatial distribution plotting.

    Parameters
    ----------
    paired_data
        Paired dataset with model and observation variables.
    obs_var
        Name of observation variable.
    model_var
        Name of model variable.
    config
        Plot configuration.
    map_config
        Map configuration.
    title
        Plot title.
    **kwargs
        Additional arguments passed to plot method.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if isinstance(config, dict):
        config = PlotConfig.from_dict(config)
    elif config is None:
        config = PlotConfig()
    if isinstance(map_config, dict):
        map_config = MapConfig.from_dict(map_config)

    if title is not None:
        config = PlotConfig(
            title=title,
            figure=config.figure,
            style=config.style,
            text=config.text,
            vmin=config.vmin,
            vmax=config.vmax,
        )

    plotter = SpatialDistributionPlotter(config=config, map_config=map_config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
