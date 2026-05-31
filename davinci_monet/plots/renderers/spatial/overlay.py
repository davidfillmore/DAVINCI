"""Spatial overlay plot renderer for DAVINCI.

This module provides overlay plotting functionality for
displaying model contours with observation point overlays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.base import (
    PlotConfig,
    format_label_with_units,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import BaseSpatialPlotter, MapConfig

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("spatial_overlay")
class SpatialOverlayPlotter(BaseSpatialPlotter):
    """Plotter for model-observation spatial overlays.

    Creates maps showing model fields as filled contours with
    observation values overlaid as scatter points.

    Parameters
    ----------
    config
        Plot configuration.
    map_config
        Map-specific configuration.

    Examples
    --------
    >>> plotter = SpatialOverlayPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     model_field=model_data["o3"],
    ... )
    """

    name: str = "spatial_overlay"
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        model_field: xr.DataArray | None = None,
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        model_lat: str = "lat",
        model_lon: str = "lon",
        time_index: int = 0,
        level_index: int | None = 0,
        cmap: str = "viridis",
        n_levels: int = 15,
        marker_size: float | None = None,
        obs_edgecolor: str = "black",
        obs_linewidth: float = 0.5,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial overlay plot.

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
        model_field
            Optional separate model field for contouring.
            If None, tries to use model_var from paired_data.
        lat_var
            Name of latitude coordinate for observations.
        lon_var
            Name of longitude coordinate for observations.
        model_lat
            Name of latitude dimension in model field.
        model_lon
            Name of longitude dimension in model field.
        time_index
            Time index to plot if model has time dimension.
        level_index
            Level index to plot if model has vertical dimension.
            Set to None to skip level selection.
        cmap
            Colormap name.
        n_levels
            Number of contour levels.
        marker_size
            Override marker size.
        obs_edgecolor
            Edge color for observation markers.
        obs_linewidth
            Edge line width for observation markers.
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
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Add map features
        self.add_map_features(ax)

        # Get model field for contouring
        if model_field is None:
            if model_var in paired_data:
                model_field = paired_data[model_var]
            else:
                raise ValueError(f"No model field provided and {model_var} not in paired_data")

        # Select time/level slice if needed
        if "time" in model_field.dims and model_field.dims.index("time") >= 0:
            if len(model_field.time) > 1:
                model_field = model_field.isel(time=time_index)

        if level_index is not None:
            for dim in ["z", "level", "lev", "vertical"]:
                if dim in model_field.dims:
                    model_field = model_field.isel({dim: level_index})
                    break

        # Get model coordinates
        if model_lat in model_field.coords:
            model_lats = model_field[model_lat].values
            model_lons = model_field[model_lon].values
        elif model_lat in model_field.dims:
            model_lats = model_field[model_lat].values
            model_lons = model_field[model_lon].values
        else:
            # Try common alternatives
            for lat_name in ["lat", "latitude", "y"]:
                if lat_name in model_field.coords:
                    model_lats = model_field[lat_name].values
                    break
            for lon_name in ["lon", "longitude", "x"]:
                if lon_name in model_field.coords:
                    model_lons = model_field[lon_name].values
                    break

        # Get value limits
        all_values = np.concatenate(
            [
                model_field.values.flatten(),
                paired_data[obs_var].values.flatten(),
            ]
        )
        all_values = all_values[np.isfinite(all_values)]

        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Create contour levels
        levels = np.linspace(vmin, vmax, n_levels)

        # Plot model field as filled contours
        if model_lats.ndim == 1:
            # Regular grid
            lon_grid, lat_grid = np.meshgrid(model_lons, model_lats)
        else:
            # Already 2D
            lon_grid, lat_grid = model_lons, model_lats

        contour = ax.contourf(
            lon_grid,
            lat_grid,
            model_field.values,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        # Get observation data
        obs_data = paired_data[obs_var]
        if "time" in obs_data.dims:
            obs_data = obs_data.mean(dim="time")

        # Get observation coordinates
        if lat_var in paired_data.coords:
            obs_lats = paired_data[lat_var].values
            obs_lons = paired_data[lon_var].values
        else:
            obs_lats = paired_data[lat_var].values
            obs_lons = paired_data[lon_var].values

        obs_values = obs_data.values.flatten()
        obs_lats_flat = (
            np.broadcast_to(obs_lats, obs_data.shape).flatten()
            if obs_lats.ndim < obs_data.ndim
            else obs_lats.flatten()
        )
        obs_lons_flat = (
            np.broadcast_to(obs_lons, obs_data.shape).flatten()
            if obs_lons.ndim < obs_data.ndim
            else obs_lons.flatten()
        )

        # Remove NaN values
        mask = np.isfinite(obs_values)
        obs_values = obs_values[mask]
        obs_lats_flat = obs_lats_flat[mask]
        obs_lons_flat = obs_lons_flat[mask]

        # Get marker size
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2

        # Overlay observation scatter
        scatter = ax.scatter(
            obs_lons_flat,
            obs_lats_flat,
            c=obs_values,
            s=ms**2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            edgecolors=obs_edgecolor,
            linewidths=obs_linewidth,
            zorder=5,
        )

        # Add colorbar. The contour (model) and scatter (obs) share this scale,
        # so the label is the chemistry variable itself — no "Observed"/"Modeled"
        # prefix (which would mislabel half the layer).
        units = get_variable_units(paired_data, obs_var)
        label = format_label_with_units(
            get_variable_label(paired_data, obs_var, include_prefix=False) or obs_var,
            units,
        )
        self.add_colorbar(fig, contour, ax, label=label)

        # Title
        if self.config.title:
            ax.set_title(
                format_plot_title(self.config.title), fontsize=self.config.text.title_fontsize
            )
        else:
            var_label = get_variable_label(paired_data, obs_var)
            ax.set_title(
                format_plot_title(f"{var_label}: Model (contour) vs Obs (points)"),
                fontsize=self.config.text.title_fontsize,
            )

        return fig


def plot_spatial_overlay(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    map_config: MapConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for spatial overlay plotting.

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
    **kwargs
        Additional arguments passed to plot method.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if isinstance(config, dict):
        config = PlotConfig.from_dict(config)
    if isinstance(map_config, dict):
        map_config = MapConfig.from_dict(map_config)

    plotter = SpatialOverlayPlotter(config=config, map_config=map_config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
