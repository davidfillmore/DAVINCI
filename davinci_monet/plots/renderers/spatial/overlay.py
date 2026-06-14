"""Spatial overlay plot renderer for DAVINCI.

This module provides overlay plotting functionality for
displaying gridded y-field contours with x point overlays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    PlotConfig,
    build_series,
    format_label_with_units,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    surface_level_index,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("spatial_overlay")
class SpatialOverlayPlotter(BaseSpatialPlotter):
    """Plotter for x-vs-y spatial overlays.

    Creates maps showing y fields as filled contours with
    x values overlaid as scatter points.

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
    ...     x_var="x_o3",
    ...     y_var="y_o3",
    ...     y_field=y_data["o3"],
    ... )
    """

    name: str = "spatial_overlay"
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a spatial overlay from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional GeoAxes to plot on.
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            y_field (xr.DataArray|None, default None),
            lat_var (str, default "latitude"),
            lon_var (str, default "longitude"),
            y_lat (str, default "lat"),
            y_lon (str, default "lon"),
            time_index (int, default 0),
            level_index (int|str|None, default "surface"; "surface" auto-detects
              the surface level — last index for CESM-style ascending-pressure
              coords, else first — an int selects that index, None skips slicing),
            cmap (str, default "viridis"),
            n_levels (int, default 15),
            marker_size (float|None, default None),
            x_edgecolor (str, default "black"),
            x_linewidth (float, default 0.5).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"SpatialOverlayPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        y_field: xr.DataArray | None = kwargs.pop("y_field", None)
        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        y_lat: str = kwargs.pop("y_lat", "lat")
        y_lon: str = kwargs.pop("y_lon", "lon")
        time_index: int = kwargs.pop("time_index", 0)
        level_index: int | str | None = kwargs.pop("level_index", "surface")
        cmap: str = kwargs.pop("cmap", "viridis")
        n_levels: int = kwargs.pop("n_levels", 15)
        marker_size: float | None = kwargs.pop("marker_size", None)
        x_edgecolor: str = kwargs.pop("x_edgecolor", "black")
        x_linewidth: float = kwargs.pop("x_linewidth", 0.5)

        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Add map features
        self.add_map_features(ax)

        # Get y field for contouring
        if y_field is None:
            if y_var in paired_data:
                y_field = paired_data[y_var]
            else:
                raise ValueError(f"No y field provided and {y_var} not in paired_data")

        # Select time/level slice if needed
        if "time" in y_field.dims and y_field.dims.index("time") >= 0:
            if len(y_field.time) > 1:
                y_field = y_field.isel(time=time_index)

        if level_index is not None:
            for dim in ["z", "level", "lev", "vertical"]:
                if dim in y_field.dims:
                    idx = (
                        surface_level_index(y_field, dim)
                        if level_index == "surface"
                        else level_index
                    )
                    y_field = y_field.isel({dim: idx})
                    break

        # Get dataset coordinates
        if y_lat in y_field.coords:
            y_lats = y_field[y_lat].values
            y_lons = y_field[y_lon].values
        elif y_lat in y_field.dims:
            y_lats = y_field[y_lat].values
            y_lons = y_field[y_lon].values
        else:
            # Try common alternatives
            for lat_name in ["lat", "latitude", "y"]:
                if lat_name in y_field.coords:
                    y_lats = y_field[lat_name].values
                    break
            for lon_name in ["lon", "longitude", "x"]:
                if lon_name in y_field.coords:
                    y_lons = y_field[lon_name].values
                    break

        # Get value limits
        all_values = np.concatenate(
            [
                y_field.values.flatten(),
                paired_data[x_var].values.flatten(),
            ]
        )
        all_values = all_values[np.isfinite(all_values)]

        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Create contour levels
        levels = np.linspace(vmin, vmax, n_levels)

        # Plot y field as filled contours
        if y_lats.ndim == 1:
            # Regular grid
            lon_grid, lat_grid = np.meshgrid(y_lons, y_lats)
        else:
            # Already 2D
            lon_grid, lat_grid = y_lons, y_lats

        contour = ax.contourf(
            lon_grid,
            lat_grid,
            y_field.values,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )

        # Get dataset data
        x_data = paired_data[x_var]
        if "time" in x_data.dims:
            x_data = x_data.mean(dim="time")

        # Get dataset coordinates
        if lat_var in paired_data.coords:
            x_lats = paired_data[lat_var].values
            x_lons = paired_data[lon_var].values
        else:
            x_lats = paired_data[lat_var].values
            x_lons = paired_data[lon_var].values

        x_values = x_data.values.flatten()
        x_lats_flat = (
            np.broadcast_to(x_lats, x_data.shape).flatten()
            if x_lats.ndim < x_data.ndim
            else x_lats.flatten()
        )
        x_lons_flat = (
            np.broadcast_to(x_lons, x_data.shape).flatten()
            if x_lons.ndim < x_data.ndim
            else x_lons.flatten()
        )

        # Remove NaN values
        mask = np.isfinite(x_values)
        x_values = x_values[mask]
        x_lats_flat = x_lats_flat[mask]
        x_lons_flat = x_lons_flat[mask]

        # Get marker size
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2

        # Overlay x scatter
        scatter = ax.scatter(
            x_lons_flat,
            x_lats_flat,
            c=x_values,
            s=ms**2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            edgecolors=x_edgecolor,
            linewidths=x_linewidth,
            zorder=5,
        )

        # Add colorbar. The contour and scatter share this scale, so the label
        # is the chemistry variable itself.
        units = get_variable_units(paired_data, x_var)
        label = format_label_with_units(
            get_variable_label(paired_data, x_var, include_prefix=False) or x_var,
            units,
        )
        self.add_colorbar(fig, contour, ax, label=label)

        # Title
        if self.config.title:
            self.set_title(ax, self.config.title)
        else:
            var_label = get_variable_label(paired_data, x_var)
            self.set_title(ax, f"{var_label}: Y (contour) vs X (points)")

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        y_field: xr.DataArray | None = None,
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        y_lat: str = "lat",
        y_lon: str = "lon",
        time_index: int = 0,
        level_index: int | str | None = "surface",
        cmap: str = "viridis",
        n_levels: int = 15,
        marker_size: float | None = None,
        x_edgecolor: str = "black",
        x_linewidth: float = 0.5,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial overlay plot.

        Thin wrapper around :meth:`render`. See that method for parameter docs.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        ax
            Optional GeoAxes to plot on.
        y_field
            Optional separate y field for contouring.
            If None, tries to use y_var from paired_data.
        lat_var
            Name of latitude coordinate for datasets.
        lon_var
            Name of longitude coordinate for datasets.
        y_lat
            Name of latitude dimension in y field.
        y_lon
            Name of longitude dimension in y field.
        time_index
            Time index to plot if dataset has time dimension.
        level_index
            Vertical level to plot if the dataset has a vertical dimension.
            ``"surface"`` (default) auto-detects the surface level (last index
            for CESM-style ascending-pressure coordinates, else first); an int
            selects that index explicitly; None skips level selection.
        cmap
            Colormap name.
        n_levels
            Number of contour levels.
        marker_size
            Override marker size.
        x_edgecolor
            Edge color for x (geometry) markers.
        x_linewidth
            Edge line width for x (geometry) markers.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, x_var, y_var),
            ax=ax,
            y_field=y_field,
            lat_var=lat_var,
            lon_var=lon_var,
            y_lat=y_lat,
            y_lon=y_lon,
            time_index=time_index,
            level_index=level_index,
            cmap=cmap,
            n_levels=n_levels,
            marker_size=marker_size,
            x_edgecolor=x_edgecolor,
            x_linewidth=x_linewidth,
            **kwargs,
        )


def plot_spatial_overlay(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    map_config: MapConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for spatial overlay plotting.

    Parameters
    ----------
    paired_data
        Paired dataset with x and y variables.
    x_var
        Name of the x variable.
    y_var
        Name of the y variable.
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
    return plotter.plot(paired_data, x_var, y_var, **kwargs)
