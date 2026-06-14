"""Spatial overlay plot renderer for DAVINCI.

This module provides overlay plotting functionality for
displaying dataset contours with dataset point overlays.
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
    """Plotter for dataset-dataset spatial overlays.

    Creates maps showing dataset fields as filled contours with
    dataset values overlaid as scatter points.

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
    ...     x_var="geometry_o3",
    ...     y_var="dataset_o3",
    ...     dataset_field=y_data["o3"],
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
            Exactly 2 series: one geometry (geometry) and one dataset (dataset).
        ax
            Optional GeoAxes to plot on.
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            dataset_field (xr.DataArray|None, default None),
            lat_var (str, default "latitude"),
            lon_var (str, default "longitude"),
            dataset_lat (str, default "lat"),
            dataset_lon (str, default "lon"),
            time_index (int, default 0),
            level_index (int|str|None, default "surface"; "surface" auto-detects
              the surface level — last index for CESM-style ascending-pressure
              coords, else first — an int selects that index, None skips slicing),
            cmap (str, default "viridis"),
            n_levels (int, default 15),
            marker_size (float|None, default None),
            geometry_edgecolor (str, default "black"),
            geometry_linewidth (float, default 0.5).

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

        dataset_field: xr.DataArray | None = kwargs.pop("dataset_field", None)
        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        dataset_lat: str = kwargs.pop("dataset_lat", "lat")
        dataset_lon: str = kwargs.pop("dataset_lon", "lon")
        time_index: int = kwargs.pop("time_index", 0)
        level_index: int | str | None = kwargs.pop("level_index", "surface")
        cmap: str = kwargs.pop("cmap", "viridis")
        n_levels: int = kwargs.pop("n_levels", 15)
        marker_size: float | None = kwargs.pop("marker_size", None)
        geometry_edgecolor: str = kwargs.pop("geometry_edgecolor", "black")
        geometry_linewidth: float = kwargs.pop("geometry_linewidth", 0.5)

        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Add map features
        self.add_map_features(ax)

        # Get dataset field for contouring
        if dataset_field is None:
            if y_var in paired_data:
                dataset_field = paired_data[y_var]
            else:
                raise ValueError(f"No dataset field provided and {y_var} not in paired_data")

        # Select time/level slice if needed
        if "time" in dataset_field.dims and dataset_field.dims.index("time") >= 0:
            if len(dataset_field.time) > 1:
                dataset_field = dataset_field.isel(time=time_index)

        if level_index is not None:
            for dim in ["z", "level", "lev", "vertical"]:
                if dim in dataset_field.dims:
                    idx = (
                        surface_level_index(dataset_field, dim)
                        if level_index == "surface"
                        else level_index
                    )
                    dataset_field = dataset_field.isel({dim: idx})
                    break

        # Get dataset coordinates
        if dataset_lat in dataset_field.coords:
            dataset_lats = dataset_field[dataset_lat].values
            dataset_lons = dataset_field[dataset_lon].values
        elif dataset_lat in dataset_field.dims:
            dataset_lats = dataset_field[dataset_lat].values
            dataset_lons = dataset_field[dataset_lon].values
        else:
            # Try common alternatives
            for lat_name in ["lat", "latitude", "y"]:
                if lat_name in dataset_field.coords:
                    dataset_lats = dataset_field[lat_name].values
                    break
            for lon_name in ["lon", "longitude", "x"]:
                if lon_name in dataset_field.coords:
                    dataset_lons = dataset_field[lon_name].values
                    break

        # Get value limits
        all_values = np.concatenate(
            [
                dataset_field.values.flatten(),
                paired_data[x_var].values.flatten(),
            ]
        )
        all_values = all_values[np.isfinite(all_values)]

        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Create contour levels
        levels = np.linspace(vmin, vmax, n_levels)

        # Plot dataset field as filled contours
        if dataset_lats.ndim == 1:
            # Regular grid
            lon_grid, lat_grid = np.meshgrid(dataset_lons, dataset_lats)
        else:
            # Already 2D
            lon_grid, lat_grid = dataset_lons, dataset_lats

        contour = ax.contourf(
            lon_grid,
            lat_grid,
            dataset_field.values,
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
            geometry_lats = paired_data[lat_var].values
            geometry_lons = paired_data[lon_var].values
        else:
            geometry_lats = paired_data[lat_var].values
            geometry_lons = paired_data[lon_var].values

        geometry_values = x_data.values.flatten()
        geometry_lats_flat = (
            np.broadcast_to(geometry_lats, x_data.shape).flatten()
            if geometry_lats.ndim < x_data.ndim
            else geometry_lats.flatten()
        )
        geometry_lons_flat = (
            np.broadcast_to(geometry_lons, x_data.shape).flatten()
            if geometry_lons.ndim < x_data.ndim
            else geometry_lons.flatten()
        )

        # Remove NaN values
        mask = np.isfinite(geometry_values)
        geometry_values = geometry_values[mask]
        geometry_lats_flat = geometry_lats_flat[mask]
        geometry_lons_flat = geometry_lons_flat[mask]

        # Get marker size
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2

        # Overlay dataset scatter
        scatter = ax.scatter(
            geometry_lons_flat,
            geometry_lats_flat,
            c=geometry_values,
            s=ms**2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            edgecolors=geometry_edgecolor,
            linewidths=geometry_linewidth,
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
            self.set_title(ax, f"{var_label}: Dataset (contour) vs Geometry (points)")

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        dataset_field: xr.DataArray | None = None,
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        dataset_lat: str = "lat",
        dataset_lon: str = "lon",
        time_index: int = 0,
        level_index: int | str | None = "surface",
        cmap: str = "viridis",
        n_levels: int = 15,
        marker_size: float | None = None,
        geometry_edgecolor: str = "black",
        geometry_linewidth: float = 0.5,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial overlay plot.

        Thin wrapper around :meth:`render`. See that method for parameter docs.

        Parameters
        ----------
        paired_data
            Paired dataset with dataset and dataset variables.
        x_var
            Name of dataset variable.
        y_var
            Name of dataset variable.
        ax
            Optional GeoAxes to plot on.
        dataset_field
            Optional separate dataset field for contouring.
            If None, tries to use y_var from paired_data.
        lat_var
            Name of latitude coordinate for datasets.
        lon_var
            Name of longitude coordinate for datasets.
        dataset_lat
            Name of latitude dimension in dataset field.
        dataset_lon
            Name of longitude dimension in dataset field.
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
        geometry_edgecolor
            Edge color for dataset markers.
        geometry_linewidth
            Edge line width for dataset markers.
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
            dataset_field=dataset_field,
            lat_var=lat_var,
            lon_var=lon_var,
            dataset_lat=dataset_lat,
            dataset_lon=dataset_lon,
            time_index=time_index,
            level_index=level_index,
            cmap=cmap,
            n_levels=n_levels,
            marker_size=marker_size,
            geometry_edgecolor=geometry_edgecolor,
            geometry_linewidth=geometry_linewidth,
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
        Paired dataset with dataset and dataset variables.
    x_var
        Name of dataset variable.
    y_var
        Name of dataset variable.
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
