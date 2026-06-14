"""Spatial bias plot renderer for DAVINCI.

This module provides spatial bias plotting functionality for
visualizing the difference between x and y values
on a map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    PlotConfig,
    build_series,
    calculate_symmetric_limits,
    format_label_with_units,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    detect_spatial_geometry,
    get_domain_extent,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("spatial_bias")
class SpatialBiasPlotter(BaseSpatialPlotter):
    """Plotter for spatial bias maps.

    Creates maps showing the spatial distribution of dataset-dataset
    bias, with points colored by bias magnitude.

    Parameters
    ----------
    config
        Plot configuration.
    map_config
        Map-specific configuration.

    Examples
    --------
    >>> plotter = SpatialBiasPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     x_var="geometry_o3",
    ...     y_var="dataset_o3",
    ... )
    """

    name: str = "spatial_bias"
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a spatial bias map from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional GeoAxes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the bias rendering logic; same kwargs as ``plot()``.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"SpatialBiasPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        time_average: bool = kwargs.pop("time_average", True)
        cmap: str = kwargs.pop("cmap", "RdBu_r")
        marker_size: float | None = kwargs.pop("marker_size", None)
        symmetric_cbar: bool = kwargs.pop("symmetric_cbar", True)
        show_zero_line: bool = kwargs.pop("show_zero_line", True)
        show_site_labels: bool = kwargs.pop("show_site_labels", False)
        site_label_var: str = kwargs.pop("site_label_var", "site_name")
        label_sites: list[str] | None = kwargs.pop("label_sites", None)
        city_labels: dict[str, tuple[float, float]] | None = kwargs.pop("city_labels", None)
        label_fontsize: int | None = kwargs.pop("label_fontsize", None)
        plot_type: str = kwargs.pop("plot_type", "auto")

        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_map_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Add map features
        self.add_map_features(ax)

        # Calculate bias
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]
        bias = y_data - x_data

        # Time average if requested
        if time_average and "time" in bias.dims:
            bias = bias.mean(dim="time")
            x_data = x_data.mean(dim="time")

        # Get coordinates — resolve common aliases
        lat_candidates = [lat_var, "lat", "latitude", "LAT", "Latitude"]
        lon_candidates = [lon_var, "lon", "longitude", "LON", "Longitude"]
        resolved_lat = next(
            (c for c in lat_candidates if c in paired_data.coords or c in paired_data),
            None,
        )
        resolved_lon = next(
            (c for c in lon_candidates if c in paired_data.coords or c in paired_data),
            None,
        )
        if resolved_lat is None or resolved_lon is None:
            raise ValueError(
                f"Could not find latitude/longitude coordinates in dataset. "
                f"Available coords: {list(paired_data.coords)}"
            )
        lats = paired_data[resolved_lat].values
        lons = paired_data[resolved_lon].values

        # Shift 0..360 longitudes to -180..180 for cartopy PlateCarree
        if lons.ndim == 1 and np.any(lons > 180):
            lons = np.where(lons > 180, lons - 360, lons)
            # Re-sort lon axis so pcolormesh gets monotonic coords
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            # Find lon dimension in bias and reorder data to match
            lon_dim = resolved_lon
            if lon_dim in bias.dims:
                bias = bias.isel({lon_dim: sort_idx})
                x_data = x_data.isel({lon_dim: sort_idx})
        elif lons.ndim > 1 and np.any(lons > 180):
            lons = np.where(lons > 180, lons - 360, lons)

        # Detect data geometry from the *DataArray* dims (not just the numpy
        # arrays), since for point/site datasets lats and lons share a
        # single dim and must not be meshgridded as if they were grid axes.
        lat_da = paired_data[resolved_lat]
        lon_da = paired_data[resolved_lon]
        _geometry = detect_spatial_geometry(lat_da, lon_da, bias)
        is_point_data = _geometry == "point"

        if is_point_data:
            # Point/site data: drop singleton dims (e.g. AirNow y=1
            # residual) and broadcast lat/lon along any remaining non-site
            # dims (e.g. if time_average wasn't applied).
            bias = bias.squeeze(drop=True)
            site_dim = lat_da.dims[0]
            if bias.ndim > 1 and site_dim in bias.dims:
                lats_flat = lat_da.broadcast_like(bias).values.flatten()
                lons_flat = lon_da.broadcast_like(bias).values.flatten()
            else:
                lats_flat = lat_da.values.flatten()
                lons_flat = lon_da.values.flatten()
            bias_values = bias.values.flatten()
        elif lats.ndim == 1 and lons.ndim == 1 and bias.ndim >= 2:
            # Regular grid: meshgrid to match bias shape (lon, lat) or (lat, lon)
            lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
            # Match bias shape — if dims are (lon, lat), indexing="ij" is correct
            if lon_grid.shape != bias.shape:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            lats_flat = lat_grid.flatten()
            lons_flat = lon_grid.flatten()
            bias_values = bias.values.flatten()
        elif lats.ndim < bias.ndim:
            lats_flat = np.broadcast_to(lats, bias.shape).flatten()
            lons_flat = np.broadcast_to(lons, bias.shape).flatten()
            bias_values = bias.values.flatten()
        else:
            lats_flat = lats.flatten()
            lons_flat = lons.flatten()
            bias_values = bias.values.flatten()

        # Remove NaN values
        mask = np.isfinite(bias_values) & np.isfinite(lats_flat) & np.isfinite(lons_flat)
        bias_values = bias_values[mask]
        lats_flat = lats_flat[mask]
        lons_flat = lons_flat[mask]

        if len(bias_values) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.config.text.fontsize,
            )
            return fig

        # Calculate color limits
        if symmetric_cbar:
            vmin, vmax = calculate_symmetric_limits(bias_values)
        else:
            vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(bias_values)
            vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(bias_values)

        # Override with config if set
        if self.config.vmin is not None:
            vmin = self.config.vmin
        if self.config.vmax is not None:
            vmax = self.config.vmax

        # Create normalization
        if symmetric_cbar and vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = None

        # Get marker size
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize * 2

        # Resolve "auto" to a concrete method based on data geometry: gridded
        # data (1-D lat/lon axes with a 2-D+ field, or 2-D curvilinear coords)
        # renders as a filled pcolormesh field; point/site data uses scatter.
        effective_plot_type = plot_type
        if plot_type == "auto":
            effective_plot_type = (
                "pcolormesh" if _geometry in ("regular_grid", "curvilinear_grid") else "scatter"
            )

        # Choose plot method based on data geometry
        if effective_plot_type == "pcolormesh" and lats.ndim == 1 and bias.ndim >= 2:
            # Regular grid with 1D coords — use pcolormesh
            bias_2d = bias.values
            scatter = ax.pcolormesh(
                lons,
                lats,
                bias_2d.T if bias_2d.shape[0] == len(lons) else bias_2d,
                cmap=cmap,
                norm=norm,
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
                transform=ccrs.PlateCarree(),
                alpha=style.alpha,
            )
        elif effective_plot_type == "pcolormesh" and lats.ndim == 2:
            # Curvilinear grid — use pcolormesh with 2D coords
            scatter = ax.pcolormesh(
                lons,
                lats,
                bias.values,
                cmap=cmap,
                norm=norm,
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
                transform=ccrs.PlateCarree(),
                alpha=style.alpha,
            )
        else:
            # Point data — use scatter
            scatter = ax.scatter(  # type: ignore[assignment]
                lons_flat,
                lats_flat,
                c=bias_values,
                s=ms**2,
                cmap=cmap,
                norm=norm,
                vmin=vmin if norm is None else None,
                vmax=vmax if norm is None else None,
                transform=ccrs.PlateCarree(),
                edgecolors="none",
                alpha=style.alpha,
            )

        # Add colorbar
        units = get_variable_units(paired_data, x_var)
        label = format_label_with_units("Bias (y - x)", units)
        self.add_colorbar(fig, scatter, ax, label=label)

        # Use config site_label size if not specified
        if label_fontsize is None:
            label_fontsize = self.config.text.site_label  # type: ignore[assignment]

        # Add site labels if requested
        if show_site_labels and site_label_var in paired_data.coords:
            site_labels = paired_data[site_label_var].values
            # Get unique site locations (after time averaging, each site has one point)
            unique_lons, unique_idx = np.unique(lons_flat, return_index=True)
            for i, idx in enumerate(unique_idx):
                site_idx = idx % len(site_labels) if len(site_labels) > 0 else 0
                if site_idx < len(site_labels):
                    site_name = str(site_labels[site_idx])
                    # Filter to specific sites if label_sites is provided
                    if label_sites is not None and site_name not in label_sites:
                        continue
                    ax.annotate(
                        site_name,
                        (lons_flat[idx], lats_flat[idx]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=label_fontsize,
                        alpha=0.8,
                        transform=ccrs.PlateCarree(),
                    )

        # Add city labels if provided
        if city_labels:
            for city_name, (lat, lon) in city_labels.items():
                ax.annotate(
                    city_name,
                    (lon, lat),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    fontweight="bold",
                    alpha=0.9,
                    transform=ccrs.PlateCarree(),
                )
                # Add a small marker for the city location
                ax.plot(
                    lon,
                    lat,
                    marker="*",
                    markersize=6,
                    color="black",
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )

        # Title
        var_label = get_variable_label(paired_data, x_var)
        if self.config.title:
            self.set_title(ax, self.config.title)
        else:
            self.set_title(ax, f"{var_label} Bias")

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        time_average: bool = True,
        cmap: str = "RdBu_r",
        marker_size: float | None = None,
        symmetric_cbar: bool = True,
        show_zero_line: bool = True,
        show_site_labels: bool = False,
        site_label_var: str = "site_name",
        label_sites: list[str] | None = None,
        city_labels: dict[str, tuple[float, float]] | None = None,
        label_fontsize: int | None = None,
        plot_type: str = "auto",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial bias plot.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        ax
            Optional GeoAxes to plot on. If None, creates new figure.
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
        symmetric_cbar
            If True, make colorbar symmetric around zero.
        show_zero_line
            If True, highlight zero contour (for gridded data).
        plot_type
            How to render the bias field.  ``"auto"`` (default) chooses
            ``"pcolormesh"`` for gridded data (1-D lat/lon axes with a 2-D+
            field, or 2-D curvilinear coordinates) and ``"scatter"`` for
            point/site data (lat/lon share a single dataset dimension).
            Pass ``"scatter"`` or ``"pcolormesh"`` to override the automatic
            selection.
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
            lat_var=lat_var,
            lon_var=lon_var,
            time_average=time_average,
            cmap=cmap,
            marker_size=marker_size,
            symmetric_cbar=symmetric_cbar,
            show_zero_line=show_zero_line,
            show_site_labels=show_site_labels,
            site_label_var=site_label_var,
            label_sites=label_sites,
            city_labels=city_labels,
            label_fontsize=label_fontsize,
            plot_type=plot_type,
            **kwargs,
        )


def plot_spatial_bias(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    map_config: MapConfig | dict[str, Any] | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for spatial bias plotting.

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

    plotter = SpatialBiasPlotter(config=config, map_config=map_config)
    return plotter.plot(paired_data, x_var, y_var, **kwargs)
