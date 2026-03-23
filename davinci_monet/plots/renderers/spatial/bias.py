"""Spatial bias plot renderer for DAVINCI.

This module provides spatial bias plotting functionality for
visualizing the difference between model and observation values
on a map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.plots.base import (
    PlotConfig,
    calculate_symmetric_limits,
    format_label_with_units,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.renderers.spatial.base import (
    BaseSpatialPlotter,
    MapConfig,
    get_domain_extent,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("spatial_bias")
class SpatialBiasPlotter(BaseSpatialPlotter):
    """Plotter for spatial bias maps.

    Creates maps showing the spatial distribution of model-observation
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
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ... )
    """

    name: str = "spatial_bias"
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
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
        plot_type: str = "scatter",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a spatial bias plot.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
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
            fig = ax.get_figure()

        # Add map features
        self.add_map_features(ax)

        # Calculate bias
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]
        bias = model_data - obs_data

        # Time average if requested
        if time_average and "time" in bias.dims:
            bias = bias.mean(dim="time")
            obs_data = obs_data.mean(dim="time")

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
                obs_data = obs_data.isel({lon_dim: sort_idx})
        elif lons.ndim > 1 and np.any(lons > 180):
            lons = np.where(lons > 180, lons - 360, lons)

        bias_values = bias.values.flatten()
        if lats.ndim == 1 and lons.ndim == 1 and bias.ndim >= 2:
            # Regular grid: meshgrid to match bias shape (lon, lat) or (lat, lon)
            lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
            # Match bias shape — if dims are (lon, lat), indexing="ij" is correct
            if lon_grid.shape != bias.shape:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            lats_flat = lat_grid.flatten()
            lons_flat = lon_grid.flatten()
        elif lats.ndim < bias.ndim:
            lats_flat = np.broadcast_to(lats, bias.shape).flatten()
            lons_flat = np.broadcast_to(lons, bias.shape).flatten()
        else:
            lats_flat = lats.flatten()
            lons_flat = lons.flatten()

        # Remove NaN values
        mask = np.isfinite(bias_values) & np.isfinite(lats_flat) & np.isfinite(lons_flat)
        bias_values = bias_values[mask]
        lats_flat = lats_flat[mask]
        lons_flat = lons_flat[mask]

        if len(bias_values) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=self.config.text.fontsize)
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

        # Choose plot method based on data geometry
        if plot_type == "pcolormesh" and lats.ndim == 1 and bias.ndim >= 2:
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
        elif plot_type == "pcolormesh" and lats.ndim == 2:
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
            scatter = ax.scatter(
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
        units = get_variable_units(paired_data, obs_var)
        label = format_label_with_units("Bias (Model - Obs)", units)
        self.add_colorbar(fig, scatter, ax, label=label)

        # Use config site_label size if not specified
        if label_fontsize is None:
            label_fontsize = self.config.text.site_label

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
                    lon, lat,
                    marker="*",
                    markersize=6,
                    color="black",
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )

        # Title
        var_label = get_variable_label(paired_data, obs_var)
        if self.config.title:
            ax.set_title(format_plot_title(self.config.title), fontsize=self.config.text.title_fontsize)
        else:
            ax.set_title(format_plot_title(f"{var_label} Bias"), fontsize=self.config.text.title_fontsize)

        return fig


def plot_spatial_bias(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    map_config: MapConfig | dict[str, Any] | None = None,
    title: str | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for spatial bias plotting.

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

    plotter = SpatialBiasPlotter(config=config, map_config=map_config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
