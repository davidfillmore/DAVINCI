"""Spatial distribution plot renderer for DAVINCI.

This module provides spatial distribution plotting functionality for
displaying observation or model values on a map without comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    PlotConfig,
    build_series,
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
    detect_spatial_geometry,
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
    default_figsize: tuple[float, float] = (8, 5)  # Wide for geographic extent

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a spatial distribution map from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one reference (obs) and one comparand (model).
        ax
            Optional GeoAxes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the distribution rendering logic; same kwargs as
            ``plot()`` (including ``show_var`` which controls which side
            is shown).

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"SpatialDistributionPlotter.render requires exactly 2 series;"
                f" got {len(series)}."
            )
        ref = next((s for s in series if s.pair_role == "reference"), series[0])
        comp = next((s for s in series if s.pair_role == "comparand"), series[1])
        paired_data = ref.dataset
        obs_var = ref.var_name
        model_var = comp.var_name

        show_var: str = kwargs.pop("show_var", "obs")
        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        time_average: bool = kwargs.pop("time_average", True)
        cmap: str = kwargs.pop("cmap", "viridis")
        marker_size: float | None = kwargs.pop("marker_size", None)
        plot_type: str = kwargs.pop("plot_type", "auto")
        alpha: float | None = kwargs.pop("alpha", None)

        import cartopy.crs as ccrs

        # Create figure if needed
        if ax is None:
            if show_var == "both":
                # Create side-by-side subplots with standard size
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=(8, 5),
                    dpi=self.config.figure.dpi,
                    subplot_kw={"projection": ccrs.PlateCarree()},
                )
                ax_obs, ax_model = axes
            else:
                fig, ax = self.create_map_figure()
                ax_obs = ax_model = ax
        else:
            fig = ax.get_figure()  # type: ignore[assignment]
            ax_obs = ax_model = ax

        # Get data
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Time average if requested
        if time_average and "time" in obs_data.dims:
            obs_data = obs_data.mean(dim="time")
            model_data = model_data.mean(dim="time")

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
            sort_idx = np.argsort(lons)
            lons = lons[sort_idx]
            lon_dim = resolved_lon
            if lon_dim in obs_data.dims:
                obs_data = obs_data.isel({lon_dim: sort_idx})
                model_data = model_data.isel({lon_dim: sort_idx})
        elif lons.ndim > 1 and np.any(lons > 180):
            lons = np.where(lons > 180, lons - 360, lons)

        # Detect data geometry from the *DataArray* dims (not just the numpy
        # arrays), since for point/site observations lats and lons share a
        # single dim and must not be meshgridded as if they were grid axes.
        lat_da = paired_data[resolved_lat]
        lon_da = paired_data[resolved_lon]
        # Reference DataArray for geometry detection — use obs_data dims.
        _geometry = detect_spatial_geometry(lat_da, lon_da, obs_data)

        # Resolve "auto" to a concrete method based on data geometry: gridded
        # data (1-D lat/lon axes with a 2-D+ field, or 2-D curvilinear coords)
        # renders as a filled pcolormesh field; point/site data uses scatter.
        effective_plot_type = plot_type
        if plot_type == "auto":
            effective_plot_type = (
                "pcolormesh" if _geometry in ("regular_grid", "curvilinear_grid") else "scatter"
            )

        # Calculate common limits
        if show_var == "both":
            all_values = np.concatenate(
                [
                    obs_data.values.flatten(),
                    model_data.values.flatten(),
                ]
            )
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
            self.add_map_features(target_ax)  # type: ignore[arg-type]

            mappable = self._plot_data(
                target_ax,  # type: ignore[arg-type]
                obs_data.values,
                lats,
                lons,
                effective_plot_type,
                cmap,
                vmin,
                vmax,
                ms,
                a,
            )

            if show_var == "both":
                self.add_colorbar(fig, mappable, target_ax, label=cbar_label)  # type: ignore[arg-type]
                target_ax.set_title("Observations", fontsize=self.config.text.title_fontsize)  # type: ignore[union-attr]

        # Plot model
        if show_var in ("model", "both"):
            target_ax = ax_model if show_var == "both" else ax
            self.add_map_features(target_ax)  # type: ignore[arg-type]

            mappable = self._plot_data(
                target_ax,  # type: ignore[arg-type]
                model_data.values,
                lats,
                lons,
                effective_plot_type,
                cmap,
                vmin,
                vmax,
                ms,
                a,
            )

            if show_var == "both":
                self.add_colorbar(fig, mappable, target_ax, label=cbar_label)  # type: ignore[arg-type]
                target_ax.set_title("Model", fontsize=self.config.text.title_fontsize)  # type: ignore[union-attr]

        # Add colorbar and title for single panel
        if show_var != "both":
            self.add_colorbar(fig, mappable, ax, label=cbar_label)  # type: ignore[arg-type]
            if self.config.title:
                title = self.config.title
            else:
                # Use base variable name without prefix for cleaner title
                base_label = get_variable_label(paired_data, obs_var, include_prefix=False)
                title = f"{base_label} ({'Observations' if show_var == 'obs' else 'Model'})"
            ax.set_title(format_plot_title(title), fontsize=self.config.text.title_fontsize)  # type: ignore[union-attr]

        plt.tight_layout()
        return fig

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
        plot_type: Literal["auto", "scatter", "pcolormesh"] = "auto",
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
            How to render the distribution field.  ``"auto"`` (default)
            chooses ``"pcolormesh"`` for gridded data (1-D lat/lon axes
            with a 2-D+ field, or 2-D curvilinear coordinates) and
            ``"scatter"`` for point/site data (lat/lon share a single
            observation dimension).  Pass ``"scatter"`` or
            ``"pcolormesh"`` to override the automatic selection.
        alpha
            Override alpha.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, obs_var, model_var),
            ax=ax,
            show_var=show_var,
            lat_var=lat_var,
            lon_var=lon_var,
            time_average=time_average,
            cmap=cmap,
            marker_size=marker_size,
            plot_type=plot_type,
            alpha=alpha,
            **kwargs,
        )

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
        if plot_type != "scatter" and lats.ndim == 1 and lons.ndim == 1 and data.ndim >= 2:
            # Regular grid: lat/lon are independent axes — build a meshgrid so
            # each grid cell gets the correct coordinate.  Only do this for
            # pcolormesh; for scatter (point/site data) lat/lon are already
            # per-observation and must be broadcast, not meshgridded.
            lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
            if lon_grid.shape != data.shape:
                lon_grid, lat_grid = np.meshgrid(lons, lats)
            lats_flat = lat_grid.flatten()
            lons_flat = lon_grid.flatten()
        elif lats.ndim < data.ndim:
            lats_flat = np.broadcast_to(lats, data.shape).flatten()
            lons_flat = np.broadcast_to(lons, data.shape).flatten()
        else:
            lats_flat = lats.flatten()
            lons_flat = lons.flatten()

        # Remove NaN values
        mask = np.isfinite(data_flat)
        data_flat = data_flat[mask]
        lats_flat = lats_flat[mask]
        lons_flat = lons_flat[mask]

        if plot_type == "pcolormesh" and lats.ndim == 2:
            # Curvilinear grid - use pcolormesh with 2D coords
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
        elif plot_type == "pcolormesh" and lats.ndim == 1 and data.ndim >= 2:
            # Regular grid with 1D coords - pcolormesh handles natively
            return ax.pcolormesh(
                lons,
                lats,
                data.T if data.shape[0] == len(lons) else data,
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
