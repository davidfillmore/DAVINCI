"""3D flight track map renderer for observation-only data.

Renders a 3D plot showing an aircraft flight path with longitude, latitude,
and altitude axes, colored by a variable value (e.g., O3 concentration).
Includes coastline outlines on the surface plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from davinci_monet.plots.base import (
    BasePlotter,
    format_label_with_units,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_alias, register_plotter
from davinci_monet.plots.renderers.track_map_3d import (
    _get_border_segments,
    _get_coastline_segments,
    _render_surface_map,
)
from davinci_monet.plots.style import get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("flight_track")
class FlightTrackPlotter(BasePlotter):
    """Plotter for 3D flight track maps colored by variable value.

    Creates a 3D plot with the flight path rendered as scatter points
    colored by the selected variable, with coastlines on the surface plane.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = FlightTrackPlotter()
    >>> fig = plotter.plot(obs_data, "O3", title="DC3 Flight O3")
    """

    name: str = "flight_track"
    default_figsize: tuple[float, float] = (7, 6)

    def render(
        self,
        series: list[Any],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> Any:
        """Unified entry: render a single source's flight track."""
        s = series[0]
        return self.plot(s.dataset, s.var_name, **kwargs)

    def plot(  # type: ignore[override]
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        title: str | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        marker_size: float = 20.0,
        alpha: float = 0.7,
        lat_coord: str = "latitude",
        lon_coord: str = "longitude",
        alt_coord: str = "altitude",
        alt_scale: float = 0.001,
        elev: float = 25,
        azim: float = -60,
        show_projection: bool = True,
        projection_alpha: float = 0.3,
        show_coastlines: bool = True,
        coastline_color: str = "black",
        coastline_alpha: float = 1.0,
        coastline_linewidth: float = 0.8,
        coastline_scale: str = "10m",
        show_borders: bool = False,
        border_color: str = "gray",
        border_alpha: float = 0.5,
        border_linewidth: float = 0.5,
        show_surface_map: bool = False,
        surface_map_resolution: int = 250,
        land_color: str = "#E8E8E8",
        ocean_color: str = "#D4E9F7",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a 3D flight track map.

        Parameters
        ----------
        obs_data
            Observation dataset with lat/lon/alt coordinates and the variable.
        variable
            Name of the variable to color by.
        ax
            Ignored (always creates new 3D axes).
        title
            Plot title. Defaults to "{variable} Flight Track".
        cmap
            Colormap name. Defaults to the sequential NCAR colormap.
        vmin, vmax
            Colorbar limits. If None, auto-determined from data.
        marker_size
            Size of scatter markers.
        alpha
            Transparency of markers.
        lat_coord, lon_coord
            Names of latitude/longitude coordinates.
        alt_coord
            Name of the altitude coordinate.
        alt_scale
            Scale factor for altitude (e.g., 0.001 to convert m to km).
        elev
            Elevation angle for 3D view.
        azim
            Azimuth angle for 3D view.
        show_projection
            If True, show 2D projection on the bottom (z=0) plane.
        projection_alpha
            Transparency of projection markers.
        show_coastlines
            If True, draw continent outlines on the surface plane.
        coastline_color
            Color for coastline lines.
        coastline_alpha
            Transparency of coastlines.
        coastline_linewidth
            Line width for coastlines.
        coastline_scale
            Natural Earth scale for coastlines ('110m', '50m', or '10m').
        show_borders
            If True, draw country borders on surface plane.
        border_color
            Color for border lines.
        border_alpha
            Transparency of borders.
        border_linewidth
            Line width for borders.
        show_surface_map
            If True, render a filled map image on the z=0 plane.
        surface_map_resolution
            Resolution of surface map image in pixels.
        land_color
            Color for land areas on surface map.
        ocean_color
            Color for ocean areas on surface map.
        **kwargs
            Additional arguments passed to ax.scatter.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Extract coordinates
        lats = obs_data[lat_coord].values
        lons = obs_data[lon_coord].values
        values = obs_data[variable].values

        # Get altitude
        if alt_coord in obs_data.coords:
            alts = obs_data[alt_coord].values * alt_scale
        elif alt_coord in obs_data.data_vars:
            alts = obs_data[alt_coord].values * alt_scale
        else:
            raise ValueError(
                f"Altitude coordinate '{alt_coord}' not found. "
                f"Available: {list(obs_data.coords) + list(obs_data.data_vars)}"
            )

        # Filter valid data
        valid = np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons) & np.isfinite(alts)
        lats = lats[valid]
        lons = lons[valid]
        alts = alts[valid]
        values = values[valid]

        if len(values) == 0:
            raise ValueError("No valid data points for 3D track plot")

        # Determine colorbar limits
        if vmin is None:
            vmin = float(np.nanmin(values))
        if vmax is None:
            vmax = float(np.nanmax(values))

        # Create figure with 3D axes
        fig = plt.figure(
            figsize=self.config.figure.figsize,
            dpi=self.config.figure.dpi,
        )
        ax3d = fig.add_subplot(111, projection="3d")

        text_cfg = self.config.text
        cmap = cmap or get_sequential_cmap()

        # Plot 3D scatter
        scatter = ax3d.scatter(  # type: ignore[misc]
            lons,
            lats,
            alts,
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
            ax3d.scatter(  # type: ignore[misc]
                lons,
                lats,
                np.zeros_like(alts),
                c=values,
                cmap=cmap,
                s=marker_size * 0.3,
                alpha=projection_alpha,
                vmin=vmin,
                vmax=vmax,
            )

        # Calculate bounds for map features
        lon_min, lon_max = float(np.nanmin(lons)), float(np.nanmax(lons))
        lat_min, lat_max = float(np.nanmin(lats)), float(np.nanmax(lats))
        lon_pad = (lon_max - lon_min) * 0.1
        lat_pad = (lat_max - lat_min) * 0.1
        lon_min -= lon_pad
        lon_max += lon_pad
        lat_min -= lat_pad
        lat_max += lat_pad

        # Draw surface map or coastlines
        if show_surface_map:
            map_img = _render_surface_map(
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                resolution=surface_map_resolution,
                land_color=land_color,
                ocean_color=ocean_color,
                coastline_color=coastline_color,
                coastline_linewidth=coastline_linewidth,
                show_borders=show_borders,
                border_color=border_color,
                border_linewidth=border_linewidth,
            )
            if map_img is not None:
                img_h, img_w = map_img.shape[:2]
                lon_grid = np.linspace(lon_min, lon_max, img_w)
                lat_grid = np.linspace(lat_max, lat_min, img_h)
                X, Y = np.meshgrid(lon_grid, lat_grid)
                Z = np.zeros_like(X)
                ax3d.plot_surface(  # type: ignore[attr-defined]
                    X,
                    Y,
                    Z,
                    facecolors=map_img,
                    rstride=1,
                    cstride=1,
                    shade=False,
                    zorder=1,
                )
        elif show_coastlines:
            coastline_segments = _get_coastline_segments(
                lon_min, lon_max, lat_min, lat_max, scale=coastline_scale
            )
            for seg_lons, seg_lats in coastline_segments:
                ax3d.plot(
                    seg_lons,
                    seg_lats,
                    np.zeros(len(seg_lons)),
                    color=coastline_color,
                    alpha=coastline_alpha,
                    linewidth=coastline_linewidth,
                    zorder=2,
                )

        # Draw country borders on surface plane (only if not using surface map)
        if show_borders and not show_surface_map:
            border_segments = _get_border_segments(lon_min, lon_max, lat_min, lat_max)
            for seg_lons, seg_lats in border_segments:
                ax3d.plot(
                    seg_lons,
                    seg_lats,
                    np.zeros(len(seg_lons)),
                    color=border_color,
                    alpha=border_alpha,
                    linewidth=border_linewidth,
                    linestyle="--",
                    zorder=2,
                )

        # Set axis limits
        ax3d.set_xlim(lon_min, lon_max)
        ax3d.set_ylim(lat_min, lat_max)

        # Set view angle
        ax3d.view_init(elev=elev, azim=azim)  # type: ignore[attr-defined]

        # Limit tick count to prevent overlap, then format labels
        ax3d.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax3d.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
        ax3d.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax3d.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

        # Axis labels
        ax3d.set_xlabel("Longitude (\u00b0E)", fontsize=text_cfg.fontsize, labelpad=8)
        ax3d.set_ylabel("Latitude (\u00b0N)", fontsize=text_cfg.fontsize, labelpad=8)
        ax3d.set_zlabel("Altitude (km)", fontsize=text_cfg.fontsize, labelpad=8)  # type: ignore[attr-defined]
        ax3d.tick_params(axis="both", labelsize=text_cfg.tick_fontsize)

        # Colorbar
        var_label = get_variable_label(obs_data, variable, include_prefix=False)
        units = get_variable_units(obs_data, variable)
        cbar_label = format_label_with_units(var_label, units)
        cbar_label = format_plot_title(cbar_label)
        cbar = fig.colorbar(scatter, ax=ax3d, shrink=0.6, pad=0.1)
        cbar.set_label(cbar_label, fontsize=text_cfg.fontsize)
        cbar.ax.tick_params(labelsize=text_cfg.tick_fontsize)

        # Title
        if title is None:
            title = f"{var_label} Flight Track"
        else:
            title = format_plot_title(title)
        fig.suptitle(title, fontsize=text_cfg.title_fontsize, y=0.85)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        return fig


# ``obs_flight_track`` is a deprecated alias of the unified renderer.
register_alias("obs_flight_track", "flight_track")
