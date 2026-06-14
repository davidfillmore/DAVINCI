"""3D track map renderer for DAVINCI.

This module provides 3D visualization of aircraft flight tracks,
showing longitude, latitude, and altitude with color-coded values.
Includes continent outlines and city markers on the surface plane.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    calculate_symmetric_limits,
    format_label_with_units,
    format_plot_title,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter

# Shared 3D-track drawing mechanics live in ``_track3d``.  The geometry helpers
# are re-exported here so existing imports
# (``from ...renderers.track_map_3d import _get_coastline_segments`` etc.)
# keep working.
from davinci_monet.plots.renderers._track3d import (
    _get_border_segments,
    _get_coastline_segments,
    _get_land_polygons,
    _render_surface_map,
    draw_track_3d,
)
from davinci_monet.plots.titles import title_for_labeled_subset

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

__all__ = [
    "TrackMap3DPlotter",
    "plot_track_map_3d",
    "_get_coastline_segments",
    "_get_land_polygons",
    "_get_border_segments",
    "_render_surface_map",
]


@register_plotter("track_map_3d")
class TrackMap3DPlotter(BasePlotter):
    """Plotter for 3D flight track visualization.

    Creates 3D plots showing aircraft trajectory with:
    - X-axis: Longitude
    - Y-axis: Latitude
    - Z-axis: Altitude
    - Color: Variable value (geometry, dataset, or bias)

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TrackMap3DPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     x_var="geometry_O3",
    ...     y_var="dataset_O3",
    ...     show_var="bias",
    ... )
    """

    name: str = "track_map_3d"
    default_figsize: tuple[float, float] = (7, 6)  # Near-square for 3D viewing

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a 3D track map from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one geometry (geometry) and one dataset (dataset).
        ax
            Ignored (creates new 3D axes).
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            alt_var (str, default "altitude"),
            lat_var (str, default "latitude"),
            lon_var (str, default "longitude"),
            show_var (str, default "bias"),
            cmap (str|None, default None),
            marker_size (float, default 20),
            alpha (float, default 0.7),
            elev (float, default 25),
            azim (float, default -60),
            show_projection (bool, default True),
            projection_alpha (float, default 0.3),
            alt_scale (float, default 0.001),
            show_coastlines (bool, default True),
            coastline_color (str, default "black"),
            coastline_alpha (float, default 1.0),
            coastline_linewidth (float, default 0.8),
            coastline_scale (str, default "10m"),
            show_borders (bool, default False),
            border_color (str, default "gray"),
            border_alpha (float, default 0.5),
            border_linewidth (float, default 0.5),
            city_labels (dict|None, default None),
            city_marker_size (float, default 50),
            city_marker_color (str, default "red"),
            city_font_size (float, default 10),
            show_surface_map (bool, default False),
            surface_map_resolution (int, default 250),
            land_color (str, default "#E8E8E8"),
            ocean_color (str, default "#D4E9F7").

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"TrackMap3DPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        alt_var: str = kwargs.pop("alt_var", "altitude")
        lat_var: str = kwargs.pop("lat_var", "latitude")
        lon_var: str = kwargs.pop("lon_var", "longitude")
        show_var: Literal["geometry", "dataset", "bias"] = kwargs.pop("show_var", "bias")
        cmap: str | None = kwargs.pop("cmap", None)
        marker_size: float = kwargs.pop("marker_size", 20)
        alpha: float = kwargs.pop("alpha", 0.7)
        elev: float = kwargs.pop("elev", 25)
        azim: float = kwargs.pop("azim", -60)
        show_projection: bool = kwargs.pop("show_projection", True)
        projection_alpha: float = kwargs.pop("projection_alpha", 0.3)
        alt_scale: float = kwargs.pop("alt_scale", 0.001)
        show_coastlines: bool = kwargs.pop("show_coastlines", True)
        coastline_color: str = kwargs.pop("coastline_color", "black")
        coastline_alpha: float = kwargs.pop("coastline_alpha", 1.0)
        coastline_linewidth: float = kwargs.pop("coastline_linewidth", 0.8)
        coastline_scale: str = kwargs.pop("coastline_scale", "10m")
        show_borders: bool = kwargs.pop("show_borders", False)
        border_color: str = kwargs.pop("border_color", "gray")
        border_alpha: float = kwargs.pop("border_alpha", 0.5)
        border_linewidth: float = kwargs.pop("border_linewidth", 0.5)
        city_labels: dict[str, list[float]] | None = kwargs.pop("city_labels", None)
        city_marker_size: float = kwargs.pop("city_marker_size", 50)
        city_marker_color: str = kwargs.pop("city_marker_color", "red")
        city_font_size: float = kwargs.pop("city_font_size", 10)
        show_surface_map: bool = kwargs.pop("show_surface_map", False)
        surface_map_resolution: int = kwargs.pop("surface_map_resolution", 250)
        land_color: str = kwargs.pop("land_color", "#E8E8E8")
        ocean_color: str = kwargs.pop("ocean_color", "#D4E9F7")

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
        geometry_vals = paired_data[x_var].values
        dataset_vals = paired_data[y_var].values

        # Calculate what to show
        if show_var == "geometry":
            values = geometry_vals
            default_cmap = "viridis"
            label = get_variable_label(paired_data, x_var, include_prefix=False)
        elif show_var == "dataset":
            values = dataset_vals
            default_cmap = "viridis"
            label = get_variable_label(paired_data, y_var, include_prefix=False)
        else:  # bias
            values = dataset_vals - geometry_vals
            default_cmap = "RdBu_r"
            # Consistent bias label with other plotters
            label = "Bias (Dataset - Geometry)"

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

        # Text settings - use absolute point sizes from config
        text_cfg = self.config.text

        # Set color limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(values)
        else:
            vmin = self.config.vmin if self.config.vmin is not None else float(np.nanmin(values))
            vmax = self.config.vmax if self.config.vmax is not None else float(np.nanmax(values))

        # Colorbar label
        units = get_variable_units(paired_data, x_var)
        cbar_label = format_label_with_units(label, units)
        cbar_label = format_plot_title(cbar_label)  # Apply subscript formatting

        # Draw the shared 3D track body (scatter, projection, surface-plane map
        # features, city markers, axis setup, labels, colorbar).
        draw_track_3d(
            fig,
            ax3d,
            lons,
            lats,
            alts,
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            text_cfg=text_cfg,
            cbar_label=cbar_label,
            marker_size=marker_size,
            alpha=alpha,
            elev=elev,
            azim=azim,
            show_projection=show_projection,
            projection_alpha=projection_alpha,
            show_coastlines=show_coastlines,
            coastline_color=coastline_color,
            coastline_alpha=coastline_alpha,
            coastline_linewidth=coastline_linewidth,
            coastline_scale=coastline_scale,
            show_borders=show_borders,
            border_color=border_color,
            border_alpha=border_alpha,
            border_linewidth=border_linewidth,
            show_surface_map=show_surface_map,
            surface_map_resolution=surface_map_resolution,
            land_color=land_color,
            ocean_color=ocean_color,
            city_labels=city_labels,
            city_marker_size=city_marker_size,
            city_marker_color=city_marker_color,
            city_font_size=city_font_size,
        )

        # Title - centered above plot
        if self.config.title:
            self.set_figure_title(fig, self.config.title, y=0.85)

        plt.tight_layout(rect=(0, 0, 1, 0.95))  # Leave room at top for title
        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        alt_var: str = "altitude",
        lat_var: str = "latitude",
        lon_var: str = "longitude",
        show_var: Literal["geometry", "dataset", "bias"] = "bias",
        cmap: str | None = None,
        marker_size: float = 20,
        alpha: float = 0.7,
        elev: float = 25,
        azim: float = -60,
        show_projection: bool = True,
        projection_alpha: float = 0.3,
        alt_scale: float = 0.001,
        show_coastlines: bool = True,
        coastline_color: str = "black",
        coastline_alpha: float = 1.0,
        coastline_linewidth: float = 0.8,
        coastline_scale: str = "10m",
        show_borders: bool = False,
        border_color: str = "gray",
        border_alpha: float = 0.5,
        border_linewidth: float = 0.5,
        city_labels: dict[str, list[float]] | None = None,
        city_marker_size: float = 50,
        city_marker_color: str = "red",
        city_font_size: float = 10,  # City label font size
        show_surface_map: bool = False,
        surface_map_resolution: int = 250,
        land_color: str = "#E8E8E8",
        ocean_color: str = "#D4E9F7",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a 3D track map.

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
            Existing axes (ignored, creates new 3D axes).
        alt_var
            Name of altitude coordinate.
        lat_var
            Name of latitude coordinate.
        lon_var
            Name of longitude coordinate.
        show_var
            Which variable to show: 'geometry', 'dataset', or 'bias'.
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
        show_coastlines
            If True, draw continent outlines on the surface plane.
        coastline_color
            Color for coastline lines.
        coastline_alpha
            Transparency of coastlines.
        coastline_linewidth
            Line width for coastlines.
        show_borders
            If True, draw country borders on surface plane.
        border_color
            Color for border lines.
        border_alpha
            Transparency of borders.
        border_linewidth
            Line width for borders.
        city_labels
            Dictionary of city names to [lat, lon] coordinates.
            Cities will be plotted as markers on the surface plane.
        city_marker_size
            Size of city markers.
        city_marker_color
            Color for city markers.
        city_font_size
            Font size for city labels.
        show_surface_map
            If True, render a filled map image on the z=0 plane using cartopy.
            Shows land and ocean colors. Overrides show_coastlines when True.
        surface_map_resolution
            Resolution of surface map image in pixels (default: 250).
        land_color
            Color for land areas on surface map.
        ocean_color
            Color for ocean areas on surface map.
        **kwargs
            Additional options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, x_var, y_var),
            ax=ax,
            alt_var=alt_var,
            lat_var=lat_var,
            lon_var=lon_var,
            show_var=show_var,
            cmap=cmap,
            marker_size=marker_size,
            alpha=alpha,
            elev=elev,
            azim=azim,
            show_projection=show_projection,
            projection_alpha=projection_alpha,
            alt_scale=alt_scale,
            show_coastlines=show_coastlines,
            coastline_color=coastline_color,
            coastline_alpha=coastline_alpha,
            coastline_linewidth=coastline_linewidth,
            coastline_scale=coastline_scale,
            show_borders=show_borders,
            border_color=border_color,
            border_alpha=border_alpha,
            border_linewidth=border_linewidth,
            city_labels=city_labels,
            city_marker_size=city_marker_size,
            city_marker_color=city_marker_color,
            city_font_size=city_font_size,
            show_surface_map=show_surface_map,
            surface_map_resolution=surface_map_resolution,
            land_color=land_color,
            ocean_color=ocean_color,
            **kwargs,
        )

    def plot_per_flight(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        flight_coord: str = "flight",
        min_points: int = 10,
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate 3D track maps for each flight.

        Yields one figure per unique flight in the data.

        Parameters
        ----------
        paired_data
            Paired dataset with dataset and dataset variables.
        x_var
            Name of dataset variable.
        y_var
            Name of dataset variable.
        flight_coord
            Name of the flight coordinate (default: "flight").
        min_points
            Minimum valid data points per flight to generate a plot.
        **kwargs
            Additional arguments passed to plot method.

        Yields
        ------
        tuple[str, matplotlib.figure.Figure]
            Tuple of (flight_id, figure) for each flight.
        """
        # Check for flight coordinate
        if flight_coord not in paired_data.coords:
            raise ValueError(
                f"Flight coordinate '{flight_coord}' not found in paired data. "
                f"Available coordinates: {list(paired_data.coords)}"
            )

        # Get unique flights
        flight_values = paired_data[flight_coord].values
        unique_flights = np.unique(flight_values)

        for flight in unique_flights:
            # Convert to string
            flight_str = str(flight)

            # Filter data for this flight
            mask = flight_values == flight
            flight_data = paired_data.isel(time=mask)

            # Check for minimum points
            geometry_vals = flight_data[x_var].values.flatten()
            dataset_vals = flight_data[y_var].values.flatten()
            valid = np.isfinite(geometry_vals) & np.isfinite(dataset_vals)

            if valid.sum() < min_points:
                continue

            # Update title/subtitle to identify this flight.
            original_title = self.config.title
            original_subtitle = self.config.subtitle
            self.config.title, flight_subtitle = title_for_labeled_subset(
                original_title,
                flight_str,
                label_prefix="Flight",
            )
            if flight_subtitle:
                self.config.subtitle = flight_subtitle

            # Generate plot for this flight
            try:
                fig = self.plot(flight_data, x_var, y_var, **kwargs)
            except ValueError:
                # Skip if no valid data after filtering
                self.config.title = original_title
                self.config.subtitle = original_subtitle
                continue

            # Restore original title/subtitle for next iteration
            self.config.title = original_title
            self.config.subtitle = original_subtitle

            # Format flight ID for filename (YYYYMMDD format)
            flight_id = flight_str.replace("-", "")

            yield flight_id, fig


def plot_track_map_3d(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    title: str | None = None,
    show_var: Literal["geometry", "dataset", "bias"] = "bias",
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for 3D track map plots.

    Parameters
    ----------
    paired_data
        Paired dataset with dataset and dataset variables.
    x_var
        Name of dataset variable.
    y_var
        Name of dataset variable.
    title
        Plot title.
    show_var
        Which variable to show: 'geometry', 'dataset', or 'bias'.
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
        x_var,
        y_var,
        show_var=show_var,
        **kwargs,
    )
