"""3D track map renderer for DAVINCI-MONET.

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


def _get_coastline_segments(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    scale: str = "10m",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract coastline segments from Natural Earth data within bounds.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.
    scale
        Natural Earth scale: '110m', '50m', or '10m'.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (lons, lats) arrays for each coastline segment.
    """
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        return []

    segments = []
    coastline = cfeature.NaturalEarthFeature(
        category="physical",
        name="coastline",
        scale=scale,
        facecolor="none",
    )
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    for geom in coastline.geometries():
        # Clip to bounding box
        try:
            clipped = geom.intersection(bbox)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Handle different geometry types
        if hasattr(clipped, "geoms"):
            # MultiLineString or GeometryCollection
            geoms = list(clipped.geoms)
        else:
            geoms = [clipped]

        for g in geoms:
            if hasattr(g, "coords"):
                coords = np.array(g.coords)
                if len(coords) > 1:
                    segments.append((coords[:, 0], coords[:, 1]))

    return segments


def _get_land_polygons(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float,
    scale: str = "50m",
) -> list[np.ndarray]:
    """Extract land polygon vertices from Natural Earth data within bounds.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.
    scale
        Natural Earth scale: '110m', '50m', or '10m'.

    Returns
    -------
    list[np.ndarray]
        List of polygon vertex arrays, each with shape (N, 2) for lon, lat.
    """
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        return []

    polygons = []
    # Use specified resolution
    land = cfeature.NaturalEarthFeature(
        category="physical",
        name="land",
        scale=scale,
        facecolor="none",
    )
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    for geom in land.geometries():
        # Clip to bounding box
        try:
            clipped = geom.intersection(bbox)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        # Handle different geometry types
        if hasattr(clipped, "geoms"):
            geoms = list(clipped.geoms)
        else:
            geoms = [clipped]

        for g in geoms:
            # Get exterior ring of polygon
            if hasattr(g, "exterior"):
                coords = np.array(g.exterior.coords)
                if len(coords) >= 3:
                    polygons.append(coords[:, :2])  # lon, lat only

    return polygons


def _get_border_segments(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract country border segments from Natural Earth data within bounds.

    Parameters
    ----------
    lon_min, lon_max
        Longitude bounds.
    lat_min, lat_max
        Latitude bounds.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (lons, lats) arrays for each border segment.
    """
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError:
        return []

    segments = []
    borders = cfeature.BORDERS
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    for geom in borders.geometries():
        try:
            clipped = geom.intersection(bbox)
        except Exception:
            continue

        if clipped.is_empty:
            continue

        if hasattr(clipped, "geoms"):
            geoms = list(clipped.geoms)
        else:
            geoms = [clipped]

        for g in geoms:
            if hasattr(g, "coords"):
                coords = np.array(g.coords)
                if len(coords) > 1:
                    segments.append((coords[:, 0], coords[:, 1]))

    return segments


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
        city_font_size: float = 8,
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
            # Consistent bias label with other plotters
            label = "Bias (Model - Obs)"

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

        # Calculate bounds for map features and set axis limits
        lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
        lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
        # Add padding
        lon_pad = (lon_max - lon_min) * 0.1
        lat_pad = (lat_max - lat_min) * 0.1
        lon_min -= lon_pad
        lon_max += lon_pad
        lat_min -= lat_pad
        lat_max += lat_pad

        # Draw coastlines on surface plane (z=0)
        if show_coastlines:
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

        # Draw country borders on surface plane (z=0)
        if show_borders:
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

        # Draw city markers and labels on surface plane
        if city_labels:
            for city_name, coords in city_labels.items():
                city_lat, city_lon = coords[0], coords[1]
                # Only plot if within bounds
                if lon_min <= city_lon <= lon_max and lat_min <= city_lat <= lat_max:
                    # Plot marker
                    ax3d.scatter(
                        [city_lon], [city_lat], [0],
                        s=city_marker_size,
                        c=city_marker_color,
                        marker="^",
                        alpha=0.9,
                        zorder=10,
                    )
                    # Add label
                    ax3d.text(
                        city_lon, city_lat, 0,
                        f"  {city_name}",
                        fontsize=city_font_size,
                        color="black",
                        ha="left",
                        va="bottom",
                        zorder=11,
                    )

        # Set axis limits to include padding
        ax3d.set_xlim(lon_min, lon_max)
        ax3d.set_ylim(lat_min, lat_max)

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
            ax3d.set_title(format_plot_title(self.config.title), fontsize=12)

        plt.tight_layout()
        return fig

    def plot_per_flight(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        flight_coord: str = "flight",
        min_points: int = 10,
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate 3D track maps for each flight.

        Yields one figure per unique flight in the data.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
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
            obs_vals = flight_data[obs_var].values.flatten()
            model_vals = flight_data[model_var].values.flatten()
            valid = np.isfinite(obs_vals) & np.isfinite(model_vals)

            if valid.sum() < min_points:
                continue

            # Update title to include flight ID
            original_title = self.config.title
            if original_title:
                self.config.title = f"{original_title} - Flight {flight_str}"
            else:
                self.config.title = f"Flight {flight_str}"

            # Generate plot for this flight
            try:
                fig = self.plot(flight_data, obs_var, model_var, **kwargs)
            except ValueError:
                # Skip if no valid data after filtering
                self.config.title = original_title
                continue

            # Restore original title for next iteration
            self.config.title = original_title

            # Format flight ID for filename (YYYYMMDD format)
            flight_id = flight_str.replace("-", "")

            yield flight_id, fig


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
