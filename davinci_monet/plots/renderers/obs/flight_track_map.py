"""Flight track map renderer for DAVINCI-MONET.

Renders a Cartopy map showing an aircraft flight path colored by a variable
value (e.g., O3 concentration). Supports single or multi-flight datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import get_sequential_cmap

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_flight_track")
class FlightTrackMapPlotter(ObsPlotter):
    """Plotter for flight track maps colored by variable value.

    Creates a Cartopy map with the flight path rendered as scatter points
    colored by the selected variable. Auto-zooms to data extent with padding.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = FlightTrackMapPlotter()
    >>> fig = plotter.plot(obs_data, "O3", title="DC3 Flight O3")
    """

    name: str = "obs_flight_track"
    default_figsize: tuple[float, float] = (10, 7)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        title: str | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        marker_size: float = 15.0,
        lat_coord: str = "latitude",
        lon_coord: str = "longitude",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a flight track map.

        Parameters
        ----------
        obs_data
            Observation dataset with lat/lon coordinates and the variable.
        variable
            Name of the variable to color by.
        ax
            Optional GeoAxes to plot on. If None, creates new figure.
        title
            Plot title. Defaults to "{variable} Flight Track".
        cmap
            Colormap name. Defaults to the sequential NCAR colormap.
        vmin, vmax
            Colorbar limits. If None, auto-determined from data.
        marker_size
            Size of scatter points.
        lat_coord, lon_coord
            Names of latitude/longitude coordinates.
        **kwargs
            Additional arguments passed to ax.scatter.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        projection = ccrs.PlateCarree()

        if ax is None:
            fig = plt.figure(
                figsize=self.config.figure.figsize,
                dpi=self.config.figure.dpi,
                facecolor=self.config.figure.facecolor,
            )
            ax = fig.add_subplot(1, 1, 1, projection=projection)
        else:
            fig = ax.get_figure()

        # Extract data
        lats = obs_data[lat_coord].values
        lons = obs_data[lon_coord].values
        values = obs_data[variable].values

        # Remove NaNs for clean plotting
        valid = np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons)
        lats = lats[valid]
        lons = lons[valid]
        values = values[valid]

        # Determine colorbar limits
        if vmin is None:
            vmin = float(np.nanmin(values))
        if vmax is None:
            vmax = float(np.nanmax(values))

        # Plot scatter
        cmap = cmap or get_sequential_cmap()
        sc = ax.scatter(
            lons,
            lats,
            c=values,
            s=marker_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=projection,
            zorder=5,
            **kwargs,
        )

        # Auto-zoom with padding
        lon_range = lons.max() - lons.min()
        lat_range = lats.max() - lats.min()
        pad_lon = max(lon_range * 0.1, 0.5)
        pad_lat = max(lat_range * 0.1, 0.5)
        ax.set_extent(
            [
                lons.min() - pad_lon,
                lons.max() + pad_lon,
                lats.min() - pad_lat,
                lats.max() + pad_lat,
            ],
            crs=projection,
        )

        # Map features
        ax.add_feature(cfeature.LAND, facecolor="#F1F0EE", zorder=0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.3, zorder=2)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Colorbar
        units = obs_data[variable].attrs.get("units", "")
        label = variable
        if units:
            label = f"{variable} ({units})"
        fig.colorbar(sc, ax=ax, label=label, shrink=0.8, pad=0.02)

        # Title
        if title is None:
            title = f"{variable} Flight Track"
        ax.set_title(title, fontsize=self.config.text.title_fontsize)

        return fig
