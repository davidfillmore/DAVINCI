"""Observation-only time series renderer for DAVINCI-MONET.

Renders variable vs. time for raw observation data. Supports per-flight
coloring and optional altitude overlay on a secondary y-axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from davinci_monet.plots.base import format_plot_title, get_variable_label
from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_PALETTE, NCAR_PRIMARY

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_timeseries")
class ObsTimeSeriesPlotter(ObsPlotter):
    """Plotter for observation-only time series.

    If the dataset contains a ``flight`` coordinate, each flight is plotted
    in a different color from ``NCAR_PALETTE``. Otherwise a single line
    in ``OBS_COLOR`` is drawn.

    Optionally overlays altitude on a secondary y-axis.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = ObsTimeSeriesPlotter()
    >>> fig = plotter.plot(obs_data, "O3", show_altitude=True)
    """

    name: str = "obs_timeseries"
    default_figsize: tuple[float, float] = (10, 5)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        show_altitude: bool = False,
        alt_coord: str = "altitude",
        title: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate an observation time series plot.

        Parameters
        ----------
        obs_data
            Observation dataset with a ``time`` dimension.
        variable
            Name of the variable to plot.
        ax
            Optional axes to plot on. If None, creates new figure.
        show_altitude
            If True, overlay altitude on a secondary y-axis.
        alt_coord
            Name of the altitude coordinate (used when show_altitude is True).
        title
            Plot title. Defaults to "{variable} Time Series".
        color
            Line color for single-flight data. Defaults to OBS_COLOR.
        **kwargs
            Additional arguments passed to the underlying plot call.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        color = color or NCAR_PRIMARY
        time_values = pd.to_datetime(obs_data["time"].values)
        values = obs_data[variable].values

        has_flights = (
            "flight" in obs_data.coords
            and len(np.unique(obs_data["flight"].values)) > 1
        )

        if has_flights:
            flight_ids = np.unique(obs_data["flight"].values)
            for i, fid in enumerate(flight_ids):
                c = NCAR_PALETTE[i % len(NCAR_PALETTE)]
                mask = obs_data["flight"].values == fid
                ax.plot(
                    time_values[mask],
                    values[mask],
                    color=c,
                    linewidth=1.2,
                    label=str(fid),
                    **kwargs,
                )
            ax.legend(fontsize=self.config.text.legend)
        else:
            ax.plot(
                time_values,
                values,
                color=color,
                linewidth=1.2,
                **kwargs,
            )

        # Labels
        var_label = get_variable_label(obs_data, variable, include_prefix=False)
        units = obs_data[variable].attrs.get("units", "")
        ylabel = f"{var_label} ({units})" if units else var_label
        ax.set_ylabel(ylabel, fontsize=self.config.text.fontsize)
        ax.set_xlabel("Time", fontsize=self.config.text.fontsize)

        # Title
        if title is None:
            title = f"{var_label} Time Series"
        else:
            title = format_plot_title(title)
        ax.set_title(title, fontsize=self.config.text.title_fontsize)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        # Altitude overlay on secondary y-axis
        if show_altitude and alt_coord in obs_data.coords:
            ax2 = ax.twinx()
            alt_values = obs_data[alt_coord].values
            ax2.plot(
                time_values,
                alt_values,
                color="#AAAAAA",
                linewidth=0.8,
                alpha=0.6,
            )
            alt_units = obs_data[alt_coord].attrs.get("units", "")
            alt_label = "Altitude"
            if alt_units:
                alt_label = f"Altitude ({alt_units})"
            ax2.set_ylabel(alt_label, fontsize=self.config.text.fontsize, color="#AAAAAA")
            ax2.tick_params(axis="y", labelcolor="#AAAAAA")

        return fig
