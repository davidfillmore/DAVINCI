"""Flight-by-flight time series plot renderer for DAVINCI-MONET.

This module provides multi-panel time series plots showing model vs observations
for individual aircraft flights. Designed for track observations where each
flight (identified by date) is plotted in a separate panel.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
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


@register_plotter("flight_timeseries")
class FlightTimeSeriesPlotter(BasePlotter):
    """Plotter for flight-by-flight time series comparisons.

    Creates a multi-panel figure with one subplot per flight,
    showing both model and observation time series along the flight track.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = FlightTimeSeriesPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_O3_ROZE_STCLAIR",
    ...     model_var="model_O3_ROZE_STCLAIR",
    ...     ncols=3,
    ... )
    """

    name: str = "flight_timeseries"
    default_figsize: tuple[float, float] = (14, 6)  # Wide for temporal data

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        ncols: int = 3,
        min_points: int = 10,
        time_dim: str = "time",
        flight_coord: str = "flight",
        show_stats: bool = True,
        scale_factor: float = 1.0,
        obs_style: str = "scatter",
        model_style: str = "line",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate flight-by-flight time series panels.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Ignored for this plot type (creates own figure).
        ncols
            Number of columns in subplot grid.
        min_points
            Minimum valid data points required to include a flight.
        time_dim
            Name of time dimension.
        flight_coord
            Name of flight coordinate (default: 'flight').
        show_stats
            If True, show N, NMB, R statistics on each panel.
        scale_factor
            Scale factor for display values.
        obs_style
            Style for observations: 'scatter' or 'line'.
        model_style
            Style for model: 'line' or 'scatter'.
        **kwargs
            Additional options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        style = self.config.style

        # Get unique flights and filter by data availability
        if flight_coord not in paired_data.coords:
            raise ValueError(f"Flight coordinate '{flight_coord}' not found in dataset")

        flights = np.unique(paired_data[flight_coord].values)
        valid_flights = []

        for flight in flights:
            mask = paired_data[flight_coord].values == flight
            obs_vals = paired_data[obs_var].values[mask]
            mod_vals = paired_data[model_var].values[mask]
            valid = ~np.isnan(obs_vals) & ~np.isnan(mod_vals)
            if valid.sum() >= min_points:
                valid_flights.append(flight)

        if not valid_flights:
            raise ValueError(f"No flights with >= {min_points} valid data points")

        n_flights = len(valid_flights)
        nrows = (n_flights + ncols - 1) // ncols

        # Create figure with standard size
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(8, 5),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Plot each flight
        for idx, flight in enumerate(valid_flights):
            ax = axes_flat[idx]
            mask = paired_data[flight_coord].values == flight

            # Get data for this flight
            times = pd.to_datetime(paired_data[time_dim].values[mask])
            obs_vals = paired_data[obs_var].values[mask] * scale_factor
            mod_vals = paired_data[model_var].values[mask] * scale_factor

            valid_obs = ~np.isnan(obs_vals)
            valid_both = valid_obs & ~np.isnan(mod_vals)

            # Sort by time for line plots
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            obs_vals = obs_vals[sort_idx]
            mod_vals = mod_vals[sort_idx]
            valid_obs = valid_obs[sort_idx]
            valid_both = valid_both[sort_idx]

            # Plot observations
            if obs_style == "scatter":
                ax.scatter(
                    times[valid_obs], obs_vals[valid_obs],
                    s=12, alpha=0.7, color="black", label="Obs", zorder=3
                )
            else:
                ax.plot(
                    times[valid_obs], obs_vals[valid_obs],
                    "o-", color="black", markersize=3, linewidth=0.5,
                    alpha=0.7, label="Obs", zorder=3
                )

            # Plot model
            if model_style == "line":
                ax.plot(
                    times, mod_vals,
                    color=style.model_color, linewidth=1.5, alpha=0.8,
                    label="Model", zorder=2
                )
            else:
                ax.scatter(
                    times[valid_both], mod_vals[valid_both],
                    s=12, alpha=0.7, color=style.model_color,
                    label="Model", zorder=2
                )

            # Compute and display stats
            if show_stats and valid_both.sum() > 0:
                n = valid_both.sum()
                obs_mean = obs_vals[valid_both].mean()
                mod_mean = mod_vals[valid_both].mean()
                nmb = 100 * (mod_mean - obs_mean) / obs_mean if obs_mean != 0 else 0
                if valid_both.sum() > 2:
                    r = np.corrcoef(obs_vals[valid_both], mod_vals[valid_both])[0, 1]
                else:
                    r = np.nan

                stats_text = f"N={n}\nNMB={nmb:+.0f}%"
                if not np.isnan(r):
                    stats_text += f"\nR={r:.2f}"

                # Multi-panel font sizes (reduced from config for small panels)
                panel_fontsize = self.config.text.fontsize - 6
                ax.text(
                    0.97, 0.97, stats_text,
                    transform=ax.transAxes, fontsize=panel_fontsize,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )

            # Title with flight date
            panel_title_fontsize = self.config.text.title_fontsize - 6
            ax.set_title(f"Flight {flight}", fontsize=panel_title_fontsize)

            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)

            # Legend on first panel only
            if idx == 0:
                ax.legend(loc="upper left", fontsize=self.config.text.fontsize - 6)

            # Y-axis label on left column
            if idx % ncols == 0:
                units = get_variable_units(paired_data, obs_var)
                ylabel = get_variable_label(paired_data, obs_var, include_prefix=False)
                ylabel = format_label_with_units(ylabel, units)
                ax.set_ylabel(ylabel, fontsize=self.config.text.fontsize - 5)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.tick_params(axis="x", rotation=45)
            ax.set_xlabel("UTC Time", fontsize=self.config.text.fontsize - 5)

        # Hide unused subplots
        for idx in range(n_flights, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Main title
        if self.config.title:
            fig.suptitle(format_plot_title(self.config.title), fontsize=self.config.text.fontsize, y=1.02)

        plt.tight_layout()
        return fig

    def plot_per_flight(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        time_dim: str = "time",
        flight_coord: str = "flight",
        min_points: int = 10,
        show_stats: bool = True,
        scale_factor: float = 1.0,
        obs_style: str = "scatter",
        model_style: str = "line",
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate individual time series plots for each flight.

        Yields one figure per unique flight in the data, each showing
        a single-panel time series comparison.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        time_dim
            Name of time dimension.
        flight_coord
            Name of flight coordinate (default: 'flight').
        min_points
            Minimum valid data points required to include a flight.
        show_stats
            If True, show N, NMB, R statistics on each panel.
        scale_factor
            Scale factor for display values.
        obs_style
            Style for observations: 'scatter' or 'line'.
        model_style
            Style for model: 'line' or 'scatter'.
        **kwargs
            Additional options.

        Yields
        ------
        tuple[str, matplotlib.figure.Figure]
            Tuple of (flight_id, figure) for each flight.
        """
        style = self.config.style

        # Check for flight coordinate
        if flight_coord not in paired_data.coords:
            raise ValueError(
                f"Flight coordinate '{flight_coord}' not found in paired data. "
                f"Available coordinates: {list(paired_data.coords)}"
            )

        # Get unique flights
        flights = np.unique(paired_data[flight_coord].values)

        for flight in flights:
            flight_str = str(flight)
            mask = paired_data[flight_coord].values == flight

            # Get data for this flight
            times = pd.to_datetime(paired_data[time_dim].values[mask])
            obs_vals = paired_data[obs_var].values[mask] * scale_factor
            mod_vals = paired_data[model_var].values[mask] * scale_factor

            valid_obs = ~np.isnan(obs_vals)
            valid_both = valid_obs & ~np.isnan(mod_vals)

            # Check minimum points
            if valid_both.sum() < min_points:
                continue

            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            obs_vals = obs_vals[sort_idx]
            mod_vals = mod_vals[sort_idx]
            valid_obs = valid_obs[sort_idx]
            valid_both = valid_both[sort_idx]

            # Create single-panel figure
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot observations
            if obs_style == "scatter":
                ax.scatter(
                    times[valid_obs], obs_vals[valid_obs],
                    s=20, alpha=0.7, color="black", label="Obs", zorder=3
                )
            else:
                ax.plot(
                    times[valid_obs], obs_vals[valid_obs],
                    "o-", color="black", markersize=4, linewidth=0.8,
                    alpha=0.7, label="Obs", zorder=3
                )

            # Plot model
            if model_style == "line":
                ax.plot(
                    times, mod_vals,
                    color=style.model_color, linewidth=2, alpha=0.8,
                    label="Model", zorder=2
                )
            else:
                ax.scatter(
                    times[valid_both], mod_vals[valid_both],
                    s=20, alpha=0.7, color=style.model_color,
                    label="Model", zorder=2
                )

            # Compute and display stats
            if show_stats and valid_both.sum() > 0:
                n = valid_both.sum()
                obs_mean = obs_vals[valid_both].mean()
                mod_mean = mod_vals[valid_both].mean()
                nmb = 100 * (mod_mean - obs_mean) / obs_mean if obs_mean != 0 else 0
                if valid_both.sum() > 2:
                    r = np.corrcoef(obs_vals[valid_both], mod_vals[valid_both])[0, 1]
                else:
                    r = np.nan

                stats_text = f"N={n}\nNMB={nmb:+.0f}%"
                if not np.isnan(r):
                    stats_text += f"\nR={r:.2f}"

                text_cfg = self.config.text
                ax.text(
                    0.97, 0.97, stats_text,
                    transform=ax.transAxes, fontsize=text_cfg.fontsize - 2,
                    verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
                )

            # Title
            text_cfg = self.config.text
            if self.config.title:
                ax.set_title(
                    format_plot_title(f"{self.config.title} - Flight {flight_str}"),
                    fontsize=text_cfg.title_fontsize
                )
            else:
                ax.set_title(f"Flight {flight_str}", fontsize=text_cfg.title_fontsize)

            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=text_cfg.fontsize - 2)

            # Y-axis label
            units = get_variable_units(paired_data, obs_var)
            ylabel = get_variable_label(paired_data, obs_var, include_prefix=False)
            ylabel = format_label_with_units(ylabel, units)
            ax.set_ylabel(ylabel, fontsize=text_cfg.fontsize)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax.tick_params(axis="x", rotation=45, labelsize=text_cfg.tick_fontsize)
            ax.set_xlabel("UTC Time", fontsize=text_cfg.fontsize)

            plt.tight_layout()

            # Format flight ID for filename (YYYYMMDD format)
            flight_id = flight_str.replace("-", "")

            yield flight_id, fig


def plot_flight_timeseries(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    title: str | None = None,
    ncols: int = 3,
    min_points: int = 10,
    scale_factor: float = 1.0,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for flight-by-flight time series plots.

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
    ncols
        Number of columns in subplot grid.
    min_points
        Minimum valid data points required to include a flight.
    scale_factor
        Scale factor for display values.
    **kwargs
        Additional options passed to plotter.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    config = PlotConfig(title=title)
    plotter = FlightTimeSeriesPlotter(config)
    return plotter.plot(
        paired_data,
        obs_var,
        model_var,
        ncols=ncols,
        min_points=min_points,
        scale_factor=scale_factor,
        **kwargs,
    )
