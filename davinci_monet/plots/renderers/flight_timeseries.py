"""Flight-by-flight time series plot renderer for DAVINCI.

This module provides multi-panel time series plots showing dataset vs datasets
for individual aircraft flights. Designed for track datasets where each
flight (identified by date) is plotted in a separate panel.

Aircraft altitude is displayed on the right y-axis to provide context for
vertical sampling during each flight.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots._stats import annotation_metrics
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    format_label_with_units,
    get_dataset_color,
    get_series_label,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.titles import title_for_labeled_subset

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

# Default altitude variable names to search for (geometric altitude preferred over pressure)
# Note: DC-8 ICARTT files use Pressure_Altitude_BENNETT (in feet)
ALTITUDE_VAR_NAMES = (
    "Pressure_Altitude_BENNETT",  # DC-8 ASIA-AQ (feet)
    "GPS_Altitude_BENNETT",  # DC-8 GPS altitude (feet)
    "GPS_Altitude",  # Generic GPS altitude
    "Altitude",  # Generic
    "altitude",  # Standard name
    "alt",  # Short form
    "z",  # Vertical coordinate
    "ALT",  # Uppercase
    "ALTITUDE",  # Uppercase
)

# Variables that are in feet (not meters) - need conversion
FEET_ALTITUDE_VARS = {"Pressure_Altitude_BENNETT", "GPS_Altitude_BENNETT"}

# Conversion factor: feet to meters
FEET_TO_METERS = 0.3048

# Altitude color - light gray to not distract from main data
ALTITUDE_COLOR = "#A0A0A0"


def _get_altitude_data(
    dataset: xr.Dataset,
    altitude_var: str | None = None,
) -> tuple[np.ndarray[Any, np.dtype[Any]] | None, str, bool]:
    """Find and return altitude data from dataset.

    Parameters
    ----------
    dataset
        Dataset to search for altitude.
    altitude_var
        Specific altitude variable name. If None, searches common names.

    Returns
    -------
    tuple[np.ndarray | None, str, bool]
        Altitude values (or None if not found), the variable name used,
        and whether the data is in feet (needs conversion to meters).
    """
    # Build search order: specified var first, then defaults
    search_names = [altitude_var] if altitude_var else []
    search_names.extend(ALTITUDE_VAR_NAMES)

    for name in search_names:
        if name is None:
            continue
        # Check coordinates first
        if name in dataset.coords:
            is_feet = name in FEET_ALTITUDE_VARS
            return dataset.coords[name].values, name, is_feet
        # Then data variables
        if name in dataset.data_vars:
            is_feet = name in FEET_ALTITUDE_VARS
            return dataset[name].values, name, is_feet

    return None, "", False


@register_plotter("flight_timeseries")
class FlightTimeSeriesPlotter(BasePlotter):
    """Plotter for flight-by-flight time series comparisons.

    Creates a multi-panel figure with one subplot per flight,
    showing both dataset and dataset time series along the flight track.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = FlightTimeSeriesPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     geometry_var="geometry_O3_ROZE_STCLAIR",
    ...     dataset_var="dataset_O3_ROZE_STCLAIR",
    ...     ncols=3,
    ... )
    """

    name: str = "flight_timeseries"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for temporal data

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render flight-by-flight time series panels from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one geometry (geometry) and one dataset (dataset).
        ax
            Ignored for this plot type (creates own figure).
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            ncols (int, default 3), min_points (int, default 10),
            time_dim (str, default "time"), flight_coord (str, default "flight"),
            show_stats (bool, default True), scale_factor (float, default 1.0),
            geometry_style (str, default "scatter"), dataset_style (str, default "line"),
            show_altitude (bool, default True), altitude_var (str|None, default None),
            altitude_units (str, default "km").

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"FlightTimeSeriesPlotter.render requires exactly 2 series; got {len(series)}."
            )
        geometry_series = next((s for s in series if s.pair_axis == "geometry"), series[0])
        dataset_series = next((s for s in series if s.pair_axis == "dataset"), series[1])
        paired_data = geometry_series.dataset
        geometry_var = geometry_series.var_name
        dataset_var = dataset_series.var_name

        ncols: int = kwargs.pop("ncols", 3)
        min_points: int = kwargs.pop("min_points", 10)
        time_dim: str = kwargs.pop("time_dim", "time")
        flight_coord: str = kwargs.pop("flight_coord", "flight")
        show_stats: bool = kwargs.pop("show_stats", True)
        scale_factor: float = kwargs.pop("scale_factor", 1.0)
        geometry_style: str = kwargs.pop("geometry_style", "scatter")
        dataset_style: str = kwargs.pop("dataset_style", "line")
        show_altitude: bool = kwargs.pop("show_altitude", True)
        altitude_var: str | None = kwargs.pop("altitude_var", None)
        altitude_units: str = kwargs.pop("altitude_units", "km")

        style = self.config.style

        # Series colors/labels by source axis (R-3): geometry gray, dataset blue, else
        # palette; legends use the source label.
        sc_geometry, sc_dataset = style.geometry_color, style.dataset_color
        geometry_color = get_dataset_color(
            paired_data, geometry_var, 0, geometry_color=sc_geometry, dataset_color=sc_dataset
        )
        dataset_color = get_dataset_color(
            paired_data, dataset_var, 1, geometry_color=sc_geometry, dataset_color=sc_dataset
        )
        geometry_label = get_series_label(paired_data, geometry_var)
        dataset_label = get_series_label(paired_data, dataset_var)

        # Get unique flights and filter by data availability
        if flight_coord not in paired_data.coords:
            raise ValueError(f"Flight coordinate '{flight_coord}' not found in dataset")

        flights = np.unique(paired_data[flight_coord].values)
        valid_flights = []

        for flight in flights:
            mask = paired_data[flight_coord].values == flight
            geometry_vals = paired_data[geometry_var].values[mask]
            dataset_vals = paired_data[dataset_var].values[mask]
            valid = ~np.isnan(geometry_vals) & ~np.isnan(dataset_vals)
            if valid.sum() >= min_points:
                valid_flights.append(flight)

        if not valid_flights:
            raise ValueError(f"No flights with >= {min_points} valid data points")

        n_flights = len(valid_flights)
        nrows = (n_flights + ncols - 1) // ncols

        # Get altitude data if available
        alt_data, alt_var_name, is_feet = _get_altitude_data(paired_data, altitude_var)
        has_altitude = show_altitude and alt_data is not None
        # Convert feet to meters if needed, then meters to km if requested
        if is_feet:
            alt_scale = FEET_TO_METERS * (0.001 if altitude_units == "km" else 1.0)
        else:
            alt_scale = 0.001 if altitude_units == "km" else 1.0

        # Create figure with standard size
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8, 5),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Plot each flight
        for idx, flight in enumerate(valid_flights):
            panel_ax = axes_flat[idx]
            mask = paired_data[flight_coord].values == flight

            # Get data for this flight
            times = pd.to_datetime(paired_data[time_dim].values[mask])
            geometry_vals = paired_data[geometry_var].values[mask] * scale_factor
            dataset_vals = paired_data[dataset_var].values[mask] * scale_factor

            valid_geometry = ~np.isnan(geometry_vals)
            valid_both = valid_geometry & ~np.isnan(dataset_vals)

            # Sort by time for line plots
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            geometry_vals = geometry_vals[sort_idx]
            dataset_vals = dataset_vals[sort_idx]
            valid_geometry = valid_geometry[sort_idx]
            valid_both = valid_both[sort_idx]

            # Plot datasets
            if geometry_style == "scatter":
                panel_ax.scatter(
                    times[valid_geometry],
                    geometry_vals[valid_geometry],
                    s=12,
                    alpha=0.7,
                    color=geometry_color,
                    label=geometry_label,
                    zorder=3,
                )
            else:
                panel_ax.plot(
                    times[valid_geometry],
                    geometry_vals[valid_geometry],
                    "o-",
                    color=geometry_color,
                    markersize=3,
                    linewidth=0.5,
                    alpha=0.7,
                    label=geometry_label,
                    zorder=3,
                )

            # Plot dataset
            if dataset_style == "line":
                panel_ax.plot(
                    times,
                    dataset_vals,
                    color=dataset_color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=dataset_label,
                    zorder=2,
                )
            else:
                panel_ax.scatter(
                    times[valid_both],
                    dataset_vals[valid_both],
                    s=12,
                    alpha=0.7,
                    color=dataset_color,
                    label=dataset_label,
                    zorder=2,
                )

            # Plot altitude on right y-axis
            if has_altitude:
                alt_vals = alt_data[mask] * alt_scale  # type: ignore[index]
                alt_vals = alt_vals[sort_idx]
                valid_alt = ~np.isnan(alt_vals)

                ax2 = panel_ax.twinx()
                ax2.plot(
                    times[valid_alt],
                    alt_vals[valid_alt],
                    color=ALTITUDE_COLOR,
                    linewidth=1.0,
                    alpha=0.6,
                    label="Altitude",
                    zorder=1,
                )
                ax2.set_ylim(bottom=0)
                ax2.tick_params(
                    axis="y", labelcolor=ALTITUDE_COLOR, labelsize=self.config.text.annotation_small
                )
                # Only label right axis on rightmost column
                if (idx + 1) % ncols == 0 or idx == n_flights - 1:
                    ax2.set_ylabel(
                        f"Altitude ({altitude_units})",
                        fontsize=self.config.text.annotation_small,
                        color=ALTITUDE_COLOR,
                    )
                else:
                    ax2.set_ylabel("")

            # Compute and display stats (via central metric registry)
            if show_stats and valid_both.sum() > 0:
                geometry_mean = geometry_vals[valid_both].mean()
                stats = annotation_metrics(
                    geometry_vals[valid_both], dataset_vals[valid_both], ["N", "NMB", "R"]
                )
                n = int(stats["N"])
                nmb = stats["NMB"] if geometry_mean != 0 else 0
                # Preserve the renderer's <=2-point guard (registry R needs >=2)
                r = stats["R"] if valid_both.sum() > 2 else np.nan

                stats_text = f"N={n}\nNMB={nmb:+.0f}%"
                if not np.isnan(r):
                    stats_text += f"\nR={r:.2f}"

                # Multi-panel font sizes (smaller for dense panels)
                panel_ax.text(
                    0.97,
                    0.03,
                    stats_text,
                    transform=panel_ax.transAxes,
                    fontsize=self.config.text.annotation_small,
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Title with flight date
            panel_ax.set_title(f"Flight {flight}", fontsize=self.config.text.annotation_small)

            panel_ax.set_ylim(bottom=0)
            panel_ax.grid(True, alpha=0.3)

            # Legend on first panel only
            if idx == 0:
                panel_ax.legend(loc="lower left", fontsize=self.config.text.legend_small)

            # Y-axis label on left column
            if idx % ncols == 0:
                units = get_variable_units(paired_data, geometry_var)
                ylabel = get_variable_label(paired_data, geometry_var, include_prefix=False)
                ylabel = format_label_with_units(ylabel, units)
                panel_ax.set_ylabel(ylabel, fontsize=self.config.text.legend_small)

            # Format x-axis
            panel_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            panel_ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            panel_ax.tick_params(axis="x", rotation=45)
            panel_ax.set_xlabel("UTC Time", fontsize=self.config.text.legend_small)

        # Hide unused subplots
        for idx in range(n_flights, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Main title
        if self.config.title:
            self.set_figure_title(
                fig,
                self.config.title,
                y=1.02,
                fontsize=self.config.text.fontsize,
            )

        plt.tight_layout()
        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        geometry_var: str,
        dataset_var: str,
        ax: matplotlib.axes.Axes | None = None,
        ncols: int = 3,
        min_points: int = 10,
        time_dim: str = "time",
        flight_coord: str = "flight",
        show_stats: bool = True,
        scale_factor: float = 1.0,
        geometry_style: str = "scatter",
        dataset_style: str = "line",
        show_altitude: bool = True,
        altitude_var: str | None = None,
        altitude_units: str = "km",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate flight-by-flight time series panels.

        Thin wrapper around :meth:`render`. See that method for parameter docs.

        Parameters
        ----------
        paired_data
            Paired dataset with dataset and dataset variables.
        geometry_var
            Name of dataset variable.
        dataset_var
            Name of dataset variable.
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
        geometry_style
            Style for datasets: 'scatter' or 'line'.
        dataset_style
            Style for dataset: 'line' or 'scatter'.
        show_altitude
            If True, show aircraft altitude on right y-axis.
        altitude_var
            Name of altitude variable. If None, searches common names.
        altitude_units
            Units for altitude display ('km' or 'm'). Data assumed in meters.
        **kwargs
            Additional options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, geometry_var, dataset_var),
            ax=ax,
            ncols=ncols,
            min_points=min_points,
            time_dim=time_dim,
            flight_coord=flight_coord,
            show_stats=show_stats,
            scale_factor=scale_factor,
            geometry_style=geometry_style,
            dataset_style=dataset_style,
            show_altitude=show_altitude,
            altitude_var=altitude_var,
            altitude_units=altitude_units,
            **kwargs,
        )

    def plot_per_flight(
        self,
        paired_data: xr.Dataset,
        geometry_var: str,
        dataset_var: str,
        time_dim: str = "time",
        flight_coord: str = "flight",
        min_points: int = 10,
        show_stats: bool = True,
        scale_factor: float = 1.0,
        geometry_style: str = "scatter",
        dataset_style: str = "line",
        show_altitude: bool = True,
        altitude_var: str | None = None,
        altitude_units: str = "km",
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate individual time series plots for each flight.

        Yields one figure per unique flight in the data, each showing
        a single-panel time series comparison with altitude on right axis.

        Parameters
        ----------
        paired_data
            Paired dataset with dataset and dataset variables.
        geometry_var
            Name of dataset variable.
        dataset_var
            Name of dataset variable.
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
        geometry_style
            Style for datasets: 'scatter' or 'line'.
        dataset_style
            Style for dataset: 'line' or 'scatter'.
        show_altitude
            If True, show aircraft altitude on right y-axis.
        altitude_var
            Name of altitude variable. If None, searches common names.
        altitude_units
            Units for altitude display ('km' or 'm'). Data assumed in meters.
        **kwargs
            Additional options.

        Yields
        ------
        tuple[str, matplotlib.figure.Figure]
            Tuple of (flight_id, figure) for each flight.
        """
        style = self.config.style

        # Series colors/labels by source axis (R-3): geometry gray, dataset blue, else
        # palette; legends use the source label.
        sc_geometry, sc_dataset = style.geometry_color, style.dataset_color
        geometry_color = get_dataset_color(
            paired_data, geometry_var, 0, geometry_color=sc_geometry, dataset_color=sc_dataset
        )
        dataset_color = get_dataset_color(
            paired_data, dataset_var, 1, geometry_color=sc_geometry, dataset_color=sc_dataset
        )
        geometry_label = get_series_label(paired_data, geometry_var)
        dataset_label = get_series_label(paired_data, dataset_var)

        # Check for flight coordinate
        if flight_coord not in paired_data.coords:
            raise ValueError(
                f"Flight coordinate '{flight_coord}' not found in paired data. "
                f"Available coordinates: {list(paired_data.coords)}"
            )

        # Get altitude data if available
        alt_data, alt_var_name, is_feet = _get_altitude_data(paired_data, altitude_var)
        has_altitude = show_altitude and alt_data is not None
        # Convert feet to meters if needed, then meters to km if requested
        if is_feet:
            alt_scale = FEET_TO_METERS * (0.001 if altitude_units == "km" else 1.0)
        else:
            alt_scale = 0.001 if altitude_units == "km" else 1.0

        # Get unique flights
        flights = np.unique(paired_data[flight_coord].values)

        for flight in flights:
            flight_str = str(flight)
            mask = paired_data[flight_coord].values == flight

            # Get data for this flight
            times = pd.to_datetime(paired_data[time_dim].values[mask])
            geometry_vals = paired_data[geometry_var].values[mask] * scale_factor
            dataset_vals = paired_data[dataset_var].values[mask] * scale_factor

            valid_geometry = ~np.isnan(geometry_vals)
            valid_both = valid_geometry & ~np.isnan(dataset_vals)

            # Check minimum points
            if valid_both.sum() < min_points:
                continue

            # Sort by time
            sort_idx = np.argsort(times)
            times = times[sort_idx]
            geometry_vals = geometry_vals[sort_idx]
            dataset_vals = dataset_vals[sort_idx]
            valid_geometry = valid_geometry[sort_idx]
            valid_both = valid_both[sort_idx]

            # Create single-panel figure
            fig, ax = plt.subplots(figsize=(8, 5))

            # Text settings - use absolute point sizes from config
            text_cfg = self.config.text

            # Plot datasets
            if geometry_style == "scatter":
                ax.scatter(
                    times[valid_geometry],
                    geometry_vals[valid_geometry],
                    s=20,
                    alpha=0.7,
                    color=geometry_color,
                    label=geometry_label,
                    zorder=3,
                )
            else:
                ax.plot(
                    times[valid_geometry],
                    geometry_vals[valid_geometry],
                    "o-",
                    color=geometry_color,
                    markersize=4,
                    linewidth=0.8,
                    alpha=0.7,
                    label=geometry_label,
                    zorder=3,
                )

            # Plot dataset
            if dataset_style == "line":
                ax.plot(
                    times,
                    dataset_vals,
                    color=dataset_color,
                    linewidth=2,
                    alpha=0.8,
                    label=dataset_label,
                    zorder=2,
                )
            else:
                ax.scatter(
                    times[valid_both],
                    dataset_vals[valid_both],
                    s=20,
                    alpha=0.7,
                    color=dataset_color,
                    label=dataset_label,
                    zorder=2,
                )

            # Plot altitude on right y-axis
            ax2 = None
            if has_altitude:
                alt_vals = alt_data[mask] * alt_scale  # type: ignore[index]
                alt_vals = alt_vals[sort_idx]
                valid_alt = ~np.isnan(alt_vals)

                ax2 = ax.twinx()
                ax2.plot(
                    times[valid_alt],
                    alt_vals[valid_alt],
                    color=ALTITUDE_COLOR,
                    linewidth=1.2,
                    alpha=0.6,
                    label="Altitude",
                    zorder=1,
                )
                ax2.set_ylim(bottom=0)
                ax2.tick_params(
                    axis="y", labelcolor=ALTITUDE_COLOR, labelsize=text_cfg.tick_fontsize
                )
                ax2.set_ylabel(
                    f"Altitude ({altitude_units})", fontsize=text_cfg.fontsize, color=ALTITUDE_COLOR
                )

            # Compute and display stats (via central metric registry)
            if show_stats and valid_both.sum() > 0:
                geometry_mean = geometry_vals[valid_both].mean()
                stats = annotation_metrics(
                    geometry_vals[valid_both], dataset_vals[valid_both], ["N", "NMB", "R"]
                )
                n = int(stats["N"])
                nmb = stats["NMB"] if geometry_mean != 0 else 0
                # Preserve the renderer's <=2-point guard (registry R needs >=2)
                r = stats["R"] if valid_both.sum() > 2 else np.nan

                stats_text = f"N={n}\nNMB={nmb:+.0f}%"
                if not np.isnan(r):
                    stats_text += f"\nR={r:.2f}"

                ax.text(
                    0.97,
                    0.97,
                    stats_text,
                    transform=ax.transAxes,
                    fontsize=text_cfg.annotation,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Title/subtitle for this flight.
            title, flight_subtitle = title_for_labeled_subset(
                self.config.title,
                flight_str,
                label_prefix="Flight",
            )
            self.set_title(
                ax,
                title,
                subtitle=flight_subtitle,
                fontsize=text_cfg.title_fontsize,
            )

            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=text_cfg.legend)

            # Y-axis label
            units = get_variable_units(paired_data, geometry_var)
            ylabel = get_variable_label(paired_data, geometry_var, include_prefix=False)
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
    geometry_var: str,
    dataset_var: str,
    title: str | None = None,
    ncols: int = 3,
    min_points: int = 10,
    scale_factor: float = 1.0,
    show_altitude: bool = True,
    altitude_var: str | None = None,
    altitude_units: str = "km",
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for flight-by-flight time series plots.

    Parameters
    ----------
    paired_data
        Paired dataset with dataset and dataset variables.
    geometry_var
        Name of dataset variable.
    dataset_var
        Name of dataset variable.
    title
        Plot title.
    ncols
        Number of columns in subplot grid.
    min_points
        Minimum valid data points required to include a flight.
    scale_factor
        Scale factor for display values.
    show_altitude
        If True, show aircraft altitude on right y-axis.
    altitude_var
        Name of altitude variable. If None, searches common names.
    altitude_units
        Units for altitude display ('km' or 'm'). Data assumed in meters.
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
        geometry_var,
        dataset_var,
        ncols=ncols,
        min_points=min_points,
        scale_factor=scale_factor,
        show_altitude=show_altitude,
        altitude_var=altitude_var,
        altitude_units=altitude_units,
        **kwargs,
    )
