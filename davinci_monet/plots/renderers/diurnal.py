"""Diurnal cycle plot renderer for DAVINCI.

This module provides diurnal cycle plotting functionality for comparing
dataset output with datasets across the daily cycle.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    format_label_with_units,
    get_axis_color,
    get_series_label,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("diurnal")
class DiurnalPlotter(BasePlotter):
    """Plotter for diurnal cycle comparisons.

    Creates plots showing the average diurnal pattern of dataset and
    dataset values, optionally with uncertainty bands.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = DiurnalPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     x_var="geometry_o3",
    ...     y_var="dataset_o3",
    ...     show_spread="iqr",
    ... )
    """

    name: str = "diurnal"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for temporal data

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a diurnal cycle plot from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the diurnal rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"DiurnalPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        time_dim: str = kwargs.pop("time_dim", "time")
        show_spread: Literal["none", "std", "iqr", "range"] = kwargs.pop("show_spread", "iqr")
        aggregate_dim: str | None = kwargs.pop("aggregate_dim", None)
        x_label: str | None = kwargs.pop("x_label", None)
        y_label: str | None = kwargs.pop("y_label", None)
        utc_offset: int = kwargs.pop("utc_offset", 0)

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get data
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]

        # Calculate hour of day
        time_coords = paired_data[time_dim]
        hours = (time_coords.dt.hour + utc_offset) % 24

        # Add hour as a coordinate for grouping
        x_data = x_data.assign_coords(hour=hours)
        y_data = y_data.assign_coords(hour=hours)

        # Group by hour and calculate statistics
        x_hourly = x_data.groupby("hour")
        y_hourly = y_data.groupby("hour")

        # Calculate means
        x_mean = x_hourly.mean()
        y_mean = y_hourly.mean()

        # Flatten to 1D if multi-dimensional
        x_mean_vals = x_mean.values
        y_mean_vals = y_mean.values
        if x_mean_vals.ndim > 1:
            x_mean_vals = np.nanmean(x_mean_vals, axis=tuple(range(1, x_mean_vals.ndim)))
            y_mean_vals = np.nanmean(y_mean_vals, axis=tuple(range(1, y_mean_vals.ndim)))

        # Get hour values for x-axis
        hours_arr = np.arange(24)

        # Get style configuration
        style = self.config.style

        # Series legend labels prefer source identity; axis remains a styling hint.
        x_label = x_label or get_series_label(paired_data, x_var, self.config.x_label)
        y_label = y_label or get_series_label(paired_data, y_var, self.config.y_label)

        # Series colors by source axis (geometry gray, dataset blue, else palette) (R-3).
        x_color = get_axis_color(
            paired_data,
            x_var,
            0,
            x_color=style.x_color,
            y_color=style.y_color,
        )
        y_color = get_axis_color(
            paired_data,
            y_var,
            1,
            x_color=style.x_color,
            y_color=style.y_color,
        )

        # Plot spread if requested
        if show_spread != "none":
            self._add_spread_bands(
                ax,
                x_hourly,
                y_hourly,
                hours_arr,
                show_spread,
                x_color,
                y_color,
            )

        # Plot means
        ax.plot(
            hours_arr,
            x_mean_vals,
            color=x_color,
            linestyle=style.x_linestyle,
            marker=style.x_marker,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=x_label,
        )

        ax.plot(
            hours_arr,
            y_mean_vals,
            color=y_color,
            linestyle=style.y_linestyle,
            marker=style.y_marker,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=y_label,
        )

        # Formatting
        self.apply_text_style(ax)

        # Set labels
        units = get_variable_units(paired_data, x_var)
        ylabel = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, x_var),
            units,
        )
        xlabel = "Hour (Local)" if utc_offset != 0 else "Hour (UTC)"
        self.set_labels(ax, xlabel=xlabel, ylabel=ylabel)
        self.set_limits(ax)

        # X-axis configuration
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(np.arange(0, 24, 3))

        # Add legend
        self.add_legend(ax)

        # Grid
        ax.grid(True, alpha=0.3)

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        time_dim: str = "time",
        show_spread: Literal["none", "std", "iqr", "range"] = "iqr",
        aggregate_dim: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        utc_offset: int = 0,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a diurnal cycle plot.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        ax
            Optional axes to plot on. If None, creates new figure.
        time_dim
            Name of time dimension.
        show_spread
            Type of spread to show ('none', 'std', 'iqr', 'range').
        aggregate_dim
            Optional additional dimension to aggregate (e.g., 'site').
        x_label
            Custom label for datasets.
        y_label
            Custom label for dataset.
        utc_offset
            Offset from UTC for local time (hours).
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
            time_dim=time_dim,
            show_spread=show_spread,
            aggregate_dim=aggregate_dim,
            x_label=x_label,
            y_label=y_label,
            utc_offset=utc_offset,
            **kwargs,
        )

    def _add_spread_bands(
        self,
        ax: matplotlib.axes.Axes,
        x_hourly: Any,
        y_hourly: Any,
        hours: np.ndarray,
        spread_type: str,
        x_color: str,
        y_color: str,
    ) -> None:
        """Add spread bands to the plot.

        Parameters
        ----------
        ax
            Axes to add bands to.
        geometry_hourly, dataset_hourly
            Grouped data by hour.
        hours
            Hour values for x-axis.
        spread_type
            Type of spread ('std', 'iqr', 'range').
        x_color, y_color
            Series colors (pair-axis) so the bands match the plotted lines.
        """
        if spread_type == "std":
            x_mean = x_hourly.mean()
            y_mean = y_hourly.mean()

            # Suppress warnings for hours with single datasets (ddof > n)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                x_std = x_hourly.std()
                y_std = y_hourly.std()

            x_lower = x_mean - x_std
            x_upper = x_mean + x_std
            y_lower = y_mean - y_std
            y_upper = y_mean + y_std

        elif spread_type == "iqr":
            x_lower = x_hourly.quantile(0.25)
            x_upper = x_hourly.quantile(0.75)
            y_lower = y_hourly.quantile(0.25)
            y_upper = y_hourly.quantile(0.75)

        else:  # range
            x_lower = x_hourly.min()
            x_upper = x_hourly.max()
            y_lower = y_hourly.min()
            y_upper = y_hourly.max()

        # Flatten to 1D if multi-dimensional (e.g., when grouping over time with site dimension)
        x_lower_vals = x_lower.values
        x_upper_vals = x_upper.values
        y_lower_vals = y_lower.values
        y_upper_vals = y_upper.values

        # If multi-dimensional, average over non-hour dimensions
        if x_lower_vals.ndim > 1:
            x_lower_vals = np.nanmean(x_lower_vals, axis=tuple(range(1, x_lower_vals.ndim)))
            x_upper_vals = np.nanmean(x_upper_vals, axis=tuple(range(1, x_upper_vals.ndim)))
            y_lower_vals = np.nanmean(y_lower_vals, axis=tuple(range(1, y_lower_vals.ndim)))
            y_upper_vals = np.nanmean(y_upper_vals, axis=tuple(range(1, y_upper_vals.ndim)))

        # Plot bands (pair-axis colors, matching the series; R-3)
        ax.fill_between(
            hours,
            x_lower_vals,
            x_upper_vals,
            color=x_color,
            alpha=0.2,
        )
        ax.fill_between(
            hours,
            y_lower_vals,
            y_upper_vals,
            color=y_color,
            alpha=0.2,
        )


def plot_diurnal(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for diurnal cycle plotting.

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
    **kwargs
        Additional arguments passed to plot method.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    if isinstance(config, dict):
        config = PlotConfig.from_dict(config)

    plotter = DiurnalPlotter(config=config)
    return plotter.plot(paired_data, x_var, y_var, **kwargs)
