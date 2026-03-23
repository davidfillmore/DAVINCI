"""Diurnal cycle plot renderer for DAVINCI.

This module provides diurnal cycle plotting functionality for comparing
model output with observations across the daily cycle.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    format_label_with_units,
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

    Creates plots showing the average diurnal pattern of model and
    observation values, optionally with uncertainty bands.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = DiurnalPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     show_spread="iqr",
    ... )
    """

    name: str = "diurnal"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for temporal data

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        time_dim: str = "time",
        show_spread: Literal["none", "std", "iqr", "range"] = "iqr",
        aggregate_dim: str | None = None,
        obs_label: str | None = None,
        model_label: str | None = None,
        utc_offset: int = 0,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a diurnal cycle plot.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Optional axes to plot on. If None, creates new figure.
        time_dim
            Name of time dimension.
        show_spread
            Type of spread to show ('none', 'std', 'iqr', 'range').
        aggregate_dim
            Optional additional dimension to aggregate (e.g., 'site').
        obs_label
            Custom label for observations.
        model_label
            Custom label for model.
        utc_offset
            Offset from UTC for local time (hours).
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get data
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Calculate hour of day
        time_coords = paired_data[time_dim]
        hours = (time_coords.dt.hour + utc_offset) % 24

        # Add hour as a coordinate for grouping
        obs_data = obs_data.assign_coords(hour=hours)
        model_data = model_data.assign_coords(hour=hours)

        # Group by hour and calculate statistics
        obs_hourly = obs_data.groupby("hour")
        model_hourly = model_data.groupby("hour")

        # Calculate means
        obs_mean = obs_hourly.mean()
        model_mean = model_hourly.mean()

        # Flatten to 1D if multi-dimensional
        obs_mean_vals = obs_mean.values
        model_mean_vals = model_mean.values
        if obs_mean_vals.ndim > 1:
            obs_mean_vals = np.nanmean(obs_mean_vals, axis=tuple(range(1, obs_mean_vals.ndim)))
            model_mean_vals = np.nanmean(
                model_mean_vals, axis=tuple(range(1, model_mean_vals.ndim))
            )

        # Get hour values for x-axis
        hours_arr = np.arange(24)

        # Get style configuration
        style = self.config.style

        # Get labels
        obs_label = (
            obs_label
            or get_variable_label(paired_data, obs_var, self.config.obs_label)
            or "Observations"
        )
        model_label = (
            model_label
            or get_variable_label(paired_data, model_var, self.config.model_label)
            or "Model"
        )

        # Plot spread if requested
        if show_spread != "none":
            self._add_spread_bands(ax, obs_hourly, model_hourly, hours_arr, style, show_spread)

        # Plot means
        ax.plot(
            hours_arr,
            obs_mean_vals,
            color=style.obs_color,
            linestyle=style.obs_linestyle,
            marker=style.obs_marker,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=obs_label,
        )

        ax.plot(
            hours_arr,
            model_mean_vals,
            color=style.model_color,
            linestyle=style.model_linestyle,
            marker=style.model_marker,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=model_label,
        )

        # Formatting
        self.apply_text_style(ax)

        # Set labels
        units = get_variable_units(paired_data, obs_var)
        ylabel = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, obs_var),
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

    def _add_spread_bands(
        self,
        ax: matplotlib.axes.Axes,
        obs_hourly: Any,
        model_hourly: Any,
        hours: np.ndarray,
        style: Any,
        spread_type: str,
    ) -> None:
        """Add spread bands to the plot.

        Parameters
        ----------
        ax
            Axes to add bands to.
        obs_hourly, model_hourly
            Grouped data by hour.
        hours
            Hour values for x-axis.
        style
            Style configuration.
        spread_type
            Type of spread ('std', 'iqr', 'range').
        """
        if spread_type == "std":
            obs_mean = obs_hourly.mean()
            model_mean = model_hourly.mean()

            # Suppress warnings for hours with single observations (ddof > n)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                obs_std = obs_hourly.std()
                model_std = model_hourly.std()

            obs_lower = obs_mean - obs_std
            obs_upper = obs_mean + obs_std
            model_lower = model_mean - model_std
            model_upper = model_mean + model_std

        elif spread_type == "iqr":
            obs_lower = obs_hourly.quantile(0.25)
            obs_upper = obs_hourly.quantile(0.75)
            model_lower = model_hourly.quantile(0.25)
            model_upper = model_hourly.quantile(0.75)

        else:  # range
            obs_lower = obs_hourly.min()
            obs_upper = obs_hourly.max()
            model_lower = model_hourly.min()
            model_upper = model_hourly.max()

        # Flatten to 1D if multi-dimensional (e.g., when grouping over time with site dimension)
        obs_lower_vals = obs_lower.values
        obs_upper_vals = obs_upper.values
        model_lower_vals = model_lower.values
        model_upper_vals = model_upper.values

        # If multi-dimensional, average over non-hour dimensions
        if obs_lower_vals.ndim > 1:
            obs_lower_vals = np.nanmean(obs_lower_vals, axis=tuple(range(1, obs_lower_vals.ndim)))
            obs_upper_vals = np.nanmean(obs_upper_vals, axis=tuple(range(1, obs_upper_vals.ndim)))
            model_lower_vals = np.nanmean(
                model_lower_vals, axis=tuple(range(1, model_lower_vals.ndim))
            )
            model_upper_vals = np.nanmean(
                model_upper_vals, axis=tuple(range(1, model_upper_vals.ndim))
            )

        # Plot bands
        ax.fill_between(
            hours,
            obs_lower_vals,
            obs_upper_vals,
            color=style.obs_color,
            alpha=0.2,
        )
        ax.fill_between(
            hours,
            model_lower_vals,
            model_upper_vals,
            color=style.model_color,
            alpha=0.2,
        )


def plot_diurnal(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for diurnal cycle plotting.

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
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
