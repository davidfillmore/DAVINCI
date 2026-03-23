"""Observation-only histogram renderer for DAVINCI.

Renders a distribution histogram of observed values with optional
statistics annotation box showing N, Mean, Median, Std, P10, P90.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from davinci_monet.plots.base import format_plot_title, get_variable_label
from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_PRIMARY

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_histogram")
class ObsHistogramPlotter(ObsPlotter):
    """Plotter for observation value distribution histograms.

    Draws a histogram with a red dashed median line and an optional
    statistics annotation box in the upper-right corner.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = ObsHistogramPlotter()
    >>> fig = plotter.plot(obs_data, "O3", n_bins=50, show_stats=True)
    """

    name: str = "obs_histogram"
    default_figsize: tuple[float, float] = (8, 5)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        n_bins: int = 30,
        show_stats: bool = True,
        title: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a histogram of observation values.

        Parameters
        ----------
        obs_data
            Observation dataset containing the variable.
        variable
            Name of the variable to histogram.
        ax
            Optional axes to plot on. If None, creates new figure.
        n_bins
            Number of histogram bins.
        show_stats
            If True, add a statistics annotation box.
        title
            Plot title. Defaults to "{variable} Distribution".
        color
            Histogram bar color. Defaults to OBS_COLOR.
        **kwargs
            Additional arguments passed to ax.hist.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        color = color or NCAR_PRIMARY

        # Extract finite values
        values = obs_data[variable].values.ravel()
        values = values[np.isfinite(values)]

        # Histogram
        ax.hist(
            values,
            bins=n_bins,
            color=color,
            edgecolor="white",
            alpha=0.8,
            **kwargs,
        )

        # Median line
        median = float(np.median(values))
        ax.axvline(median, color="#D62839", linestyle="--", linewidth=1.5)

        # Stats annotation
        if show_stats:
            n = len(values)
            mean = float(np.mean(values))
            std = float(np.std(values))
            p10 = float(np.percentile(values, 10))
            p90 = float(np.percentile(values, 90))

            stats_text = (
                f"N={n}\n"
                f"Mean={mean:.2f}\n"
                f"Median={median:.2f}\n"
                f"Std={std:.2f}\n"
                f"P10={p10:.2f}\n"
                f"P90={p90:.2f}"
            )
            ax.text(
                0.97,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=self.config.text.annotation,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(
                    boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"
                ),
            )

        # Labels
        var_label = get_variable_label(obs_data, variable, include_prefix=False)
        units = obs_data[variable].attrs.get("units", "")
        xlabel = f"{var_label} ({units})" if units else var_label
        ax.set_xlabel(xlabel, fontsize=self.config.text.fontsize)
        ax.set_ylabel("Count", fontsize=self.config.text.fontsize)

        # Title
        if title is None:
            title = f"{var_label} Distribution"
        else:
            title = format_plot_title(title)
        ax.set_title(title, fontsize=self.config.text.title_fontsize)

        ax.grid(True, alpha=0.3, axis="y")

        return fig
