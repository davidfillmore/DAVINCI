"""Per-site time series plot renderer for DAVINCI.

This module provides individual time series plots for each monitoring site,
saved as separate files. Designed for AirNow, AERONET, and Pandora surface/column
datasets where detailed per-site analysis is needed.

Uses the same generator pattern as FlightTimeSeriesPlotter.plot_per_flight().
"""

from __future__ import annotations

import re
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


def _superscript(n: int) -> str:
    """Convert integer to superscript string."""
    superscripts = {
        "-": "\u207b",
        "0": "\u2070",
        "1": "\u00b9",
        "2": "\u00b2",
        "3": "\u00b3",
        "4": "\u2074",
        "5": "\u2075",
        "6": "\u2076",
        "7": "\u2077",
        "8": "\u2078",
        "9": "\u2079",
    }
    return "".join(superscripts.get(c, c) for c in str(n))


def sanitize_site_id(name: str) -> str:
    """Sanitize a site name for use in filenames.

    Replaces spaces and special characters with underscores, removes
    consecutive underscores, and strips leading/trailing underscores.
    """
    sanitized = re.sub(r"[^\w\-]", "_", str(name))
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


@register_plotter("per_site_timeseries")
class PerSiteTimeSeriesPlotter(BasePlotter):
    """Plotter that generates one detailed figure per monitoring site.

    Each figure is a single-panel timeseries showing dataset vs datasets
    with statistics, coordinates, and smart date formatting.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = PerSiteTimeSeriesPlotter()
    >>> for site_id, fig in plotter.plot_per_site(
    ...     paired_data, "geometry_o3", "dataset_o3"
    ... ):
    ...     fig.savefig(f"site_{site_id}.png")
    ...     plt.close(fig)
    """

    name: str = "per_site_timeseries"
    default_figsize: tuple[float, float] = (9, 4)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a single-site timeseries from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Ignored (creates own figure).
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            site (str|None, default None), min_points (int, default 20),
            time_dim (str, default "time"), site_dim (str, default "site"),
            show_stats (bool, default True), scale_factor (float, default 1.0),
            x_style (str, default "scatter"), y_style (str, default "line").

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"PerSiteTimeSeriesPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        site: str | None = kwargs.pop("site", None)
        min_points: int = kwargs.pop("min_points", 20)
        time_dim: str = kwargs.pop("time_dim", "time")
        site_dim: str = kwargs.pop("site_dim", "site")
        show_stats: bool = kwargs.pop("show_stats", True)
        scale_factor: float = kwargs.pop("scale_factor", 1.0)
        x_style: str = kwargs.pop("x_style", "scatter")
        y_style: str = kwargs.pop("y_style", "line")

        if site_dim not in paired_data.dims:
            raise ValueError(f"Site dimension '{site_dim}' not found in dataset")

        if site is not None:
            # Use the specified site
            site_data = paired_data.sel({site_dim: site})
        else:
            # Pick the first site with enough data
            sites = paired_data[site_dim].values
            chosen_site = None
            for s in sites:
                sd = paired_data.sel({site_dim: s})
                x_vals = sd[x_var].values
                y_vals = sd[y_var].values
                valid = ~np.isnan(x_vals) & ~np.isnan(y_vals)
                if valid.sum() >= min_points:
                    chosen_site = s
                    break
            if chosen_site is None:
                raise ValueError(f"No sites with >= {min_points} valid data points")
            site = chosen_site
            site_data = paired_data.sel({site_dim: site})

        fig, plot_ax = plt.subplots(figsize=self.default_figsize)

        self._plot_site_panel(
            plot_ax,
            site_data,
            paired_data,
            site,
            x_var,
            y_var,
            time_dim,
            site_dim,
            scale_factor,
            x_style,
            y_style,
            show_stats,
            single_panel=True,
        )

        plt.tight_layout()
        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        site: str | None = None,
        min_points: int = 20,
        time_dim: str = "time",
        site_dim: str = "site",
        show_stats: bool = True,
        scale_factor: float = 1.0,
        x_style: str = "scatter",
        y_style: str = "line",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot a single site timeseries.

        Thin wrapper around :meth:`render`. See that method for parameter docs.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        ax
            Ignored (creates own figure).
        site
            Site to plot. If None, picks the first site with sufficient data.
        min_points
            Minimum valid data points required.
        time_dim
            Name of time dimension.
        site_dim
            Name of site dimension.
        show_stats
            If True, show statistics box.
        scale_factor
            Scale factor for display values.
        x_style
            Style for datasets: 'scatter' or 'line'.
        y_style
            Style for dataset: 'line' or 'scatter'.
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
            site=site,
            min_points=min_points,
            time_dim=time_dim,
            site_dim=site_dim,
            show_stats=show_stats,
            scale_factor=scale_factor,
            x_style=x_style,
            y_style=y_style,
            **kwargs,
        )

    def plot_per_site(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        time_dim: str = "time",
        site_dim: str = "site",
        min_points: int = 20,
        show_stats: bool = True,
        scale_factor: float = 1.0,
        x_style: str = "scatter",
        y_style: str = "line",
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate individual time series plots for each site.

        Yields one figure per site in the data, each showing a single-panel
        time series comparison with statistics.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        time_dim
            Name of time dimension.
        site_dim
            Name of site dimension.
        min_points
            Minimum valid data points required to include a site.
        show_stats
            If True, show statistics box.
        scale_factor
            Scale factor for display values.
        x_style
            Style for datasets: 'scatter' or 'line'.
        y_style
            Style for dataset: 'line' or 'scatter'.
        **kwargs
            Additional options.

        Yields
        ------
        tuple[str, matplotlib.figure.Figure]
            Tuple of (sanitized_site_id, figure) for each site.
        """
        if site_dim not in paired_data.dims:
            raise ValueError(
                f"Site dimension '{site_dim}' not found in paired data. "
                f"Available dimensions: {list(paired_data.dims)}"
            )

        sites = paired_data[site_dim].values

        for site in sites:
            site_data = paired_data.sel({site_dim: site})

            x_vals = site_data[x_var].values
            y_vals = site_data[y_var].values
            valid_both = ~np.isnan(x_vals) & ~np.isnan(y_vals)

            if valid_both.sum() < min_points:
                continue

            fig, ax = plt.subplots(figsize=self.default_figsize)

            self._plot_site_panel(
                ax,
                site_data,
                paired_data,
                site,
                x_var,
                y_var,
                time_dim,
                site_dim,
                scale_factor,
                x_style,
                y_style,
                show_stats,
                single_panel=True,
            )

            plt.tight_layout()

            site_id = sanitize_site_id(str(site))
            yield site_id, fig

    def _plot_site_panel(
        self,
        ax: matplotlib.axes.Axes,
        site_data: xr.Dataset,
        full_data: xr.Dataset,
        site: Any,
        x_var: str,
        y_var: str,
        time_dim: str,
        site_dim: str,
        scale_factor: float,
        x_style: str,
        y_style: str,
        show_stats: bool,
        *,
        single_panel: bool = False,
    ) -> None:
        """Render a single site panel onto an axes.

        Parameters
        ----------
        ax
            Matplotlib axes to draw on.
        site_data
            Dataset sliced to a single site.
        full_data
            Full paired dataset (for coordinate lookup).
        site
            Site identifier value.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        time_dim
            Time dimension name.
        site_dim
            Site dimension name.
        scale_factor
            Multiplicative scale factor for display.
        x_style
            'scatter' or 'line' for datasets.
        y_style
            'line' or 'scatter' for dataset.
        show_stats
            Whether to display the statistics box.
        single_panel
            If True, use larger fonts for single-panel display.
        """
        style = self.config.style
        text_cfg = self.config.text

        times = pd.to_datetime(site_data[time_dim].values)
        x_vals = site_data[x_var].values * scale_factor
        y_vals = site_data[y_var].values * scale_factor

        valid_geometry = ~np.isnan(x_vals)
        valid_both = valid_geometry & ~np.isnan(y_vals)

        # Series colors/labels by source axis (R-3): geometry gray, dataset blue, else
        # palette; legends use the source label.
        x_color = get_axis_color(
            site_data,
            x_var,
            0,
            x_color=style.x_color,
            y_color=style.y_color,
        )
        y_color = get_axis_color(
            site_data,
            y_var,
            1,
            x_color=style.x_color,
            y_color=style.y_color,
        )
        x_label = get_series_label(site_data, x_var)
        y_label = get_series_label(site_data, y_var)

        # Plot datasets
        if x_style == "scatter":
            ax.scatter(
                times[valid_geometry],
                x_vals[valid_geometry],
                s=20,
                alpha=0.7,
                color=x_color,
                label=x_label,
                zorder=3,
            )
        else:
            ax.plot(
                times[valid_geometry],
                x_vals[valid_geometry],
                "o-",
                color=x_color,
                markersize=4,
                linewidth=0.8,
                alpha=0.7,
                label=x_label,
                zorder=3,
            )

        # Plot dataset
        if y_style == "line":
            ax.plot(
                times,
                y_vals,
                color=y_color,
                linewidth=2,
                alpha=0.8,
                label=y_label,
                zorder=2,
            )
        else:
            ax.scatter(
                times[valid_both],
                y_vals[valid_both],
                s=20,
                alpha=0.7,
                color=y_color,
                label=y_label,
                zorder=2,
            )

        # Statistics box (via central metric registry)
        if show_stats and valid_both.sum() > 0:
            x_mean = float(x_vals[valid_both].mean())
            stats = annotation_metrics(
                x_vals[valid_both],
                y_vals[valid_both],
                ["N", "MB", "RMSE", "NMB", "R"],
            )
            n = int(stats["N"])
            mb = stats["MB"]
            rmse = stats["RMSE"]
            nmb = stats["NMB"] if x_mean != 0 else 0.0
            # Preserve the renderer's <=2-point guard (registry R needs >=2)
            r = stats["R"] if valid_both.sum() > 2 else np.nan

            stats_text = f"N={n}\nMB={mb:+.2f}\nRMSE={rmse:.2f}\nNMB={nmb:+.0f}%"
            if not np.isnan(r):
                stats_text += f"\nR={r:.2f}"

            fontsize = text_cfg.annotation if single_panel else text_cfg.annotation_small
            ax.text(
                0.97,
                0.97,
                stats_text,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Title with site name and coordinates
        has_coords = "latitude" in full_data.coords and "longitude" in full_data.coords
        site_str = str(site)

        if has_coords:
            lat = float(full_data["latitude"].sel({site_dim: site}).values)
            lon = float(full_data["longitude"].sel({site_dim: site}).values)
            site_title = f"{site_str} ({lat:.1f}\u00b0N, {lon:.1f}\u00b0E)"
        else:
            site_title = site_str

        if self.config.title:
            full_title = f"{self.config.title} \u2014 {site_title}"
        else:
            full_title = site_title

        title_fontsize = text_cfg.title_fontsize if single_panel else text_cfg.annotation_small
        self.set_title(ax, full_title, fontsize=title_fontsize)

        # Y-axis
        ax.set_ylim(bottom=0)
        units = get_variable_units(full_data, x_var)
        ylabel = get_variable_label(full_data, x_var, include_prefix=False)
        if scale_factor != 1.0:
            exp = int(np.log10(1 / scale_factor))
            if units and units != "1":
                ylabel = f"{ylabel}\n(\u00d710{_superscript(exp)} {units})"
            else:
                ylabel = f"{ylabel}\n(\u00d710{_superscript(exp)})"
        else:
            ylabel = format_label_with_units(ylabel, units)

        label_fontsize = text_cfg.fontsize if single_panel else text_cfg.legend_small
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

        ax.grid(True, alpha=0.3)
        legend_fontsize = text_cfg.legend if single_panel else text_cfg.legend_small
        ax.legend(loc="upper left", fontsize=legend_fontsize)

        # Smart x-axis date formatting
        time_range = times.max() - times.min()
        if time_range <= pd.Timedelta(hours=24):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        elif time_range <= pd.Timedelta(days=7):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(time_range.days / 8))))

        tick_fontsize = text_cfg.tick_fontsize if single_panel else text_cfg.annotation_small
        ax.tick_params(axis="x", rotation=45, labelsize=tick_fontsize)
        ax.set_xlabel("UTC Time", fontsize=label_fontsize)


def plot_per_site_timeseries(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    title: str | None = None,
    site: str | None = None,
    min_points: int = 20,
    scale_factor: float = 1.0,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for a single per-site timeseries plot.

    Parameters
    ----------
    paired_data
        Paired dataset with x and y variables.
    x_var
        Name of the x variable.
    y_var
        Name of the y variable.
    title
        Plot title.
    site
        Specific site to plot. If None, picks the first valid site.
    min_points
        Minimum valid data points required.
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
    plotter = PerSiteTimeSeriesPlotter(config)
    return plotter.plot(
        paired_data,
        x_var,
        y_var,
        site=site,
        min_points=min_points,
        scale_factor=scale_factor,
        **kwargs,
    )
