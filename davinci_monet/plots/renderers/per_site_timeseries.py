"""Per-site time series plot renderer for DAVINCI-MONET.

This module provides individual time series plots for each monitoring site,
saved as separate files. Designed for AirNow, AERONET, and Pandora surface/column
observations where detailed per-site analysis is needed.

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


def _superscript(n: int) -> str:
    """Convert integer to superscript string."""
    superscripts = {
        "-": "\u207b", "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
        "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077", "8": "\u2078", "9": "\u2079",
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

    Each figure is a single-panel timeseries showing model vs observations
    with statistics, coordinates, and smart date formatting.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = PerSiteTimeSeriesPlotter()
    >>> for site_id, fig in plotter.plot_per_site(
    ...     paired_data, "obs_o3", "model_o3"
    ... ):
    ...     fig.savefig(f"site_{site_id}.png")
    ...     plt.close(fig)
    """

    name: str = "per_site_timeseries"
    default_figsize: tuple[float, float] = (9, 4)

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        site: str | None = None,
        min_points: int = 20,
        time_dim: str = "time",
        site_dim: str = "site",
        show_stats: bool = True,
        scale_factor: float = 1.0,
        obs_style: str = "scatter",
        model_style: str = "line",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot a single site timeseries.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
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
        if site_dim not in paired_data.dims:
            raise ValueError(f"Site dimension '{site_dim}' not found in dataset")

        if site is not None:
            # Use the specified site
            site_data = paired_data.sel({site_dim: site})
        else:
            # Pick the first site with enough data
            sites = paired_data[site_dim].values
            site = None
            for s in sites:
                sd = paired_data.sel({site_dim: s})
                obs_vals = sd[obs_var].values
                mod_vals = sd[model_var].values
                valid = ~np.isnan(obs_vals) & ~np.isnan(mod_vals)
                if valid.sum() >= min_points:
                    site = s
                    break
            if site is None:
                raise ValueError(f"No sites with >= {min_points} valid data points")
            site_data = paired_data.sel({site_dim: site})

        fig, plot_ax = plt.subplots(figsize=self.default_figsize)

        self._plot_site_panel(
            plot_ax, site_data, paired_data, site,
            obs_var, model_var, time_dim, site_dim,
            scale_factor, obs_style, model_style, show_stats,
            single_panel=True,
        )

        plt.tight_layout()
        return fig

    def plot_per_site(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        time_dim: str = "time",
        site_dim: str = "site",
        min_points: int = 20,
        show_stats: bool = True,
        scale_factor: float = 1.0,
        obs_style: str = "scatter",
        model_style: str = "line",
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate individual time series plots for each site.

        Yields one figure per site in the data, each showing a single-panel
        time series comparison with statistics.

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
        site_dim
            Name of site dimension.
        min_points
            Minimum valid data points required to include a site.
        show_stats
            If True, show statistics box.
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

            obs_vals = site_data[obs_var].values
            mod_vals = site_data[model_var].values
            valid_both = ~np.isnan(obs_vals) & ~np.isnan(mod_vals)

            if valid_both.sum() < min_points:
                continue

            fig, ax = plt.subplots(figsize=self.default_figsize)

            self._plot_site_panel(
                ax, site_data, paired_data, site,
                obs_var, model_var, time_dim, site_dim,
                scale_factor, obs_style, model_style, show_stats,
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
        obs_var: str,
        model_var: str,
        time_dim: str,
        site_dim: str,
        scale_factor: float,
        obs_style: str,
        model_style: str,
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
        obs_var
            Observation variable name.
        model_var
            Model variable name.
        time_dim
            Time dimension name.
        site_dim
            Site dimension name.
        scale_factor
            Multiplicative scale factor for display.
        obs_style
            'scatter' or 'line' for observations.
        model_style
            'line' or 'scatter' for model.
        show_stats
            Whether to display the statistics box.
        single_panel
            If True, use larger fonts for single-panel display.
        """
        style = self.config.style
        text_cfg = self.config.text

        times = pd.to_datetime(site_data[time_dim].values)
        obs_vals = site_data[obs_var].values * scale_factor
        mod_vals = site_data[model_var].values * scale_factor

        valid_obs = ~np.isnan(obs_vals)
        valid_both = valid_obs & ~np.isnan(mod_vals)

        # Plot observations
        if obs_style == "scatter":
            ax.scatter(
                times[valid_obs], obs_vals[valid_obs],
                s=20, alpha=0.7, color="black", label="Obs", zorder=3,
            )
        else:
            ax.plot(
                times[valid_obs], obs_vals[valid_obs],
                "o-", color="black", markersize=4, linewidth=0.8,
                alpha=0.7, label="Obs", zorder=3,
            )

        # Plot model
        if model_style == "line":
            ax.plot(
                times, mod_vals,
                color=style.model_color, linewidth=2, alpha=0.8,
                label="Model", zorder=2,
            )
        else:
            ax.scatter(
                times[valid_both], mod_vals[valid_both],
                s=20, alpha=0.7, color=style.model_color,
                label="Model", zorder=2,
            )

        # Statistics box
        if show_stats and valid_both.sum() > 0:
            n = int(valid_both.sum())
            obs_mean = float(obs_vals[valid_both].mean())
            mod_mean = float(mod_vals[valid_both].mean())
            mb = mod_mean - obs_mean
            nmb = 100 * mb / obs_mean if obs_mean != 0 else 0.0

            # RMSE
            diff = mod_vals[valid_both] - obs_vals[valid_both]
            rmse = float(np.sqrt(np.mean(diff**2)))

            # Correlation
            if valid_both.sum() > 2:
                r = float(np.corrcoef(obs_vals[valid_both], mod_vals[valid_both])[0, 1])
            else:
                r = np.nan

            stats_text = f"N={n}\nMB={mb:+.2f}\nRMSE={rmse:.2f}\nNMB={nmb:+.0f}%"
            if not np.isnan(r):
                stats_text += f"\nR={r:.2f}"

            fontsize = text_cfg.annotation if single_panel else text_cfg.annotation_small
            ax.text(
                0.97, 0.97, stats_text,
                transform=ax.transAxes, fontsize=fontsize,
                verticalalignment="top", horizontalalignment="right",
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
        ax.set_title(format_plot_title(full_title), fontsize=title_fontsize)

        # Y-axis
        ax.set_ylim(bottom=0)
        units = get_variable_units(full_data, obs_var)
        ylabel = get_variable_label(full_data, obs_var, include_prefix=False)
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
    obs_var: str,
    model_var: str,
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
        Paired dataset with model and observation variables.
    obs_var
        Name of observation variable.
    model_var
        Name of model variable.
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
        obs_var,
        model_var,
        site=site,
        min_points=min_points,
        scale_factor=scale_factor,
        **kwargs,
    )
