"""Site-by-site time series plot renderer for DAVINCI.

This module provides multi-panel time series plots showing dataset vs datasets
at individual monitoring sites. Useful for point datasets (surface stations,
column measurements) where each site has different characteristics.
"""

from __future__ import annotations

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


@register_plotter("site_timeseries")
class SiteTimeSeriesPlotter(BasePlotter):
    """Plotter for site-by-site time series comparisons.

    Creates a multi-panel figure with one subplot per monitoring site,
    showing both dataset and dataset time series for direct comparison.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = SiteTimeSeriesPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     x_var="geometry_no2_column",
    ...     y_var="dataset_no2_column",
    ...     ncols=3,
    ... )
    """

    name: str = "site_timeseries"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for temporal data

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render site-by-site time series panels from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one geometry (geometry) and one dataset (dataset).
        ax
            Ignored for this plot type (creates own figure).
        **kwargs
            Forwarded kwargs; renderer-specific ones:
            ncols (int, default 3), min_points (int, default 20),
            time_dim (str, default "time"), site_dim (str, default "site"),
            show_stats (bool, default True), scale_factor (float, default 1.0),
            geometry_style (str, default "scatter"), dataset_style (str, default "line").

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"SiteTimeSeriesPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        ncols: int = kwargs.pop("ncols", 3)
        min_points: int = kwargs.pop("min_points", 20)
        time_dim: str = kwargs.pop("time_dim", "time")
        site_dim: str = kwargs.pop("site_dim", "site")
        show_stats: bool = kwargs.pop("show_stats", True)
        scale_factor: float = kwargs.pop("scale_factor", 1.0)
        geometry_style: str = kwargs.pop("geometry_style", "scatter")
        dataset_style: str = kwargs.pop("dataset_style", "line")

        style = self.config.style

        # Get sites and filter by data availability
        if site_dim not in paired_data.dims:
            raise ValueError(f"Site dimension '{site_dim}' not found in dataset")

        sites = paired_data[site_dim].values
        valid_sites = []

        for site in sites:
            site_data = paired_data.sel({site_dim: site})
            geometry_vals = site_data[x_var].values
            dataset_vals = site_data[y_var].values
            valid = ~np.isnan(geometry_vals) & ~np.isnan(dataset_vals)
            if valid.sum() >= min_points:
                valid_sites.append(site)

        if not valid_sites:
            raise ValueError(f"No sites with >= {min_points} valid data points")

        n_sites = len(valid_sites)
        nrows = (n_sites + ncols - 1) // ncols

        # Create figure with standard size
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8, 5),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Get coordinate info if available
        has_coords = "latitude" in paired_data.coords and "longitude" in paired_data.coords

        # Plot each site
        for idx, site in enumerate(valid_sites):
            panel_ax = axes_flat[idx]
            site_data = paired_data.sel({site_dim: site})

            # Get data
            geometry_da = site_data[x_var]
            dataset_da = site_data[y_var]
            times = pd.to_datetime(site_data[time_dim].values)

            geometry_vals = geometry_da.values * scale_factor
            dataset_vals = dataset_da.values * scale_factor

            valid_geometry = ~np.isnan(geometry_vals)
            valid_both = valid_geometry & ~np.isnan(dataset_vals)

            # Series colors/labels by source axis (R-3): geometry gray, dataset blue,
            # else palette; legends use the source label.
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
            if geometry_style == "scatter":
                panel_ax.scatter(
                    times[valid_geometry],
                    geometry_vals[valid_geometry],
                    s=8,
                    alpha=0.6,
                    color=x_color,
                    label=x_label,
                    zorder=3,
                )
            else:
                panel_ax.plot(
                    times[valid_geometry],
                    geometry_vals[valid_geometry],
                    "o-",
                    color=x_color,
                    markersize=3,
                    linewidth=0.5,
                    alpha=0.7,
                    label=x_label,
                    zorder=3,
                )

            # Plot dataset
            if dataset_style == "line":
                panel_ax.plot(
                    times,
                    dataset_vals,
                    color=y_color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=y_label,
                    zorder=2,
                )
            else:
                panel_ax.scatter(
                    times[valid_both],
                    dataset_vals[valid_both],
                    s=8,
                    alpha=0.6,
                    color=y_color,
                    label=y_label,
                    zorder=2,
                )

            # Compute and display stats
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
                    0.97,
                    stats_text,
                    transform=panel_ax.transAxes,
                    fontsize=self.config.text.annotation_small,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            # Title with site name and coordinates
            if has_coords:
                lat = float(paired_data["latitude"].sel({site_dim: site}).values)
                lon = float(paired_data["longitude"].sel({site_dim: site}).values)
                panel_ax.set_title(
                    f"{site} ({lat:.1f}°N, {lon:.1f}°E)",
                    fontsize=self.config.text.annotation_small,
                )
            else:
                panel_ax.set_title(str(site), fontsize=self.config.text.annotation_small)

            panel_ax.set_ylim(bottom=0)
            panel_ax.grid(True, alpha=0.3)

            # Legend on first panel only
            if idx == 0:
                panel_ax.legend(loc="upper left", fontsize=self.config.text.legend_small)

            # Y-axis label on left column - use automatic variable display name (no prefix)
            if idx % ncols == 0:
                units = get_variable_units(paired_data, x_var)
                ylabel = get_variable_label(paired_data, x_var, include_prefix=False)
                if scale_factor != 1.0:
                    exp = int(np.log10(1 / scale_factor))
                    ylabel = (
                        f"{ylabel}\n(×10{_superscript(exp)} {units})"
                        if units and units != "1"
                        else f"{ylabel}\n(×10{_superscript(exp)})"
                    )
                else:
                    ylabel = format_label_with_units(ylabel, units)
                panel_ax.set_ylabel(ylabel, fontsize=self.config.text.legend_small)

        # Set x-axis limits to actual data range (avoid extra ticks beyond data)
        all_times = pd.to_datetime(paired_data[time_dim].values)
        axes_flat[0].set_xlim(all_times.min(), all_times.max())

        # Format x-axis on bottom row
        for idx in range(n_sites):
            if idx >= n_sites - ncols:  # Bottom row
                bottom_ax = axes_flat[idx]
                bottom_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                bottom_ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                bottom_ax.tick_params(axis="x", rotation=45)

        # Hide unused subplots
        for idx in range(n_sites, len(axes_flat)):
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
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        ncols: int = 3,
        min_points: int = 20,
        time_dim: str = "time",
        site_dim: str = "site",
        show_stats: bool = True,
        scale_factor: float = 1.0,
        geometry_style: str = "scatter",
        dataset_style: str = "line",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate site-by-site time series panels.

        Thin wrapper around :meth:`render`. See that method for parameter docs.

        Parameters
        ----------
        paired_data
            Paired dataset with dataset and dataset variables.
        x_var
            Name of dataset variable.
        y_var
            Name of dataset variable.
        ax
            Ignored for this plot type (creates own figure).
        ncols
            Number of columns in subplot grid.
        min_points
            Minimum valid data points required to include a site.
        time_dim
            Name of time dimension.
        site_dim
            Name of site dimension.
        show_stats
            If True, show N, NMB, R statistics on each panel.
        scale_factor
            Scale factor for display (e.g., 1e4 for mol/m2 -> 10^-4 mol/m2).
        geometry_style
            Style for datasets: 'scatter' or 'line'.
        dataset_style
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
            ncols=ncols,
            min_points=min_points,
            time_dim=time_dim,
            site_dim=site_dim,
            show_stats=show_stats,
            scale_factor=scale_factor,
            geometry_style=geometry_style,
            dataset_style=dataset_style,
            **kwargs,
        )


def _superscript(n: int) -> str:
    """Convert integer to superscript string."""
    superscripts = {
        "-": "⁻",
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    return "".join(superscripts.get(c, c) for c in str(n))


def plot_site_timeseries(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    title: str | None = None,
    ncols: int = 3,
    min_points: int = 20,
    scale_factor: float = 1.0,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for site-by-site time series plots.

    Parameters
    ----------
    paired_data
        Paired dataset with dataset and dataset variables.
    x_var
        Name of dataset variable.
    y_var
        Name of dataset variable.
    title
        Plot title.
    ncols
        Number of columns in subplot grid.
    min_points
        Minimum valid data points required to include a site.
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
    plotter = SiteTimeSeriesPlotter(config)
    return plotter.plot(
        paired_data,
        x_var,
        y_var,
        ncols=ncols,
        min_points=min_points,
        scale_factor=scale_factor,
        **kwargs,
    )
