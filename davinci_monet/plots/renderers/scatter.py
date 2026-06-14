"""Scatter plot renderer for DAVINCI.

This module provides scatter plot functionality for paired source
comparisons, including density coloring and regression lines.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots._stats import annotation_metrics
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    format_label_with_units,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_ACCENT
from davinci_monet.plots.titles import title_for_labeled_subset

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


def _source_display_name(dataset_label: Any) -> str:
    """Return the label form used for dataset names on axes."""
    return str(dataset_label).replace("_", " ").upper()


@register_plotter("scatter")
class ScatterPlotter(BasePlotter):
    """Plotter for scatter comparisons.

    Creates scatter plots comparing paired source values, with options
    for density coloring, regression lines, and
    statistical annotations.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = ScatterPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     x_var="airnow_o3",
    ...     y_var="cam_o3",
    ...     show_density=True,
    ...     show_regression=True,
    ... )
    """

    name: str = "scatter"
    default_figsize: tuple[float, float] = (6, 6)  # Square for x vs y

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a scatter plot from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one geometry and one dataset.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the scatter rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"ScatterPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.pair_axis == "geometry"), series[0])
        y_series = next((s for s in series if s.pair_axis == "dataset"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        show_density: bool = kwargs.pop("show_density", False)
        density_cmap: str = kwargs.pop("density_cmap", "viridis")
        density_bins: int = kwargs.pop("density_bins", 50)
        show_regression: bool = kwargs.pop("show_regression", True)
        show_one_to_one: bool = kwargs.pop("show_one_to_one", True)
        show_stats: bool = kwargs.pop("show_stats", True)
        color_by: str | None = kwargs.pop("color_by", None)
        marker_size: float | None = kwargs.pop("marker_size", None)
        alpha: float | None = kwargs.pop("alpha", None)

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get data and flatten
        geometry_values = paired_data[x_var].values.flatten()
        dataset_values = paired_data[y_var].values.flatten()

        # Remove NaN values
        mask = np.isfinite(geometry_values) & np.isfinite(dataset_values)
        geometry_values = geometry_values[mask]
        dataset_values = dataset_values[mask]

        if len(geometry_values) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=self.config.text.fontsize,
            )
            return fig

        # Calculate limits
        all_values = np.concatenate([geometry_values, dataset_values])
        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Add padding
        padding = (vmax - vmin) * 0.05
        vmin -= padding
        vmax += padding

        # Get style configuration
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize
        a = alpha if alpha is not None else (0.5 if len(geometry_values) > 1000 else style.alpha)

        # Scatter plot
        if show_density and len(geometry_values) > 10:
            # Density-colored scatter
            density = self._calculate_density(geometry_values, dataset_values, density_bins)
            scatter = ax.scatter(
                geometry_values,
                dataset_values,
                c=density,
                s=ms**2,
                alpha=a,
                cmap=density_cmap,
                norm=LogNorm() if density.max() / density.min() > 10 else None,
            )
            fig.colorbar(scatter, ax=ax, label="Point Density")
        elif color_by is not None and color_by in paired_data:
            # Color by another variable
            color_values = paired_data[color_by].values.flatten()[mask]
            scatter = ax.scatter(
                geometry_values,
                dataset_values,
                c=color_values,
                s=ms**2,
                alpha=a,
                cmap=density_cmap,
            )
            color_label = get_variable_label(paired_data, color_by)
            fig.colorbar(scatter, ax=ax, label=color_label)
        else:
            ax.scatter(
                geometry_values,
                dataset_values,
                c=style.y_color,
                s=ms**2,
                alpha=a,
                edgecolors="none",
            )

        # 1:1 line
        if show_one_to_one:
            ax.plot(
                [vmin, vmax],
                [vmin, vmax],
                "k--",
                linewidth=1,
                label="1:1",
                zorder=1,
            )

        # Regression line
        if show_regression:
            self._add_regression_line(ax, geometry_values, dataset_values, vmin, vmax)

        # Statistics annotation
        if show_stats:
            self._add_stats_annotation(ax, geometry_values, dataset_values)

        # Set equal aspect and limits
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")

        # Formatting
        self.apply_text_style(ax)

        # Set labels. Explicit labels are complete overrides; otherwise qualify
        # comparison axes by source identity when available.
        units = get_variable_units(paired_data, x_var)
        geometry_label_text = get_variable_label(paired_data, x_var)
        dataset_label_text = get_variable_label(paired_data, y_var)
        if self.config.geometry_label:
            geometry_label_text = self.config.geometry_label
        elif x_var in paired_data:
            dataset_label = paired_data[x_var].attrs.get("dataset_label")
            if dataset_label:
                geometry_label_text = f"{_source_display_name(dataset_label)} {geometry_label_text}"
        if self.config.dataset_label:
            dataset_label_text = self.config.dataset_label
        elif y_var in paired_data:
            dataset_label = paired_data[y_var].attrs.get("dataset_label")
            if dataset_label:
                dataset_label_text = f"{_source_display_name(dataset_label)} {dataset_label_text}"
        geometry_label = format_label_with_units(
            geometry_label_text or "Geometry",
            units,
        )
        dataset_label = format_label_with_units(
            dataset_label_text or "Dataset",
            units,
        )
        self.set_labels(ax, xlabel=geometry_label, ylabel=dataset_label)

        # Grid
        ax.grid(True, alpha=0.3)

        # Legend
        if show_one_to_one or show_regression:
            self.add_legend(ax, loc="upper left")

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        show_density: bool = False,
        density_cmap: str = "viridis",
        density_bins: int = 50,
        show_regression: bool = True,
        show_one_to_one: bool = True,
        show_stats: bool = True,
        color_by: str | None = None,
        marker_size: float | None = None,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a scatter plot.

        Parameters
        ----------
        paired_data
            Paired dataset with geometry and dataset variables.
        x_var
            Name of geometry variable.
        y_var
            Name of dataset variable.
        ax
            Optional axes to plot on. If None, creates new figure.
        show_density
            If True, color points by density.
        density_cmap
            Colormap for density coloring.
        density_bins
            Number of bins for density calculation.
        show_regression
            If True, show linear regression line.
        show_one_to_one
            If True, show 1:1 geometry line.
        show_stats
            If True, show statistics annotation.
        color_by
            Optional variable name to color points by.
        marker_size
            Override marker size.
        alpha
            Override alpha.
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
            show_density=show_density,
            density_cmap=density_cmap,
            density_bins=density_bins,
            show_regression=show_regression,
            show_one_to_one=show_one_to_one,
            show_stats=show_stats,
            color_by=color_by,
            marker_size=marker_size,
            alpha=alpha,
            **kwargs,
        )

    def plot_per_flight(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        flight_coord: str = "flight",
        min_points: int = 10,
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate scatter plots for each flight.

        Yields one figure per unique flight in the data.

        Parameters
        ----------
        paired_data
            Paired dataset with geometry and dataset variables.
        x_var
            Name of geometry variable.
        y_var
            Name of dataset variable.
        flight_coord
            Name of the flight coordinate (default: "flight").
        min_points
            Minimum valid data points per flight to generate a plot.
        **kwargs
            Additional arguments passed to plot method.

        Yields
        ------
        tuple[str, matplotlib.figure.Figure]
            Tuple of (flight_id, figure) for each flight.
        """
        # Check for flight coordinate
        if flight_coord not in paired_data.coords:
            raise ValueError(
                f"Flight coordinate '{flight_coord}' not found in paired data. "
                f"Available coordinates: {list(paired_data.coords)}"
            )

        # Get unique flights
        flight_values = paired_data[flight_coord].values
        unique_flights = np.unique(flight_values)

        for flight in unique_flights:
            # Convert to string (may be datetime or string)
            flight_str = str(flight)

            # Filter data for this flight
            mask = flight_values == flight
            flight_data = paired_data.isel(time=mask)

            # Check for minimum points
            geometry_vals = flight_data[x_var].values.flatten()
            dataset_vals = flight_data[y_var].values.flatten()
            valid = np.isfinite(geometry_vals) & np.isfinite(dataset_vals)

            if valid.sum() < min_points:
                continue

            # Update title/subtitle to identify this flight.
            original_title = self.config.title
            original_subtitle = self.config.subtitle
            self.config.title, flight_subtitle = title_for_labeled_subset(
                original_title,
                flight_str,
                label_prefix="Flight",
            )
            if flight_subtitle:
                self.config.subtitle = flight_subtitle

            # Generate plot for this flight
            fig = self.plot(flight_data, x_var, y_var, **kwargs)

            # Restore original title/subtitle for next iteration
            self.config.title = original_title
            self.config.subtitle = original_subtitle

            # Format flight ID for filename (YYYYMMDD format)
            flight_id = flight_str.replace("-", "")

            yield flight_id, fig

    def _calculate_density(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: int,
    ) -> np.ndarray:
        """Calculate point density for coloring.

        Parameters
        ----------
        x, y
            Data coordinates.
        bins
            Number of bins for histogram.

        Returns
        -------
        np.ndarray
            Density values for each point.
        """
        # 2D histogram
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # Find bin indices for each point
        xidx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
        yidx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)

        # Get density at each point
        density = hist[xidx, yidx]

        return density

    def _add_regression_line(
        self,
        ax: matplotlib.axes.Axes,
        geometry: np.ndarray,
        dataset: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        """Add regression line to plot.

        Parameters
        ----------
        ax
            Axes to add line to.
        geometry, dataset
            Data arrays.
        vmin, vmax
            Axis limits.
        """
        # Linear regression
        coeffs = np.polyfit(geometry, dataset, 1)
        slope, intercept = coeffs

        # Plot line
        x_line = np.array([vmin, vmax])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color=NCAR_ACCENT,
            linestyle="-",
            linewidth=1.5,
            label=f"y = {slope:.2f}x + {intercept:.2f}",
            zorder=2,
        )

    def _add_stats_annotation(
        self,
        ax: matplotlib.axes.Axes,
        geometry: np.ndarray,
        dataset: np.ndarray,
    ) -> None:
        """Add statistics annotation to plot.

        Parameters
        ----------
        ax
            Axes to annotate.
        geometry, dataset
            Data arrays.
        """
        # Calculate statistics (via central metric registry)
        stats = annotation_metrics(geometry, dataset, ["N", "MB", "RMSE", "R", "NME"])
        n = int(stats["N"])
        mb = stats["MB"]
        rmse = stats["RMSE"]
        r = stats["R"]
        nme = stats["NME"]

        # Create annotation text
        stats_text = (
            f"N = {n:,}\n"
            f"MB = {mb:.3g}\n"
            f"RMSE = {rmse:.3g}\n"
            f"R = {r:.3f}\n"
            f"NME = {nme:.1f}%"
        )

        # Add text box
        props = dict(boxstyle="round", facecolor="white", alpha=0.8)
        ax.text(
            0.95,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=self.config.text.annotation,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )


def plot_scatter(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for scatter plotting.

    Parameters
    ----------
    paired_data
        Paired dataset with geometry and dataset variables.
    x_var
        Name of geometry variable.
    y_var
        Name of dataset variable.
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

    plotter = ScatterPlotter(config=config)
    return plotter.plot(paired_data, x_var, y_var, **kwargs)
