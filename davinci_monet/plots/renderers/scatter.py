"""Scatter plot renderer for DAVINCI-MONET.

This module provides scatter plot functionality for comparing
model output with observations, including density coloring and
regression lines.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    format_label_with_units,
    get_variable_label,
    get_variable_units,
)
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_ACCENT

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("scatter")
class ScatterPlotter(BasePlotter):
    """Plotter for scatter comparisons.

    Creates scatter plots comparing model and observation values,
    with options for density coloring, regression lines, and
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
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     show_density=True,
    ...     show_regression=True,
    ... )
    """

    name: str = "scatter"

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
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
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
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
            If True, show 1:1 reference line.
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
        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        # Get data and flatten
        obs_values = paired_data[obs_var].values.flatten()
        model_values = paired_data[model_var].values.flatten()

        # Remove NaN values
        mask = np.isfinite(obs_values) & np.isfinite(model_values)
        obs_values = obs_values[mask]
        model_values = model_values[mask]

        if len(obs_values) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            return fig

        # Calculate limits
        all_values = np.concatenate([obs_values, model_values])
        vmin = self.config.vmin if self.config.vmin is not None else np.nanmin(all_values)
        vmax = self.config.vmax if self.config.vmax is not None else np.nanmax(all_values)

        # Add padding
        padding = (vmax - vmin) * 0.05
        vmin -= padding
        vmax += padding

        # Get style configuration
        style = self.config.style
        ms = marker_size if marker_size is not None else style.markersize
        a = alpha if alpha is not None else (0.5 if len(obs_values) > 1000 else style.alpha)

        # Scatter plot
        if show_density and len(obs_values) > 10:
            # Density-colored scatter
            density = self._calculate_density(obs_values, model_values, density_bins)
            scatter = ax.scatter(
                obs_values,
                model_values,
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
                obs_values,
                model_values,
                c=color_values,
                s=ms**2,
                alpha=a,
                cmap=density_cmap,
            )
            color_label = get_variable_label(paired_data, color_by)
            fig.colorbar(scatter, ax=ax, label=color_label)
        else:
            ax.scatter(
                obs_values,
                model_values,
                c=style.model_color,
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
            self._add_regression_line(ax, obs_values, model_values, vmin, vmax)

        # Statistics annotation
        if show_stats:
            self._add_stats_annotation(ax, obs_values, model_values)

        # Set equal aspect and limits
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")

        # Formatting
        self.apply_text_style(ax)

        # Set labels
        units = get_variable_units(paired_data, obs_var)
        obs_label = format_label_with_units(
            get_variable_label(paired_data, obs_var, self.config.obs_label) or "Observations",
            units,
        )
        model_label = format_label_with_units(
            get_variable_label(paired_data, model_var, self.config.model_label) or "Model",
            units,
        )
        self.set_labels(ax, xlabel=obs_label, ylabel=model_label)

        # Grid
        ax.grid(True, alpha=0.3)

        # Legend
        if show_one_to_one or show_regression:
            self.add_legend(ax, loc="upper left")

        return fig

    def plot_per_flight(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        flight_coord: str = "flight",
        min_points: int = 10,
        **kwargs: Any,
    ) -> Iterator[tuple[str, matplotlib.figure.Figure]]:
        """Generate scatter plots for each flight.

        Yields one figure per unique flight in the data.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
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
            obs_vals = flight_data[obs_var].values.flatten()
            model_vals = flight_data[model_var].values.flatten()
            valid = np.isfinite(obs_vals) & np.isfinite(model_vals)

            if valid.sum() < min_points:
                continue

            # Update title to include flight ID
            original_title = self.config.title
            if original_title:
                self.config.title = f"{original_title} - Flight {flight_str}"
            else:
                self.config.title = f"Flight {flight_str}"

            # Generate plot for this flight
            fig = self.plot(flight_data, obs_var, model_var, **kwargs)

            # Restore original title for next iteration
            self.config.title = original_title

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
        obs: np.ndarray,
        model: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> None:
        """Add regression line to plot.

        Parameters
        ----------
        ax
            Axes to add line to.
        obs, model
            Data arrays.
        vmin, vmax
            Axis limits.
        """
        # Linear regression
        coeffs = np.polyfit(obs, model, 1)
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
        obs: np.ndarray,
        model: np.ndarray,
    ) -> None:
        """Add statistics annotation to plot.

        Parameters
        ----------
        ax
            Axes to annotate.
        obs, model
            Data arrays.
        """
        # Calculate statistics
        n = len(obs)
        mb = np.mean(model - obs)
        rmse = np.sqrt(np.mean((model - obs) ** 2))
        r = np.corrcoef(obs, model)[0, 1]
        nme = np.mean(np.abs(model - obs)) / np.mean(obs) * 100 if np.mean(obs) != 0 else np.nan

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
            fontsize=self.config.text.fontsize - 2,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )


def plot_scatter(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for scatter plotting.

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

    plotter = ScatterPlotter(config=config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
