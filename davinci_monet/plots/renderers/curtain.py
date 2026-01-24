"""Curtain plot renderer for DAVINCI-MONET.

This module provides curtain plot functionality for visualizing
vertical cross-sections of aircraft or model data along a trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    calculate_symmetric_limits,
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


@register_plotter("curtain")
class CurtainPlotter(BasePlotter):
    """Plotter for vertical curtain plots.

    Creates 2D plots showing values along a trajectory (time × altitude),
    useful for aircraft data and vertical cross-sections.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = CurtainPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     alt_var="altitude",
    ... )
    """

    name: str = "curtain"
    default_figsize: tuple[float, float] = (14, 8)  # Wide for geographic extent

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        alt_var: str = "altitude",
        time_dim: str = "time",
        show_var: Literal["obs", "model", "bias"] = "bias",
        cmap: str | None = None,
        n_levels: int = 20,
        show_scatter: bool = True,
        scatter_size: float | None = None,
        invert_yaxis: bool = False,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a curtain plot.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Optional axes to plot on.
        alt_var
            Name of altitude coordinate/variable.
        time_dim
            Name of time dimension.
        show_var
            Which variable to show ('obs', 'model', 'bias').
        cmap
            Colormap. Defaults to RdBu_r for bias, viridis otherwise.
        n_levels
            Number of contour levels.
        show_scatter
            If True, overlay observation points as scatter.
        scatter_size
            Size of scatter points.
        invert_yaxis
            If True, invert y-axis (for pressure coordinates).
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

        # Get data
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Calculate bias if needed
        if show_var == "bias":
            plot_data = model_data - obs_data
            default_cmap = "RdBu_r"
        elif show_var == "obs":
            plot_data = obs_data
            default_cmap = "viridis"
        else:
            plot_data = model_data
            default_cmap = "viridis"

        cmap = cmap or default_cmap

        # Get coordinates
        time = paired_data[time_dim]
        if alt_var in paired_data.coords:
            altitude = paired_data[alt_var]
        elif alt_var in paired_data:
            altitude = paired_data[alt_var]
        else:
            raise ValueError(f"Could not find altitude variable {alt_var}")

        # Get values
        time_values = time.values
        alt_values = altitude.values
        data_values = plot_data.values

        # Handle different data shapes
        if data_values.ndim == 1:
            # 1D trajectory data - create 2D representation
            # This is a simple approach; more sophisticated binning could be used
            self._plot_1d_trajectory(
                ax, time_values, alt_values, data_values,
                obs_data.values if show_scatter else None,
                cmap, show_var, scatter_size,
            )
        else:
            # 2D or higher - use contourf
            self._plot_2d_curtain(
                ax, time_values, alt_values, data_values,
                cmap, n_levels, show_var,
            )

        # Calculate limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(data_values)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            data_flat = data_values[np.isfinite(data_values)]
            vmin = self.config.vmin if self.config.vmin is not None else np.nanpercentile(data_flat, 2)
            vmax = self.config.vmax if self.config.vmax is not None else np.nanpercentile(data_flat, 98)
            norm = None

        # Formatting
        self.apply_text_style(ax)

        # Labels
        units = get_variable_units(paired_data, obs_var)

        if show_var == "bias":
            ylabel_text = "Bias (Model - Obs)"
        else:
            ylabel_text = get_variable_label(paired_data, obs_var if show_var == "obs" else model_var)

        alt_units = get_variable_units(paired_data, alt_var) or "m"
        self.set_labels(
            ax,
            xlabel="Time",
            ylabel=f"Altitude ({alt_units})",
        )

        # Title
        if self.config.title:
            ax.set_title(format_plot_title(self.config.title), fontsize=self.config.text.title_fontsize)
        else:
            var_label = get_variable_label(paired_data, obs_var)
            ax.set_title(
                format_plot_title(f"{var_label} Curtain ({show_var.title()})"),
                fontsize=self.config.text.title_fontsize,
            )

        # Invert y-axis if needed (e.g., for pressure)
        if invert_yaxis:
            ax.invert_yaxis()

        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

        return fig

    def _plot_1d_trajectory(
        self,
        ax: matplotlib.axes.Axes,
        time_values: np.ndarray,
        alt_values: np.ndarray,
        data_values: np.ndarray,
        obs_values: np.ndarray | None,
        cmap: str,
        show_var: str,
        scatter_size: float | None,
    ) -> None:
        """Plot 1D trajectory data as colored scatter.

        Parameters
        ----------
        ax
            Axes to plot on.
        time_values
            Time coordinates.
        alt_values
            Altitude values.
        data_values
            Data to color by.
        obs_values
            Optional observation values for scatter.
        cmap
            Colormap.
        show_var
            Which variable is shown.
        scatter_size
            Scatter point size.
        """
        import pandas as pd

        # Convert time to numeric for plotting
        time_numeric = pd.to_datetime(time_values)

        # Remove NaN
        mask = np.isfinite(data_values) & np.isfinite(alt_values)
        time_plot = time_numeric[mask]
        alt_plot = alt_values[mask]
        data_plot = data_values[mask]

        # Calculate limits
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(data_plot)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            vmin = np.nanpercentile(data_plot, 2)
            vmax = np.nanpercentile(data_plot, 98)
            norm = None

        # Scatter plot
        style = self.config.style
        ms = scatter_size if scatter_size is not None else style.markersize * 1.5

        scatter = ax.scatter(
            time_plot,
            alt_plot,
            c=data_plot,
            s=ms**2,
            cmap=cmap,
            norm=norm,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            alpha=style.alpha,
            edgecolors="none",
        )

        # Add colorbar
        units = None  # Would need to pass this in
        cbar = ax.get_figure().colorbar(scatter, ax=ax)
        if show_var == "bias":
            cbar.set_label("Bias (Model - Obs)")
        else:
            cbar.set_label(show_var.title())

    def _plot_2d_curtain(
        self,
        ax: matplotlib.axes.Axes,
        time_values: np.ndarray,
        alt_values: np.ndarray,
        data_values: np.ndarray,
        cmap: str,
        n_levels: int,
        show_var: str,
    ) -> None:
        """Plot 2D curtain data as contourf.

        Parameters
        ----------
        ax
            Axes to plot on.
        time_values
            Time coordinates.
        alt_values
            Altitude coordinates.
        data_values
            2D data array.
        cmap
            Colormap.
        n_levels
            Number of contour levels.
        show_var
            Which variable is shown.
        """
        import pandas as pd

        # Convert time
        time_numeric = pd.to_datetime(time_values)

        # Create meshgrid if needed
        if alt_values.ndim == 1:
            time_grid, alt_grid = np.meshgrid(
                np.arange(len(time_numeric)),
                alt_values,
            )
            # Transpose data if needed
            if data_values.shape[0] == len(time_numeric):
                data_values = data_values.T
        else:
            time_grid = np.broadcast_to(
                np.arange(len(time_numeric)),
                alt_values.shape,
            )
            alt_grid = alt_values

        # Calculate limits
        data_flat = data_values[np.isfinite(data_values)]
        if show_var == "bias":
            vmin, vmax = calculate_symmetric_limits(data_flat)
        else:
            vmin = np.nanpercentile(data_flat, 2)
            vmax = np.nanpercentile(data_flat, 98)

        levels = np.linspace(vmin, vmax, n_levels)

        # Contourf plot
        contour = ax.contourf(
            time_grid,
            alt_grid,
            data_values,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

        # Set x-axis to show times
        ax.set_xticks(np.linspace(0, len(time_numeric) - 1, 5).astype(int))
        ax.set_xticklabels([
            time_numeric[int(i)].strftime("%H:%M")
            for i in np.linspace(0, len(time_numeric) - 1, 5)
        ])

        # Add colorbar
        cbar = ax.get_figure().colorbar(contour, ax=ax)
        if show_var == "bias":
            cbar.set_label("Bias (Model - Obs)")
        else:
            cbar.set_label(show_var.title())


def plot_curtain(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for curtain plotting.

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

    plotter = CurtainPlotter(config=config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
