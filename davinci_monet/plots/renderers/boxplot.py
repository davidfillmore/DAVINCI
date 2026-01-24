"""Box plot renderer for DAVINCI-MONET.

This module provides box plot functionality for comparing
model and observation distributions.
"""

from __future__ import annotations

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


@register_plotter("boxplot")
class BoxPlotter(BasePlotter):
    """Plotter for box plot comparisons.

    Creates box plots showing the distribution of model and
    observation values, optionally grouped by categories.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = BoxPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     group_by="site",
    ... )
    """

    name: str = "boxplot"
    default_figsize: tuple[float, float] = (8, 5)  # Balanced

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        group_by: str | None = None,
        show_means: bool = True,
        show_outliers: bool = True,
        notch: bool = False,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        obs_label: str | None = None,
        model_label: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a box plot.

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
        group_by
            Optional dimension/coordinate to group data by.
        show_means
            If True, show mean markers on boxes.
        show_outliers
            If True, show outlier points.
        notch
            If True, use notched box style.
        orientation
            Box orientation ('vertical' or 'horizontal').
        obs_label
            Custom label for observations.
        model_label
            Custom label for model.
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

        # Get style configuration
        style = self.config.style

        # Get labels
        obs_label = obs_label or get_variable_label(
            paired_data, obs_var, self.config.obs_label
        ) or "Obs"
        model_label = model_label or get_variable_label(
            paired_data, model_var, self.config.model_label
        ) or "Model"

        vert = orientation == "vertical"

        if group_by is not None and group_by in paired_data.dims:
            # Grouped box plot
            self._plot_grouped(
                ax,
                paired_data,
                obs_var,
                model_var,
                group_by,
                obs_label,
                model_label,
                style,
                show_means,
                show_outliers,
                notch,
                vert,
            )
        else:
            # Simple comparison
            self._plot_simple(
                ax,
                paired_data,
                obs_var,
                model_var,
                obs_label,
                model_label,
                style,
                show_means,
                show_outliers,
                notch,
                vert,
            )

        # Formatting
        self.apply_text_style(ax)

        # Set labels
        units = get_variable_units(paired_data, obs_var)
        value_label = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, obs_var) or "Value",
            units,
        )

        if vert:
            self.set_labels(ax, ylabel=value_label)
        else:
            self.set_labels(ax, xlabel=value_label)

        self.set_limits(ax, axis="y" if vert else "x")

        return fig

    def _plot_simple(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        obs_label: str,
        model_label: str,
        style: Any,
        show_means: bool,
        show_outliers: bool,
        notch: bool,
        vert: bool,
    ) -> None:
        """Plot simple obs vs model comparison.

        Parameters
        ----------
        ax
            Axes to plot on.
        paired_data
            Paired dataset.
        obs_var, model_var
            Variable names.
        obs_label, model_label
            Labels.
        style
            Style configuration.
        show_means, show_outliers, notch, vert
            Plot options.
        """
        obs_values = paired_data[obs_var].values.flatten()
        model_values = paired_data[model_var].values.flatten()

        # Remove NaN values
        obs_values = obs_values[np.isfinite(obs_values)]
        model_values = model_values[np.isfinite(model_values)]

        data = [obs_values, model_values]
        labels = [obs_label, model_label]
        colors = [style.obs_color, style.model_color]

        bp = ax.boxplot(
            data,
            labels=labels,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        for median in bp["medians"]:
            median.set_color("black")

    def _plot_grouped(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        group_by: str,
        obs_label: str,
        model_label: str,
        style: Any,
        show_means: bool,
        show_outliers: bool,
        notch: bool,
        vert: bool,
    ) -> None:
        """Plot grouped box plot.

        Parameters
        ----------
        ax
            Axes to plot on.
        paired_data
            Paired dataset.
        obs_var, model_var
            Variable names.
        group_by
            Dimension to group by.
        obs_label, model_label
            Labels.
        style
            Style configuration.
        show_means, show_outliers, notch, vert
            Plot options.
        """
        groups = paired_data[group_by].values
        n_groups = len(groups)

        # Collect data for each group
        obs_data = []
        model_data = []

        for i in range(n_groups):
            obs_vals = paired_data[obs_var].isel({group_by: i}).values.flatten()
            model_vals = paired_data[model_var].isel({group_by: i}).values.flatten()

            obs_vals = obs_vals[np.isfinite(obs_vals)]
            model_vals = model_vals[np.isfinite(model_vals)]

            obs_data.append(obs_vals)
            model_data.append(model_vals)

        # Calculate positions
        width = 0.35
        positions_obs = np.arange(n_groups) - width / 2
        positions_model = np.arange(n_groups) + width / 2

        # Plot observation boxes
        bp_obs = ax.boxplot(
            obs_data,
            positions=positions_obs,
            widths=width * 0.8,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Plot model boxes
        bp_model = ax.boxplot(
            model_data,
            positions=positions_model,
            widths=width * 0.8,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Color the boxes
        for patch in bp_obs["boxes"]:
            patch.set_facecolor(style.obs_color)
            patch.set_alpha(0.5)
        for patch in bp_model["boxes"]:
            patch.set_facecolor(style.model_color)
            patch.set_alpha(0.5)

        # Set tick labels
        if vert:
            ax.set_xticks(np.arange(n_groups))
            ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
        else:
            ax.set_yticks(np.arange(n_groups))
            ax.set_yticklabels([str(g) for g in groups])

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=style.obs_color, alpha=0.5, label=obs_label),
            Patch(facecolor=style.model_color, alpha=0.5, label=model_label),
        ]
        ax.legend(handles=legend_elements, loc="best")


def plot_boxplot(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for box plotting.

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

    plotter = BoxPlotter(config=config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
