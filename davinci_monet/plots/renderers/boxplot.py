"""Box plot renderer for DAVINCI.

This module provides box plot functionality for comparing
x and y distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    extract_xy_series,
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


@register_plotter("boxplot")
class BoxPlotter(BasePlotter):
    """Plotter for box plot comparisons.

    Creates box plots showing the distribution of x and
    y values, optionally grouped by categories.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = BoxPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     x_var="x_o3",
    ...     y_var="y_o3",
    ...     group_by="site",
    ... )
    """

    name: str = "boxplot"
    default_figsize: tuple[float, float] = (8, 5)  # Balanced

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a box plot from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one x series and one y series.
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the box plot rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        paired_data, x_var, y_var = extract_xy_series(series, "BoxPlotter.render")

        group_by: str | None = kwargs.pop("group_by", None)
        show_means: bool = kwargs.pop("show_means", True)
        show_outliers: bool = kwargs.pop("show_outliers", True)
        notch: bool = kwargs.pop("notch", False)
        orientation: Literal["vertical", "horizontal"] = kwargs.pop("orientation", "vertical")
        x_label: str | None = kwargs.pop("x_label", None)
        y_label: str | None = kwargs.pop("y_label", None)

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get style configuration
        style = self.config.style

        # Series legend labels prefer the source label over X/Y (R-3).
        x_label = x_label or get_series_label(paired_data, x_var, self.config.x_label)
        y_label = y_label or get_series_label(paired_data, y_var, self.config.y_label)

        vert = orientation == "vertical"

        if group_by is not None and group_by in paired_data.dims:
            # Grouped box plot
            self._plot_grouped(
                ax,
                paired_data,
                x_var,
                y_var,
                group_by,
                x_label,
                y_label,
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
                x_var,
                y_var,
                x_label,
                y_label,
                style,
                show_means,
                show_outliers,
                notch,
                vert,
            )

        # Formatting
        self.apply_text_style(ax)

        # Set labels
        units = get_variable_units(paired_data, x_var)
        value_label = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, x_var) or "Value",
            units,
        )

        if vert:
            self.set_labels(ax, ylabel=value_label)
        else:
            self.set_labels(ax, xlabel=value_label)

        self.set_limits(ax, axis="y" if vert else "x")

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        group_by: str | None = None,
        show_means: bool = True,
        show_outliers: bool = True,
        notch: bool = False,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        x_label: str | None = None,
        y_label: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a box plot.

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
        x_label
            Custom label for the x series.
        y_label
            Custom label for the y series.
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
            group_by=group_by,
            show_means=show_means,
            show_outliers=show_outliers,
            notch=notch,
            orientation=orientation,
            x_label=x_label,
            y_label=y_label,
            **kwargs,
        )

    def _plot_simple(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        x_label: str,
        y_label: str,
        style: Any,
        show_means: bool,
        show_outliers: bool,
        notch: bool,
        vert: bool,
    ) -> None:
        """Plot simple x vs y comparison.

        Parameters
        ----------
        ax
            Axes to plot on.
        paired_data
            Paired dataset.
        x_var, y_var
            Variable names.
        x_label, y_label
            Labels.
        style
            Style configuration.
        show_means, show_outliers, notch, vert
            Plot options.
        """
        x_values = paired_data[x_var].values.flatten()
        y_values = paired_data[y_var].values.flatten()

        # Remove NaN values
        x_values = x_values[np.isfinite(x_values)]
        y_values = y_values[np.isfinite(y_values)]

        data = [x_values, y_values]
        labels = [x_label, y_label]
        colors = [
            get_axis_color(
                paired_data,
                x_var,
                0,
                x_color=style.x_color,
                y_color=style.y_color,
            ),
            get_axis_color(
                paired_data,
                y_var,
                1,
                x_color=style.x_color,
                y_color=style.y_color,
            ),
        ]

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
        x_var: str,
        y_var: str,
        group_by: str,
        x_label: str,
        y_label: str,
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
        x_var, y_var
            Variable names.
        group_by
            Dimension to group by.
        x_label, y_label
            Labels.
        style
            Style configuration.
        show_means, show_outliers, notch, vert
            Plot options.
        """
        groups = paired_data[group_by].values
        n_groups = len(groups)

        # Collect data for each group
        x_data = []
        y_data = []

        for i in range(n_groups):
            x_vals = paired_data[x_var].isel({group_by: i}).values.flatten()
            y_vals = paired_data[y_var].isel({group_by: i}).values.flatten()

            x_vals = x_vals[np.isfinite(x_vals)]
            y_vals = y_vals[np.isfinite(y_vals)]

            x_data.append(x_vals)
            y_data.append(y_vals)

        # Calculate positions
        width = 0.35
        positions_x = np.arange(n_groups) - width / 2
        positions_y = np.arange(n_groups) + width / 2

        # Plot x boxes
        bp_x = ax.boxplot(
            x_data,
            positions=positions_x,
            widths=width * 0.8,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Plot y boxes
        bp_y = ax.boxplot(
            y_data,
            positions=positions_y,
            widths=width * 0.8,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Color the boxes by source axis (R-3)
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
        for patch in bp_x["boxes"]:
            patch.set_facecolor(x_color)
            patch.set_alpha(0.5)
        for patch in bp_y["boxes"]:
            patch.set_facecolor(y_color)
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
            Patch(facecolor=x_color, alpha=0.5, label=x_label),
            Patch(facecolor=y_color, alpha=0.5, label=y_label),
        ]
        ax.legend(handles=legend_elements, loc="best")


def plot_boxplot(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for box plotting.

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

    plotter = BoxPlotter(config=config)
    return plotter.plot(paired_data, x_var, y_var, **kwargs)
