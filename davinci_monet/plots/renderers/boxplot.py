"""Box plot renderer for DAVINCI.

This module provides box plot functionality for comparing
dataset and dataset distributions.
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
    format_label_with_units,
    get_dataset_color,
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

    Creates box plots showing the distribution of dataset and
    dataset values, optionally grouped by categories.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = BoxPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     geometry_var="geometry_o3",
    ...     dataset_var="dataset_o3",
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
            Exactly 2 series: one geometry (geometry) and one dataset (dataset).
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to the box plot rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"BoxPlotter.render requires exactly 2 series; got {len(series)}."
            )
        geometry_series = next((s for s in series if s.pair_axis == "geometry"), series[0])
        dataset_series = next((s for s in series if s.pair_axis == "dataset"), series[1])
        paired_data = geometry_series.dataset
        geometry_var = geometry_series.var_name
        dataset_var = dataset_series.var_name

        group_by: str | None = kwargs.pop("group_by", None)
        show_means: bool = kwargs.pop("show_means", True)
        show_outliers: bool = kwargs.pop("show_outliers", True)
        notch: bool = kwargs.pop("notch", False)
        orientation: Literal["vertical", "horizontal"] = kwargs.pop("orientation", "vertical")
        geometry_label: str | None = kwargs.pop("geometry_label", None)
        dataset_label: str | None = kwargs.pop("dataset_label", None)

        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get style configuration
        style = self.config.style

        # Series legend labels prefer the source label over Geometry/Dataset (R-3).
        geometry_label = geometry_label or get_series_label(
            paired_data, geometry_var, self.config.geometry_label
        )
        dataset_label = dataset_label or get_series_label(
            paired_data, dataset_var, self.config.dataset_label
        )

        vert = orientation == "vertical"

        if group_by is not None and group_by in paired_data.dims:
            # Grouped box plot
            self._plot_grouped(
                ax,
                paired_data,
                geometry_var,
                dataset_var,
                group_by,
                geometry_label,
                dataset_label,
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
                geometry_var,
                dataset_var,
                geometry_label,
                dataset_label,
                style,
                show_means,
                show_outliers,
                notch,
                vert,
            )

        # Formatting
        self.apply_text_style(ax)

        # Set labels
        units = get_variable_units(paired_data, geometry_var)
        value_label = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, geometry_var) or "Value",
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
        geometry_var: str,
        dataset_var: str,
        ax: matplotlib.axes.Axes | None = None,
        group_by: str | None = None,
        show_means: bool = True,
        show_outliers: bool = True,
        notch: bool = False,
        orientation: Literal["vertical", "horizontal"] = "vertical",
        geometry_label: str | None = None,
        dataset_label: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a box plot.

        Parameters
        ----------
        paired_data
            Paired dataset with dataset and dataset variables.
        geometry_var
            Name of dataset variable.
        dataset_var
            Name of dataset variable.
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
        geometry_label
            Custom label for datasets.
        dataset_label
            Custom label for dataset.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, geometry_var, dataset_var),
            ax=ax,
            group_by=group_by,
            show_means=show_means,
            show_outliers=show_outliers,
            notch=notch,
            orientation=orientation,
            geometry_label=geometry_label,
            dataset_label=dataset_label,
            **kwargs,
        )

    def _plot_simple(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        geometry_var: str,
        dataset_var: str,
        geometry_label: str,
        dataset_label: str,
        style: Any,
        show_means: bool,
        show_outliers: bool,
        notch: bool,
        vert: bool,
    ) -> None:
        """Plot simple geometry vs dataset comparison.

        Parameters
        ----------
        ax
            Axes to plot on.
        paired_data
            Paired dataset.
        geometry_var, dataset_var
            Variable names.
        geometry_label, dataset_label
            Labels.
        style
            Style configuration.
        show_means, show_outliers, notch, vert
            Plot options.
        """
        geometry_values = paired_data[geometry_var].values.flatten()
        dataset_values = paired_data[dataset_var].values.flatten()

        # Remove NaN values
        geometry_values = geometry_values[np.isfinite(geometry_values)]
        dataset_values = dataset_values[np.isfinite(dataset_values)]

        data = [geometry_values, dataset_values]
        labels = [geometry_label, dataset_label]
        colors = [
            get_dataset_color(
                paired_data,
                geometry_var,
                0,
                geometry_color=style.geometry_color,
                dataset_color=style.dataset_color,
            ),
            get_dataset_color(
                paired_data,
                dataset_var,
                1,
                geometry_color=style.geometry_color,
                dataset_color=style.dataset_color,
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
        geometry_var: str,
        dataset_var: str,
        group_by: str,
        geometry_label: str,
        dataset_label: str,
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
        geometry_var, dataset_var
            Variable names.
        group_by
            Dimension to group by.
        geometry_label, dataset_label
            Labels.
        style
            Style configuration.
        show_means, show_outliers, notch, vert
            Plot options.
        """
        groups = paired_data[group_by].values
        n_groups = len(groups)

        # Collect data for each group
        geometry_data = []
        dataset_data = []

        for i in range(n_groups):
            geometry_vals = paired_data[geometry_var].isel({group_by: i}).values.flatten()
            dataset_vals = paired_data[dataset_var].isel({group_by: i}).values.flatten()

            geometry_vals = geometry_vals[np.isfinite(geometry_vals)]
            dataset_vals = dataset_vals[np.isfinite(dataset_vals)]

            geometry_data.append(geometry_vals)
            dataset_data.append(dataset_vals)

        # Calculate positions
        width = 0.35
        positions_geometry = np.arange(n_groups) - width / 2
        positions_dataset = np.arange(n_groups) + width / 2

        # Plot dataset boxes
        bp_geometry = ax.boxplot(
            geometry_data,
            positions=positions_geometry,
            widths=width * 0.8,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Plot dataset boxes
        bp_dataset = ax.boxplot(
            dataset_data,
            positions=positions_dataset,
            widths=width * 0.8,
            notch=notch,
            vert=vert,
            showmeans=show_means,
            showfliers=show_outliers,
            patch_artist=True,
        )

        # Color the boxes by source axis (R-3)
        geometry_color = get_dataset_color(
            paired_data,
            geometry_var,
            0,
            geometry_color=style.geometry_color,
            dataset_color=style.dataset_color,
        )
        dataset_color = get_dataset_color(
            paired_data,
            dataset_var,
            1,
            geometry_color=style.geometry_color,
            dataset_color=style.dataset_color,
        )
        for patch in bp_geometry["boxes"]:
            patch.set_facecolor(geometry_color)
            patch.set_alpha(0.5)
        for patch in bp_dataset["boxes"]:
            patch.set_facecolor(dataset_color)
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
            Patch(facecolor=geometry_color, alpha=0.5, label=geometry_label),
            Patch(facecolor=dataset_color, alpha=0.5, label=dataset_label),
        ]
        ax.legend(handles=legend_elements, loc="best")


def plot_boxplot(
    paired_data: xr.Dataset,
    geometry_var: str,
    dataset_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for box plotting.

    Parameters
    ----------
    paired_data
        Paired dataset with dataset and dataset variables.
    geometry_var
        Name of dataset variable.
    dataset_var
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

    plotter = BoxPlotter(config=config)
    return plotter.plot(paired_data, geometry_var, dataset_var, **kwargs)
