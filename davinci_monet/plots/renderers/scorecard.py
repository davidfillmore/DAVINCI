"""Scorecard plot renderer for DAVINCI.

This module provides scorecard/heatmap plotting functionality for
displaying statistics across multiple variables, sites, or datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots._stats import annotation_metrics
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    canonical_variable_name,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import pandas as pd
    import xarray as xr


@register_plotter("scorecard")
class ScorecardPlotter(BasePlotter):
    """Plotter for scorecard/heatmap displays.

    Creates heatmap-style scorecards showing statistics across
    multiple categories (variables, sites, datasets, etc.).

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = ScorecardPlotter()
    >>> fig = plotter.plot_from_stats(
    ...     stats_df,
    ...     row_var="variable",
    ...     col_var="dataset",
    ...     value_var="correlation",
    ... )
    """

    name: str = "scorecard"
    default_figsize: tuple[float, float] = (8, 5)  # Balanced

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a scorecard from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one geometry (geometry) and one dataset (dataset).
        ax
            Optional axes to plot on. If None, creates new figure.
        **kwargs
            Forwarded to plot_from_dataframe.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"ScorecardPlotter.render requires exactly 2 series; got {len(series)}."
            )
        x_series = next((s for s in series if s.axis == "x"), series[0])
        y_series = next((s for s in series if s.axis == "y"), series[1])
        paired_data = x_series.dataset
        x_var = x_series.var_name
        y_var = y_series.var_name

        # Calculate basic statistics (via central metric registry)
        geometry = paired_data[x_var].values.flatten()
        dataset = paired_data[y_var].values.flatten()

        registry_stats = annotation_metrics(geometry, dataset, ["N", "MG", "MD", "MB", "RMSE", "R"])
        stats = {
            "N": int(registry_stats["N"]),
            "Mean Geometry": registry_stats["MG"],
            "Mean Dataset": registry_stats["MD"],
            "MB": registry_stats["MB"],
            "RMSE": registry_stats["RMSE"],
            "R": registry_stats["R"],
        }

        # Create simple 1-row scorecard
        import pandas as pd

        stats_df = pd.DataFrame([stats])
        stats_df.index = [canonical_variable_name(paired_data, x_var)]

        return self.plot_from_dataframe(stats_df, ax=ax, **kwargs)

    def plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a scorecard from paired data.

        This method calculates statistics from paired data and creates
        a scorecard. For pre-computed statistics, use plot_from_stats().

        Parameters
        ----------
        paired_data
            Paired dataset with geometry and dataset variables.
        x_var
            Compatibility name for geometry variable.
        y_var
            Compatibility name for dataset variable.
        ax
            Optional axes to plot on.
        **kwargs
            Additional arguments passed to plot_from_dataframe.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, x_var, y_var),
            ax=ax,
            **kwargs,
        )

    def plot_from_dataframe(
        self,
        stats_df: pd.DataFrame,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "RdYlGn",
        annot: bool = True,
        fmt: str = ".2f",
        center: float | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        cbar: bool = True,
        cbar_label: str | None = None,
        row_colors: dict[str, str] | None = None,
        col_colors: dict[str, str] | None = None,
        highlight_best: bool = False,
        highlight_axis: Literal["row", "col"] = "col",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a scorecard from a DataFrame.

        Parameters
        ----------
        stats_df
            DataFrame with statistics. Rows and columns become
            the scorecard axes.
        ax
            Optional axes to plot on.
        cmap
            Colormap name.
        annot
            If True, annotate cells with values.
        fmt
            Format string for annotations.
        center
            Value to center colormap on.
        vmin, vmax
            Value limits for colormap.
        cbar
            If True, show colorbar.
        cbar_label
            Label for colorbar.
        row_colors
            Dict mapping row labels to colors for sidebar.
        col_colors
            Dict mapping column labels to colors for topbar.
        highlight_best
            If True, highlight best value in each row/col.
        highlight_axis
            Axis to highlight best along ('row' or 'col').
        **kwargs
            Additional arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=self.config.figure.dpi)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Convert to numpy for plotting
        data = stats_df.to_numpy(dtype=float, copy=False)
        row_labels = list(stats_df.index)
        col_labels = list(stats_df.columns)

        # Calculate limits
        data_finite = data[np.isfinite(data)]
        if len(data_finite) == 0:
            vmin_calc, vmax_calc = 0, 1
        else:
            vmin_calc = np.nanmin(data_finite) if vmin is None else vmin  # type: ignore[assignment]
            vmax_calc = np.nanmax(data_finite) if vmax is None else vmax  # type: ignore[assignment]

        # Create heatmap
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            vmin=vmin_calc,
            vmax=vmax_calc,
        )

        # Set ticks
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, fontsize=self.config.text.tick_fontsize)
        ax.set_yticklabels(row_labels, fontsize=self.config.text.tick_fontsize)

        # Rotate column labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add annotations
        if annot:
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    value = data[i, j]
                    if np.isfinite(value):
                        # Determine text color based on background
                        bg_val = (value - vmin_calc) / (vmax_calc - vmin_calc)
                        text_color = "white" if 0.3 < bg_val < 0.7 else "black"

                        text = format(value, fmt)
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=self.config.text.fontsize - 2,
                        )

        # Highlight best values
        if highlight_best:
            self._highlight_best(ax, data, highlight_axis)

        # Add colorbar
        if cbar:
            cbar_obj = fig.colorbar(im, ax=ax, shrink=0.8)
            if cbar_label:
                cbar_obj.set_label(cbar_label, fontsize=self.config.text.fontsize)

        # Title
        if self.config.title:
            self.set_title(ax, self.config.title)

        plt.tight_layout()
        return fig

    def _highlight_best(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        axis: str,
    ) -> None:
        """Highlight best values in each row or column.

        Parameters
        ----------
        ax
            Axes to add highlights to.
        data
            Data array.
        axis
            'row' to highlight best in each row, 'col' for each column.
        """
        from matplotlib.patches import Rectangle

        if axis == "row":
            # Best (max) in each row
            for i in range(data.shape[0]):
                row = data[i, :]
                if np.any(np.isfinite(row)):
                    best_j = int(np.nanargmax(row))
                    rect = Rectangle(
                        (best_j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="gold",
                        linewidth=2,
                    )
                    ax.add_patch(rect)
        else:
            # Best (max) in each column
            for j in range(data.shape[1]):
                col = data[:, j]
                if np.any(np.isfinite(col)):
                    best_i = int(np.nanargmax(col))
                    rect = Rectangle(
                        (j - 0.5, best_i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="gold",
                        linewidth=2,
                    )
                    ax.add_patch(rect)

    def plot_multi_metric(
        self,
        stats_dict: dict[str, pd.DataFrame],
        metrics: Sequence[str],
        ax: matplotlib.axes.Axes | None = None,
        cmaps: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a multi-metric scorecard.

        Creates a scorecard with multiple metrics shown as subplots.

        Parameters
        ----------
        stats_dict
            Dict mapping dataset names to DataFrames of statistics.
        metrics
            List of metric names (columns) to show.
        ax
            Optional axes (will create subplots if None).
        cmaps
            Optional dict mapping metric names to colormaps.
        **kwargs
            Additional arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import pandas as pd

        n_metrics = len(metrics)

        # Create subplots with standard size
        fig, axes = plt.subplots(
            1,
            n_metrics,
            figsize=(8, 5),
            dpi=self.config.figure.dpi,
            squeeze=False,
        )
        axes = axes.flatten()

        # Default colormaps for common metrics
        default_cmaps = {
            "R": "RdYlGn",
            "correlation": "RdYlGn",
            "MB": "RdBu_r",
            "bias": "RdBu_r",
            "RMSE": "RdYlGn_r",
            "NME": "RdYlGn_r",
            "NMB": "RdBu_r",
        }
        cmaps = cmaps or {}

        for i, metric in enumerate(metrics):
            # Combine data from all datasets
            data = {}
            for y_name, df in stats_dict.items():
                if metric in df.columns:
                    data[y_name] = df[metric]

            if not data:
                continue

            metric_df = pd.DataFrame(data)

            # Get colormap
            cmap = cmaps.get(metric, default_cmaps.get(metric, "viridis"))

            # Set center for bias metrics
            center = 0 if "bias" in metric.lower() or metric in ["MB", "NMB"] else None

            self.plot_from_dataframe(
                metric_df,
                ax=axes[i],
                cmap=cmap,
                center=center,
                cbar_label=metric,
                **kwargs,
            )
            axes[i].set_title(metric, fontsize=self.config.text.title_fontsize)

        plt.tight_layout()
        return fig


def plot_scorecard(
    paired_data: xr.Dataset,
    x_var: str,
    y_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for scorecard plotting.

    Parameters
    ----------
    paired_data
        Paired dataset with dataset and dataset variables.
    x_var
        Name of dataset variable.
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

    plotter = ScorecardPlotter(config=config)
    return plotter.plot(paired_data, x_var, y_var, **kwargs)
