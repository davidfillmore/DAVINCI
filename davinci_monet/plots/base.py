"""Base plotter classes and common utilities for DAVINCI.

This module provides:
- PlotConfig: Pydantic dataset for plot configuration
- BasePlotter: Abstract base class for all plotters
- Common utilities for figure creation, styling, and saving

Implementation note
-------------------
The config dataclasses live in ``plots.plot_config``, the text/label/unit
formatting functions and lookup tables live in ``plots.labels``, and the
series/color helpers live in ``plots.series``. Everything is re-exported here
for renderers, tests, pipeline stages, and examples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt

# Import NCAR style colors for defaults
from davinci_monet.core.base import PlotSeries

# Re-export label/formatting utilities
from davinci_monet.plots.labels import (
    TITLE_FORMULA_REPLACEMENTS,
    UNIT_REPLACEMENTS,
    VARIABLE_DISPLAY_NAMES,
    calculate_data_limits,
    calculate_symmetric_limits,
    canonical_variable_name,
    format_label_with_units,
    format_plot_title,
    format_units,
    format_variable_display_name,
    get_variable_label,
    get_variable_units,
    merge_config_dicts,
)

# Re-export config dataclasses
from davinci_monet.plots.plot_config import (
    DomainConfig,
    FigureConfig,
    PlotConfig,
    StyleConfig,
    TextConfig,
)

# Re-export series/color helpers
from davinci_monet.plots.series import (
    build_series,
    dataset_label,
    get_dataset_color,
    get_series_label,
    resolve_dataset_variable,
    series_colors,
)

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


# =============================================================================
# Base Plotter
# =============================================================================


class BasePlotter(ABC):
    """Abstract base class for all plotters.

    Provides common functionality for figure management, styling,
    and saving. Subclasses implement specific plot types.

    Parameters
    ----------
    config
        Plot configuration. If None, uses defaults.

    Attributes
    ----------
    config : PlotConfig
        The plot configuration.
    default_figsize : tuple[float, float]
        Default figure size for this plotter type. Subclasses can override.
    """

    # Class-level registry name (override in subclasses)
    name: str = "base"

    # Default figure size - subclasses can override for optimal sizing
    default_figsize: tuple[float, float] = (8, 5)

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()
        # Apply plotter-specific default figsize if not explicitly set
        if self.config.figure.figsize == (8, 5):  # Original default
            self.config.figure.figsize = self.default_figsize

    @abstractmethod
    def plot(
        self,
        paired_data: xr.Dataset,
        geometry_var: str,
        dataset_var: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate the plot.

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
        **kwargs
            Additional plot-specific options.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        ...

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a list of source series (unified renderer contract).

        ``len(series) == 1`` -> single line; ``== 2`` -> geometry-vs-dataset
        (geometry gray / dataset blue); ``>= 2`` -> multi-source overlay. This
        default handles only the 2-series case by delegating to
        ``plot(paired_data, geometry_var, dataset_var)``. Renderers that support
        1 or N series override this method.
        """
        if len(series) == 2:
            geometry_series = next((s for s in series if s.pair_axis == "geometry"), series[0])
            dataset_series = next((s for s in series if s.pair_axis == "dataset"), series[1])
            return self.plot(
                geometry_series.dataset,
                geometry_series.var_name,
                dataset_series.var_name,
                ax=ax,
                **kwargs,
            )
        raise NotImplementedError(
            f"{type(self).__name__}.render does not support {len(series)} series; "
            "override render() for single-/N-source support (unification P3)."
        )

    def create_figure(
        self,
        config: FigureConfig | None = None,
        **kwargs: Any,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a new figure and axes.

        Parameters
        ----------
        config
            Figure configuration. If None, uses self.config.figure.
        **kwargs
            Additional arguments passed to plt.subplots.

        Returns
        -------
        tuple[Figure, Axes]
            The created figure and axes.
        """
        cfg = config or self.config.figure
        fig_kwargs = {
            "figsize": cfg.figsize,
            "dpi": cfg.dpi,
            "facecolor": cfg.facecolor,
            "constrained_layout": cfg.constrained_layout,
            **kwargs,
        }
        fig, ax = plt.subplots(**fig_kwargs)
        return fig, ax

    def apply_text_style(
        self,
        ax: matplotlib.axes.Axes,
        config: TextConfig | None = None,
    ) -> None:
        """Apply text styling to axes.

        Parameters
        ----------
        ax
            Axes to style.
        config
            Text configuration. If None, uses self.config.text.
        """
        cfg = config or self.config.text
        ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)

    def set_labels(
        self,
        ax: matplotlib.axes.Axes,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
    ) -> None:
        """Set axis labels and title.

        Parameters
        ----------
        ax
            Axes to label.
        xlabel
            X-axis label. Uses config if None.
        ylabel
            Y-axis label. Uses config if None.
        title
            Plot title. Uses config if None.
        """
        cfg = self.config.text

        if xlabel or self.config.xlabel:
            ax.set_xlabel(
                xlabel or self.config.xlabel,  # type: ignore[arg-type]
                fontsize=cfg.fontsize,
                fontweight=cfg.fontweight,
            )

        if ylabel or self.config.ylabel:
            ax.set_ylabel(
                ylabel or self.config.ylabel,  # type: ignore[arg-type]
                fontsize=cfg.fontsize,
                fontweight=cfg.fontweight,
            )

        self.set_title(ax, title=title)

    def set_title(
        self,
        ax: matplotlib.axes.Axes,
        title: str | None = None,
        subtitle: str | None = None,
        *,
        fontsize: float | None = None,
    ) -> None:
        """Set an axes title with an optional smaller subtitle below it."""
        cfg = self.config.text
        title_text = title or self.config.title
        subtitle_text = subtitle if subtitle is not None else self.config.subtitle

        if title or self.config.title:
            formatted_title = format_plot_title(title_text)  # type: ignore[arg-type]
            ax.set_title(
                formatted_title,
                fontsize=fontsize or cfg.title_fontsize,
                fontweight=cfg.fontweight,
                pad=22 if subtitle_text else None,
                wrap=True,
            )
        if subtitle_text:
            ax.text(
                0.5,
                1.01,
                subtitle_text,
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=cfg.annotation_small,
                color="#58595B",
                clip_on=False,
            )

    def set_figure_title(
        self,
        fig: matplotlib.figure.Figure,
        title: str | None = None,
        subtitle: str | None = None,
        *,
        y: float = 0.95,
        fontsize: float | None = None,
    ) -> None:
        """Set a figure title with an optional smaller subtitle below it."""
        cfg = self.config.text
        title_text = title or self.config.title
        subtitle_text = subtitle if subtitle is not None else self.config.subtitle

        if title_text:
            fig.suptitle(
                format_plot_title(title_text),
                fontsize=fontsize or cfg.title_fontsize,
                y=y,
            )
        if subtitle_text:
            fig.text(
                0.5,
                y - 0.04,
                subtitle_text,
                ha="center",
                va="top",
                fontsize=cfg.annotation_small,
                color="#58595B",
            )

    def set_limits(
        self,
        ax: matplotlib.axes.Axes,
        vmin: float | None = None,
        vmax: float | None = None,
        axis: Literal["x", "y", "both"] = "y",
    ) -> None:
        """Set axis limits.

        Parameters
        ----------
        ax
            Axes to configure.
        vmin
            Minimum value. Uses config if None.
        vmax
            Maximum value. Uses config if None.
        axis
            Which axis to set ('x', 'y', or 'both').
        """
        vmin = vmin if vmin is not None else self.config.vmin
        vmax = vmax if vmax is not None else self.config.vmax

        if vmin is not None or vmax is not None:
            if axis in ("y", "both"):
                ax.set_ylim(vmin, vmax)
            if axis in ("x", "both"):
                ax.set_xlim(vmin, vmax)

    def add_legend(
        self,
        ax: matplotlib.axes.Axes,
        loc: str = "best",
        **kwargs: Any,
    ) -> None:
        """Add legend to axes.

        Parameters
        ----------
        ax
            Axes to add legend to.
        loc
            Legend location.
        **kwargs
            Additional legend arguments.
        """
        legend_kwargs = {
            "loc": loc,
            "fontsize": self.config.text.legend,
            "framealpha": 0.9,
            **kwargs,
        }
        ax.legend(**legend_kwargs)

    def save(
        self,
        fig: matplotlib.figure.Figure,
        output_path: str | Path,
        dpi: int | None = None,
        bbox_inches: str = "tight",
        **kwargs: Any,
    ) -> Path:
        """Save figure to file.

        Parameters
        ----------
        fig
            Figure to save.
        output_path
            Output file path.
        dpi
            Resolution. Uses config if None.
        bbox_inches
            Bounding box setting.
        **kwargs
            Additional savefig arguments.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {
            "dpi": dpi or self.config.figure.dpi,
            "bbox_inches": bbox_inches,
            "facecolor": fig.get_facecolor(),
            **kwargs,
        }
        fig.savefig(output_path, **save_kwargs)
        return output_path

    def close(self, fig: matplotlib.figure.Figure) -> None:
        """Close a figure to free memory.

        Parameters
        ----------
        fig
            Figure to close.
        """
        plt.close(fig)


# =============================================================================
# Public re-export surface  (keep __all__ in sync with what callers expect)
# =============================================================================

__all__ = [
    # Config dataclasses
    "FigureConfig",
    "TextConfig",
    "StyleConfig",
    "DomainConfig",
    "PlotConfig",
    # Base ABC
    "BasePlotter",
    # Label/formatting utilities
    "VARIABLE_DISPLAY_NAMES",
    "TITLE_FORMULA_REPLACEMENTS",
    "UNIT_REPLACEMENTS",
    "format_plot_title",
    "format_variable_display_name",
    "canonical_variable_name",
    "get_variable_label",
    "get_variable_units",
    "format_units",
    "format_label_with_units",
    "calculate_symmetric_limits",
    "calculate_data_limits",
    "merge_config_dicts",
    # Series/color helpers
    "build_series",
    "series_colors",
    "get_dataset_color",
    "dataset_label",
    "get_series_label",
    "resolve_dataset_variable",
]
