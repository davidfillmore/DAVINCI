"""Base class for observation-only plotters.

This module provides ObsPlotter, an abstract base class for plotters that
work with raw observation datasets (not paired model-observation data).

Unlike BasePlotter which requires paired data with obs_var + model_var,
ObsPlotter takes a single xr.Dataset and variable name. This is used for
observation-only diagnostics such as flight track maps, vertical profiles,
time series, and histograms.

Usage
-----
Subclass ObsPlotter and implement the ``plot()`` method:

>>> class MyObsPlot(ObsPlotter):
...     name = "my_obs_plot"
...     def plot(self, obs_data, variable, ax=None, **kwargs):
...         fig, ax = self.create_figure()
...         ax.plot(obs_data[variable].values)
...         return fig
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from davinci_monet.plots.base import FigureConfig, PlotConfig, TextConfig
from davinci_monet.plots.style import OBS_COLOR

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


class ObsPlotter(ABC):
    """Abstract base class for observation-only plotters.

    Unlike BasePlotter which requires paired (obs + model) data,
    ObsPlotter works with raw observation datasets and a single variable.

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
    name: str = "obs_base"

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
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate the plot.

        Parameters
        ----------
        obs_data
            Observation dataset with the variable to plot.
        variable
            Name of the variable to plot.
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
        fig, ax = plt.subplots(
            figsize=cfg.figsize,
            dpi=cfg.dpi,
            facecolor=cfg.facecolor,
            constrained_layout=cfg.constrained_layout,
            **kwargs,
        )
        return fig, ax

    def save(
        self,
        fig: matplotlib.figure.Figure,
        filepath: str | Path,
        dpi: int | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save figure to file.

        Parameters
        ----------
        fig
            Figure to save.
        filepath
            Output file path.
        dpi
            Resolution. Uses config if None.
        **kwargs
            Additional savefig arguments.

        Returns
        -------
        Path
            Path to saved file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            filepath,
            dpi=dpi or self.config.figure.dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            **kwargs,
        )
        return filepath

    def close(self, fig: matplotlib.figure.Figure) -> None:
        """Close a figure to free memory.

        Parameters
        ----------
        fig
            Figure to close.
        """
        plt.close(fig)
