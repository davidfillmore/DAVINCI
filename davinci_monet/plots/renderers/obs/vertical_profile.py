"""Vertical profile renderer for DAVINCI.

Renders altitude vs. concentration plots in scatter or binned mode.
Binned mode computes altitude-bin means with standard deviation envelopes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from davinci_monet.plots.base import format_plot_title, get_variable_label
from davinci_monet.plots.obs_base import ObsPlotter
from davinci_monet.plots.registry import register_plotter
from davinci_monet.plots.style import NCAR_PALETTE, NCAR_PRIMARY

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("obs_vertical_profile")
class VerticalProfilePlotter(ObsPlotter):
    """Plotter for vertical profiles of observed variables.

    Supports two modes:
    - ``"scatter"``: Raw observation points plotted as altitude vs. value.
    - ``"binned"``: Observations are binned by altitude, showing bin means
      with standard deviation shading.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = VerticalProfilePlotter()
    >>> fig = plotter.plot(obs_data, "O3", mode="binned", n_bins=20)
    """

    name: str = "obs_vertical_profile"
    default_figsize: tuple[float, float] = (6, 8)

    def plot(
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        mode: Literal["scatter", "binned"] = "scatter",
        n_bins: int = 20,
        alt_coord: str = "altitude",
        title: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a vertical profile plot.

        Parameters
        ----------
        obs_data
            Observation dataset with altitude coordinate and the variable.
        variable
            Name of the variable to plot on the x-axis.
        ax
            Optional axes to plot on. If None, creates new figure.
        mode
            ``"scatter"`` for raw points, ``"binned"`` for altitude-bin means.
        n_bins
            Number of altitude bins (used when mode is ``"binned"``).
        alt_coord
            Name of the altitude coordinate.
        title
            Plot title. Defaults to "{variable} Vertical Profile".
        color
            Line/point color. Defaults to OBS_COLOR.
        **kwargs
            Additional arguments passed to the underlying plot call.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()

        color = color or NCAR_PRIMARY

        # Check for multi-flight data (>1 unique flight)
        has_flights = "flight" in obs_data.coords and len(np.unique(obs_data["flight"].values)) > 1

        if has_flights:
            self._plot_multi_flight(ax, obs_data, variable, alt_coord, mode, n_bins, **kwargs)
        elif mode == "binned":
            self._plot_binned(ax, obs_data, variable, alt_coord, n_bins, color, **kwargs)
        else:
            self._plot_scatter(ax, obs_data, variable, alt_coord, color, **kwargs)

        # Labels
        var_label = get_variable_label(obs_data, variable, include_prefix=False)
        units = obs_data[variable].attrs.get("units", "")
        xlabel = f"{var_label} ({units})" if units else var_label
        ax.set_xlabel(xlabel, fontsize=self.config.text.fontsize)

        alt_units = ""
        if alt_coord in obs_data.coords:
            alt_units = obs_data[alt_coord].attrs.get("units", "")
        ylabel = "Altitude"
        if alt_units:
            ylabel = f"Altitude ({alt_units})"
        ax.set_ylabel(ylabel, fontsize=self.config.text.fontsize)

        # Title
        if title is None:
            title = f"{var_label} Vertical Profile"
        else:
            title = format_plot_title(title)
        ax.set_title(title, fontsize=self.config.text.title_fontsize)

        ax.grid(True, alpha=0.3)

        if has_flights:
            ax.legend(fontsize=self.config.text.legend)

        return fig

    def _plot_scatter(
        self,
        ax: matplotlib.axes.Axes,
        obs_data: xr.Dataset,
        variable: str,
        alt_coord: str,
        color: str,
        **kwargs: Any,
    ) -> None:
        """Plot raw scatter points."""
        values = obs_data[variable].values
        altitudes = obs_data[alt_coord].values

        valid = np.isfinite(values) & np.isfinite(altitudes)
        ax.scatter(
            values[valid],
            altitudes[valid],
            c=color,
            s=8,
            alpha=0.5,
            edgecolors="none",
            **kwargs,
        )

    def _plot_binned(
        self,
        ax: matplotlib.axes.Axes,
        obs_data: xr.Dataset,
        variable: str,
        alt_coord: str,
        n_bins: int,
        color: str,
        **kwargs: Any,
    ) -> None:
        """Plot altitude-binned means with std envelope."""
        values = obs_data[variable].values
        altitudes = obs_data[alt_coord].values

        valid = np.isfinite(values) & np.isfinite(altitudes)
        values = values[valid]
        altitudes = altitudes[valid]

        # Create altitude bins
        bin_edges = np.linspace(altitudes.min(), altitudes.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.digitize(altitudes, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        means = np.full(n_bins, np.nan)
        stds = np.full(n_bins, np.nan)

        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                means[i] = np.nanmean(values[mask])
                stds[i] = np.nanstd(values[mask])

        # Plot mean line and std envelope
        valid_bins = np.isfinite(means)
        ax.plot(means[valid_bins], bin_centers[valid_bins], color=color, linewidth=1.5)
        ax.fill_betweenx(
            bin_centers[valid_bins],
            (means - stds)[valid_bins],
            (means + stds)[valid_bins],
            color=color,
            alpha=0.2,
        )

    def _plot_multi_flight(
        self,
        ax: matplotlib.axes.Axes,
        obs_data: xr.Dataset,
        variable: str,
        alt_coord: str,
        mode: str,
        n_bins: int,
        **kwargs: Any,
    ) -> None:
        """Plot separate profiles for each flight."""
        flight_ids = np.unique(obs_data["flight"].values)

        for i, fid in enumerate(flight_ids):
            color = NCAR_PALETTE[i % len(NCAR_PALETTE)]
            mask = obs_data["flight"].values == fid
            subset = obs_data.isel(time=mask)

            if mode == "binned":
                self._plot_binned(ax, subset, variable, alt_coord, n_bins, color, **kwargs)
                # Add invisible point for legend
                ax.plot([], [], color=color, linewidth=1.5, label=str(fid))
            else:
                self._plot_scatter(ax, subset, variable, alt_coord, color, **kwargs)
                ax.scatter([], [], c=color, s=8, label=str(fid))
