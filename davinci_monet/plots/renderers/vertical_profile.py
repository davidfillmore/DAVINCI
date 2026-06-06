"""Vertical profile renderer (unified obs/paired).

Promoted from the obs-only ``obs_vertical_profile`` renderer (renderer unification
P3) to a canonical renderer on :class:`BasePlotter` speaking ``render(series)``:
1 series → one profile (raw scatter or altitude-binned mean ± std), N series →
overlay colored by source. Flight splitting is handled upstream by the plotting
stage, so each ``render`` call sees a single flight. ``obs_vertical_profile`` is
kept as a deprecated alias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from davinci_monet.plots.base import (
    BasePlotter,
    build_series,
    format_plot_title,
    get_variable_label,
    series_colors,
)
from davinci_monet.plots.registry import register_alias, register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

    from davinci_monet.core.base import PlotSeries


@register_plotter("vertical_profile")
class VerticalProfilePlotter(BasePlotter):
    """Altitude vs. value profile for 1..N source series."""

    name: str = "vertical_profile"
    default_figsize: tuple[float, float] = (6, 8)

    def plot(  # type: ignore[override]
        self,
        obs_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Single-source convenience wrapper; ``render`` is the unified entry."""
        return self.render(build_series(obs_data, variable), ax=ax, **kwargs)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        *,
        mode: Literal["scatter", "binned"] = "scatter",
        n_bins: int = 20,
        alt_coord: str = "altitude",
        title: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render vertical profile(s): 1 → single source, N → overlay."""
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        colors = series_colors(series)
        multi = len(series) > 1
        for s, color in zip(series, colors):
            label = s.source_label or get_variable_label(
                s.dataset, s.var_name, include_prefix=False
            )
            if mode == "binned":
                self._plot_binned(ax, s, alt_coord, n_bins, color, label)
            else:
                self._plot_scatter(ax, s, alt_coord, color, label)

        first = series[0]
        var_label = get_variable_label(first.dataset, first.var_name, include_prefix=False)
        units = first.dataset[first.var_name].attrs.get("units", "")
        ax.set_xlabel(
            f"{var_label} ({units})" if units else var_label, fontsize=self.config.text.fontsize
        )
        alt_units = (
            first.dataset[alt_coord].attrs.get("units", "")
            if alt_coord in first.dataset.coords
            else ""
        )
        ax.set_ylabel(
            f"Altitude ({alt_units})" if alt_units else "Altitude",
            fontsize=self.config.text.fontsize,
        )
        ax.set_title(
            format_plot_title(title) if title else f"{var_label} Vertical Profile",
            fontsize=self.config.text.title_fontsize,
        )
        ax.grid(True, alpha=0.3)
        if multi:
            ax.legend(fontsize=self.config.text.legend)
        return fig

    def _plot_scatter(self, ax: Any, s: PlotSeries, alt_coord: str, color: str, label: str) -> None:
        values = s.dataset[s.var_name].values.ravel()
        altitudes = s.dataset[alt_coord].values.ravel()
        valid = np.isfinite(values) & np.isfinite(altitudes)
        ax.scatter(
            values[valid], altitudes[valid], c=color, s=8, alpha=0.5, edgecolors="none", label=label
        )

    def _plot_binned(
        self, ax: Any, s: PlotSeries, alt_coord: str, n_bins: int, color: str, label: str
    ) -> None:
        values = s.dataset[s.var_name].values.ravel()
        altitudes = s.dataset[alt_coord].values.ravel()
        valid = np.isfinite(values) & np.isfinite(altitudes)
        values, altitudes = values[valid], altitudes[valid]
        if values.size == 0:
            return
        bin_edges = np.linspace(altitudes.min(), altitudes.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.clip(np.digitize(altitudes, bin_edges) - 1, 0, n_bins - 1)
        means = np.full(n_bins, np.nan)
        stds = np.full(n_bins, np.nan)
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                means[i] = np.nanmean(values[mask])
                stds[i] = np.nanstd(values[mask])
        vb = np.isfinite(means)
        ax.plot(means[vb], bin_centers[vb], color=color, linewidth=1.5, label=label)
        ax.fill_betweenx(
            bin_centers[vb], (means - stds)[vb], (means + stds)[vb], color=color, alpha=0.2
        )


# ``obs_vertical_profile`` is a deprecated alias of the unified renderer.
register_alias("obs_vertical_profile", "vertical_profile")
