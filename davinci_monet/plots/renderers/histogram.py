"""Distribution histogram renderer for one or more source series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from davinci_monet.plots.base import (
    BasePlotter,
    build_series,
    get_variable_label,
    series_colors,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

    from davinci_monet.core.base import PlotSeries


@register_plotter("histogram")
class HistogramPlotter(BasePlotter):
    """Distribution histogram for 1..N source series."""

    name: str = "histogram"
    default_figsize: tuple[float, float] = (8, 5)

    def plot(  # type: ignore[override]
        self,
        x_data: xr.Dataset,
        variable: str,
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Single-source convenience wrapper; ``render`` is the unified entry."""
        return self.render(build_series(x_data, variable), ax=ax, **kwargs)

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        *,
        n_bins: int = 30,
        show_stats: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render distribution histogram(s).

        1 series → one histogram + red median line + (optional) N/Mean/Median/Std/
        P10/P90 stats box. N series → translucent overlay, source-colored,
        with a legend.
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        colors = series_colors(series)
        single = len(series) == 1

        for s, color in zip(series, colors):
            values = s.dataset[s.var_name].values.ravel()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            label = s.source_label or get_variable_label(
                s.dataset, s.var_name, include_prefix=False
            )
            ax.hist(
                values,
                bins=n_bins,
                color=color,
                edgecolor="white",
                alpha=0.8 if single else 0.5,
                label=label,
                **kwargs,
            )
            if single:
                median = float(np.median(values))
                ax.axvline(median, color="#D62839", linestyle="--", linewidth=1.5)
                if show_stats:
                    stats_text = (
                        f"N={len(values)}\n"
                        f"Mean={float(np.mean(values)):.2f}\n"
                        f"Median={median:.2f}\n"
                        f"Std={float(np.std(values)):.2f}\n"
                        f"P10={float(np.percentile(values, 10)):.2f}\n"
                        f"P90={float(np.percentile(values, 90)):.2f}"
                    )
                    ax.text(
                        0.97,
                        0.95,
                        stats_text,
                        transform=ax.transAxes,
                        fontsize=self.config.text.annotation,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            alpha=0.8,
                            edgecolor="#CCCCCC",
                        ),
                    )

        first = series[0]
        var_label = get_variable_label(first.dataset, first.var_name, include_prefix=False)
        units = first.dataset[first.var_name].attrs.get("units", "")
        ax.set_xlabel(
            f"{var_label} ({units})" if units else var_label, fontsize=self.config.text.fontsize
        )
        ax.set_ylabel("Count", fontsize=self.config.text.fontsize)
        self.set_title(ax, title if title else f"{var_label} Distribution")
        if not single:
            ax.legend(fontsize=self.config.text.legend)
        ax.grid(True, alpha=0.3, axis="y")
        return fig
