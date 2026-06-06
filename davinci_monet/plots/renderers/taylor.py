"""Taylor diagram plot renderer for DAVINCI.

This module provides Taylor diagram plotting functionality for
visualizing model-observation statistical relationships.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from davinci_monet.core.base import PlotSeries
from davinci_monet.plots.base import (
    BasePlotter,
    PlotConfig,
    build_series,
    get_role_color,
    get_series_label,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr


@register_plotter("taylor")
class TaylorPlotter(BasePlotter):
    """Plotter for Taylor diagrams.

    Creates Taylor diagrams showing the statistical relationship
    between model and observation data using correlation, standard
    deviation, and centered RMS difference.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TaylorPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ... )
    """

    name: str = "taylor"
    default_figsize: tuple[float, float] = (6, 6)  # Square for polar diagram

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render a Taylor diagram from a list of two PlotSeries.

        Parameters
        ----------
        series
            Exactly 2 series: one reference (obs) and one comparand (model).
        ax
            Optional axes to plot on (must be polar). If None, creates new.
        **kwargs
            Forwarded to the Taylor rendering logic.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        if len(series) != 2:
            raise NotImplementedError(
                f"TaylorPlotter.render requires exactly 2 series; got {len(series)}."
            )
        ref = next((s for s in series if s.pair_role == "reference"), series[0])
        comp = next((s for s in series if s.pair_role == "comparand"), series[1])
        paired_data = ref.dataset
        obs_var = ref.var_name
        model_var = comp.var_name

        normalize: bool = kwargs.pop("normalize", True)
        show_reference: bool = kwargs.pop("show_reference", True)
        reference_label: str | None = kwargs.pop("reference_label", None)
        model_label: str | None = kwargs.pop("model_label", None)
        marker: str | None = kwargs.pop("marker", None)
        color: str | None = kwargs.pop("color", None)

        # Get data and flatten
        obs_values = paired_data[obs_var].values.flatten()
        model_values = paired_data[model_var].values.flatten()

        # Remove NaN values
        mask = np.isfinite(obs_values) & np.isfinite(model_values)
        obs_values = obs_values[mask]
        model_values = model_values[mask]

        # Calculate statistics
        obs_std = np.std(obs_values)
        model_std = np.std(model_values)
        correlation = np.corrcoef(obs_values, model_values)[0, 1]

        # Normalize if requested
        if normalize:
            model_std_norm = model_std / obs_std
            obs_std_norm = 1.0
        else:
            model_std_norm = model_std
            obs_std_norm = obs_std

        # Create Taylor diagram
        if ax is None:
            fig, ax = self._create_taylor_axes(obs_std_norm, normalize)
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get style
        style = self.config.style
        m = marker or style.model_marker
        c = color or get_role_color(
            paired_data, model_var, 1, obs_color=style.obs_color, model_color=style.model_color
        )
        label = model_label or self.config.model_label or get_series_label(paired_data, model_var)

        # Plot model point
        # Taylor diagram uses polar coordinates: theta=arccos(correlation), r=std
        theta = np.arccos(correlation)
        ax.plot(
            theta,
            model_std_norm,
            marker=m,
            color=c,
            markersize=style.markersize * 1.5,
            label=label,
            linestyle="none",
        )

        # Plot reference (observation) point. The reference is the obs source;
        # label it with the obs source label by default (R-3), keeping the
        # conventional black star marker.
        if show_reference:
            ax.plot(
                0,  # Perfect correlation
                obs_std_norm,
                marker="*",
                color="k",
                markersize=style.markersize * 2,
                label=reference_label
                or self.config.obs_label
                or get_series_label(paired_data, obs_var),
                linestyle="none",
            )

        # Add legend
        self.add_legend(ax, loc="upper right")

        return fig

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        normalize: bool = True,
        show_reference: bool = True,
        reference_label: str | None = None,
        model_label: str | None = None,
        marker: str | None = None,
        color: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a Taylor diagram.

        Parameters
        ----------
        paired_data
            Paired dataset with model and observation variables.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        ax
            Optional axes to plot on (must be polar). If None, creates new.
        normalize
            If True, normalize by observation standard deviation.
        show_reference
            If True, show reference (observation) point.
        reference_label
            Label for reference point. Defaults to the obs source label.
        model_label
            Label for model point.
        marker
            Marker style for model point.
        color
            Color for model point.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        return self.render(
            build_series(paired_data, obs_var, model_var),
            ax=ax,
            normalize=normalize,
            show_reference=show_reference,
            reference_label=reference_label,
            model_label=model_label,
            marker=marker,
            color=color,
            **kwargs,
        )

    def _create_taylor_axes(
        self,
        ref_std: float,
        normalized: bool,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create axes for Taylor diagram.

        Parameters
        ----------
        ref_std
            Reference standard deviation for scaling.
        normalized
            Whether values are normalized.

        Returns
        -------
        tuple[Figure, Axes]
            Figure and polar axes.
        """
        fig = plt.figure(figsize=self.config.figure.figsize, dpi=self.config.figure.dpi)

        # Create polar axes for first quadrant only
        ax = fig.add_subplot(111, projection="polar")
        ax.set_thetamin(0)  # type: ignore[attr-defined]
        ax.set_thetamax(90)  # type: ignore[attr-defined]

        # Set radial limits
        max_std = ref_std * 1.5
        ax.set_ylim(0, max_std)

        # Correlation labels on angular axis
        correlation_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        ax.set_thetagrids(  # type: ignore[attr-defined]
            np.arccos(correlation_ticks) * 180 / np.pi,
            labels=[f"{c:.2g}" for c in correlation_ticks],
        )

        # Labels
        ax.set_xlabel("Standard Deviation" + (" (normalized)" if normalized else ""))
        ax.text(
            np.pi / 4,
            max_std * 1.2,
            "Correlation",
            ha="center",
            va="center",
            rotation=-45,
        )

        # Add centered RMS contours
        self._add_rms_contours(ax, ref_std, max_std)

        return fig, ax

    def _add_rms_contours(
        self,
        ax: matplotlib.axes.Axes,
        ref_std: float,
        max_std: float,
    ) -> None:
        """Add centered RMS difference contours.

        Parameters
        ----------
        ax
            Polar axes to add contours to.
        ref_std
            Reference standard deviation.
        max_std
            Maximum standard deviation for plot limits.
        """
        # RMS contours are circles centered at the reference point
        # In polar coordinates centered at origin, these become more complex
        theta = np.linspace(0, np.pi / 2, 100)

        # Draw a few RMS contours
        rms_values = (
            [0.25, 0.5, 0.75, 1.0]
            if ref_std == 1.0
            else [ref_std * x for x in [0.25, 0.5, 0.75, 1.0]]
        )

        for rms in rms_values:
            if rms > max_std:
                continue
            # Circle centered at (ref_std, 0) with radius rms
            # Parametric: x = ref_std + rms*cos(t), y = rms*sin(t)
            t = np.linspace(0, 2 * np.pi, 100)
            x = ref_std + rms * np.cos(t)
            y = rms * np.sin(t)

            # Convert to polar
            r = np.sqrt(x**2 + y**2)
            theta_rms = np.arctan2(y, x)

            # Only keep first quadrant
            mask = (theta_rms >= 0) & (theta_rms <= np.pi / 2) & (r <= max_std)
            if np.any(mask):
                ax.plot(
                    theta_rms[mask],
                    r[mask],
                    "k:",
                    alpha=0.3,
                    linewidth=0.5,
                )

    def plot_multiple(
        self,
        paired_datasets: dict[str, xr.Dataset],
        obs_var: str,
        model_var: str,
        normalize: bool = True,
        colors: dict[str, str] | None = None,
        markers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot multiple models on a single Taylor diagram.

        Parameters
        ----------
        paired_datasets
            Dictionary mapping model names to paired datasets.
        obs_var
            Name of observation variable.
        model_var
            Name of model variable.
        normalize
            If True, normalize by observation standard deviation.
        colors
            Optional color mapping for each model.
        markers
            Optional marker mapping for each model.
        **kwargs
            Additional arguments passed to plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        colors = colors or {}
        markers = markers or {}

        # Use first dataset to create axes
        first_key = next(iter(paired_datasets))
        first_data = paired_datasets[first_key]

        obs_values = first_data[obs_var].values.flatten()
        obs_values = obs_values[np.isfinite(obs_values)]
        ref_std = 1.0 if normalize else np.std(obs_values)

        fig, ax = self._create_taylor_axes(ref_std, normalize)

        # Default color cycle
        default_colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        # Plot each model
        for i, (name, data) in enumerate(paired_datasets.items()):
            color = colors.get(name, default_colors[i % len(default_colors)])
            marker = markers.get(name, "o")

            self.plot(
                data,
                obs_var,
                model_var,
                ax=ax,
                normalize=normalize,
                show_reference=(i == 0),  # Only show reference once
                model_label=name,
                color=color,
                marker=marker,
                **kwargs,
            )

        return fig


def plot_taylor(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for Taylor diagram plotting.

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

    plotter = TaylorPlotter(config=config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
