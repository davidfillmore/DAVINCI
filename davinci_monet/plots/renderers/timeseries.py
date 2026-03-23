"""Time series plot renderer for DAVINCI.

This module provides time series plotting functionality for comparing
model output with observations over time.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


@register_plotter("timeseries")
class TimeSeriesPlotter(BasePlotter):
    """Plotter for time series comparisons.

    Creates line plots showing model and observation values over time.
    Supports temporal resampling and uncertainty bands.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TimeSeriesPlotter()
    >>> fig = plotter.plot(
    ...     paired_data,
    ...     obs_var="obs_o3",
    ...     model_var="model_o3",
    ...     resample="1h",
    ... )
    """

    name: str = "timeseries"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for temporal data

    def plot(
        self,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        ax: matplotlib.axes.Axes | None = None,
        resample: str | None = None,
        show_uncertainty: bool = False,
        uncertainty_type: Literal["std", "iqr", "range"] = "std",
        time_dim: str = "time",
        aggregate_dim: str | None = None,
        obs_label: str | None = None,
        model_label: str | None = None,
        show_individual_sites: bool = False,
        site_dim: str = "site",
        site_label_var: str = "site_name",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a time series plot.

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
        resample
            Temporal resampling frequency (e.g., '1h', '1D').
        show_uncertainty
            If True, show uncertainty bands around mean.
        uncertainty_type
            Type of uncertainty band ('std', 'iqr', 'range').
        time_dim
            Name of time dimension.
        aggregate_dim
            Optional dimension to aggregate over (e.g., 'site').
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

        # Get data arrays
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Plot individual sites if requested
        if show_individual_sites and site_dim in obs_data.dims:
            return self._plot_individual_sites(
                fig, ax, paired_data, obs_var, model_var,
                time_dim, site_dim, site_label_var, resample, **kwargs
            )

        # Aggregate over non-time dimensions if specified
        if aggregate_dim is not None and aggregate_dim in obs_data.dims:
            obs_data = obs_data.mean(dim=aggregate_dim)
            model_data = model_data.mean(dim=aggregate_dim)
        elif len(obs_data.dims) > 1:
            # If multiple dimensions and no aggregate specified, average over all except time
            other_dims = [d for d in obs_data.dims if d != time_dim]
            if other_dims:
                obs_data = obs_data.mean(dim=other_dims)
                model_data = model_data.mean(dim=other_dims)

        # Get time coordinate
        time = paired_data[time_dim]

        # Resample if requested
        if resample:
            obs_data = obs_data.resample({time_dim: resample}).mean()
            model_data = model_data.resample({time_dim: resample}).mean()
            time = obs_data[time_dim]

        # Convert to numpy for plotting
        time_values = pd.to_datetime(time.values)
        obs_values = obs_data.values
        model_values = model_data.values

        # Get style configuration
        style = self.config.style

        # Get labels
        obs_label = obs_label or get_variable_label(
            paired_data, obs_var, self.config.obs_label
        ) or "Observations"
        model_label = model_label or get_variable_label(
            paired_data, model_var, self.config.model_label
        ) or "Model"

        # Plot observations
        ax.plot(
            time_values,
            obs_values,
            color=style.obs_color,
            linestyle=style.obs_linestyle,
            marker=style.obs_marker if len(time_values) < 50 else None,
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
            label=obs_label,
        )

        # Plot model
        ax.plot(
            time_values,
            model_values,
            color=style.model_color,
            linestyle=style.model_linestyle,
            marker=style.model_marker if len(time_values) < 50 else None,
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
            label=model_label,
        )

        # Show uncertainty bands if requested (requires ungrouped data)
        if show_uncertainty and aggregate_dim is not None:
            self._add_uncertainty_bands(
                ax,
                paired_data,
                obs_var,
                model_var,
                time_dim,
                aggregate_dim,
                resample,
                uncertainty_type,
            )

        # Set x-axis limits to actual data range (avoid extra ticks beyond data)
        ax.set_xlim(time_values.min(), time_values.max())

        # Formatting
        self.apply_text_style(ax)

        # Set labels - use automatic variable display name (no prefix for shared axis)
        units = get_variable_units(paired_data, obs_var)
        ylabel = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, obs_var, include_prefix=False),
            units,
        )
        self.set_labels(ax, xlabel="Time", ylabel=ylabel)

        # Smart auto-scaling for y-axis
        self._set_smart_ylim(
            ax, paired_data, obs_var, model_var, aggregate_dim,
            time_dim, resample, show_uncertainty, uncertainty_type
        )

        # Add legend
        self.add_legend(ax)

        # Rotate x-axis labels for readability
        ax.tick_params(axis="x", rotation=45)

        # Grid
        ax.grid(True, alpha=0.3)

        return fig

    def _plot_individual_sites(
        self,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        time_dim: str,
        site_dim: str,
        site_label_var: str,
        resample: str | None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot individual site time series.

        Parameters
        ----------
        fig
            Figure object.
        ax
            Axes to plot on.
        paired_data
            Paired dataset.
        obs_var, model_var
            Variable names.
        time_dim
            Time dimension name.
        site_dim
            Site dimension name.
        site_label_var
            Variable containing site labels.
        resample
            Resampling frequency.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.cm as cm

        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]
        time_values = pd.to_datetime(paired_data[time_dim].values)

        # Get site labels
        if site_label_var in paired_data.coords:
            site_labels = paired_data[site_label_var].values
        else:
            site_labels = [f"Site {i}" for i in range(paired_data.sizes[site_dim])]

        n_sites = paired_data.sizes[site_dim]

        # Use a colormap for different sites
        colors = cm.tab20(np.linspace(0, 1, min(n_sites, 20)))

        # Plot each site
        for i in range(n_sites):
            site_obs = obs_data.isel({site_dim: i})
            site_model = model_data.isel({site_dim: i})

            # Skip if all NaN
            if site_obs.isnull().all() and site_model.isnull().all():
                continue

            color = colors[i % len(colors)]
            label = str(site_labels[i]) if i < len(site_labels) else f"Site {i}"

            # Plot observations as solid lines
            ax.plot(
                time_values,
                site_obs.values,
                color=color,
                linestyle="-",
                marker="o",
                markersize=4,
                linewidth=1,
                alpha=0.7,
                label=f"{label} (obs)",
            )

            # Plot model as dashed lines
            ax.plot(
                time_values,
                site_model.values,
                color=color,
                linestyle="--",
                marker="s",
                markersize=4,
                linewidth=1,
                alpha=0.7,
                label=f"{label} (model)",
            )

        # Formatting
        self.apply_text_style(ax)

        # Set labels - use automatic variable display name (no prefix for shared axis)
        units = get_variable_units(paired_data, obs_var)
        ylabel = format_label_with_units(
            self.config.ylabel or get_variable_label(paired_data, obs_var, include_prefix=False),
            units,
        )
        self.set_labels(ax, xlabel="Time", ylabel=ylabel)

        # Legend - put outside plot if many sites
        if n_sites > 5:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=self.config.text.legend_small)
            fig.tight_layout()
        else:
            ax.legend(fontsize=self.config.text.legend)

        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        return fig

    def _add_uncertainty_bands(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        time_dim: str,
        aggregate_dim: str,
        resample: str | None,
        uncertainty_type: str,
    ) -> None:
        """Add uncertainty bands to the plot.

        Parameters
        ----------
        ax
            Axes to add bands to.
        paired_data
            Full paired dataset.
        obs_var, model_var
            Variable names.
        time_dim
            Time dimension name.
        aggregate_dim
            Dimension being aggregated.
        resample
            Resampling frequency.
        uncertainty_type
            Type of uncertainty ('std', 'iqr', 'range').
        """
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Resample first if needed
        if resample:
            obs_data = obs_data.resample({time_dim: resample}).mean()
            model_data = model_data.resample({time_dim: resample}).mean()

        time_values = pd.to_datetime(obs_data[time_dim].values)
        style = self.config.style

        # Calculate uncertainty bounds
        if uncertainty_type == "std":
            obs_mean = obs_data.mean(dim=aggregate_dim)
            model_mean = model_data.mean(dim=aggregate_dim)

            # Suppress warnings for time bins with single observations (ddof > n)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                obs_std = obs_data.std(dim=aggregate_dim)
                model_std = model_data.std(dim=aggregate_dim)

            obs_lower = obs_mean - obs_std
            obs_upper = obs_mean + obs_std
            model_lower = model_mean - model_std
            model_upper = model_mean + model_std

        elif uncertainty_type == "iqr":
            obs_lower = obs_data.quantile(0.25, dim=aggregate_dim)
            obs_upper = obs_data.quantile(0.75, dim=aggregate_dim)
            model_lower = model_data.quantile(0.25, dim=aggregate_dim)
            model_upper = model_data.quantile(0.75, dim=aggregate_dim)

        else:  # range
            obs_lower = obs_data.min(dim=aggregate_dim)
            obs_upper = obs_data.max(dim=aggregate_dim)
            model_lower = model_data.min(dim=aggregate_dim)
            model_upper = model_data.max(dim=aggregate_dim)

        # Plot bands
        ax.fill_between(
            time_values,
            obs_lower.values,
            obs_upper.values,
            color=style.obs_color,
            alpha=0.2,
        )
        ax.fill_between(
            time_values,
            model_lower.values,
            model_upper.values,
            color=style.model_color,
            alpha=0.2,
        )

    def _set_smart_ylim(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        obs_var: str,
        model_var: str,
        aggregate_dim: str | None,
        time_dim: str,
        resample: str | None,
        show_uncertainty: bool,
        uncertainty_type: str,
    ) -> None:
        """Set smart y-axis limits based on data range.

        Computes appropriate limits from the actual data, including
        uncertainty bands if shown. Uses vmin=0 for non-negative data
        and adds padding to vmax.

        Parameters
        ----------
        ax
            Axes to configure.
        paired_data
            Full paired dataset.
        obs_var, model_var
            Variable names.
        aggregate_dim
            Dimension being aggregated.
        time_dim
            Time dimension name.
        resample
            Resampling frequency.
        show_uncertainty
            Whether uncertainty bands are shown.
        uncertainty_type
            Type of uncertainty ('std', 'iqr', 'range').
        """
        # If config specifies both limits, use them
        if self.config.vmin is not None and self.config.vmax is not None:
            ax.set_ylim(self.config.vmin, self.config.vmax)
            return

        # Get data for computing range
        obs_data = paired_data[obs_var]
        model_data = paired_data[model_var]

        # Resample if needed
        if resample:
            obs_data = obs_data.resample({time_dim: resample}).mean()
            model_data = model_data.resample({time_dim: resample}).mean()

        # Compute the data range we need to display
        if show_uncertainty and aggregate_dim is not None:
            # Need to include uncertainty bands in range calculation
            if uncertainty_type == "std":
                obs_mean = obs_data.mean(dim=aggregate_dim)
                model_mean = model_data.mean(dim=aggregate_dim)

                # Suppress warnings for time bins with single observations (ddof > n)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                    obs_std = obs_data.std(dim=aggregate_dim)
                    model_std = model_data.std(dim=aggregate_dim)

                data_min = float(min(
                    np.nanmin(obs_mean.values - obs_std.values),
                    np.nanmin(model_mean.values - model_std.values)
                ))
                data_max = float(max(
                    np.nanmax(obs_mean.values + obs_std.values),
                    np.nanmax(model_mean.values + model_std.values)
                ))
            elif uncertainty_type == "iqr":
                data_min = float(min(
                    np.nanmin(obs_data.quantile(0.25, dim=aggregate_dim).values),
                    np.nanmin(model_data.quantile(0.25, dim=aggregate_dim).values)
                ))
                data_max = float(max(
                    np.nanmax(obs_data.quantile(0.75, dim=aggregate_dim).values),
                    np.nanmax(model_data.quantile(0.75, dim=aggregate_dim).values)
                ))
            else:  # range
                data_min = float(min(
                    np.nanmin(obs_data.min(dim=aggregate_dim).values),
                    np.nanmin(model_data.min(dim=aggregate_dim).values)
                ))
                data_max = float(max(
                    np.nanmax(obs_data.max(dim=aggregate_dim).values),
                    np.nanmax(model_data.max(dim=aggregate_dim).values)
                ))
        else:
            # Just use mean values
            if aggregate_dim is not None and aggregate_dim in obs_data.dims:
                obs_data = obs_data.mean(dim=aggregate_dim)
                model_data = model_data.mean(dim=aggregate_dim)

            data_min = float(min(np.nanmin(obs_data.values), np.nanmin(model_data.values)))
            data_max = float(max(np.nanmax(obs_data.values), np.nanmax(model_data.values)))

        # Check if raw data is non-negative (physical constraint)
        # Use original data before aggregation to check this
        raw_obs = paired_data[obs_var]
        raw_model = paired_data[model_var]
        raw_min = float(min(np.nanmin(raw_obs.values), np.nanmin(raw_model.values)))
        is_positive_definite = raw_min >= 0

        # Add padding (10% of range)
        data_range = data_max - data_min
        padding = data_range * 0.1 if data_range > 0 else 0.1

        # Determine vmin: use 0 for non-negative data, otherwise add padding
        if self.config.vmin is not None:
            vmin = self.config.vmin
        elif is_positive_definite:
            # Non-negative data (concentrations, AOD, etc.) - start at 0
            vmin = 0.0
        else:
            vmin = data_min - padding

        # Determine vmax: use config or add padding
        if self.config.vmax is not None:
            vmax = self.config.vmax
        else:
            vmax = data_max + padding

        ax.set_ylim(vmin, vmax)


def plot_timeseries(
    paired_data: xr.Dataset,
    obs_var: str,
    model_var: str,
    config: PlotConfig | dict[str, Any] | None = None,
    **kwargs: Any,
) -> matplotlib.figure.Figure:
    """Convenience function for time series plotting.

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

    plotter = TimeSeriesPlotter(config=config)
    return plotter.plot(paired_data, obs_var, model_var, **kwargs)
