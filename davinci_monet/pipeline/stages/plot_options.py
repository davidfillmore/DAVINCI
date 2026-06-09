"""Plotting-stage option assembly helpers."""

from __future__ import annotations

from typing import Any

SINGLE_SOURCE_SCHEMA_KEYS = {
    "type",
    "obs",
    "source",
    "variable",
    "fig_kwargs",
    "default_plot_kwargs",
    "text_kwargs",
    "domain_type",
    "domain_name",
    "data",
    "data_proc",
}


FORWARDED_COMPARISON_OPTION_KEYS = {
    "show_site_labels",
    "show_individual_sites",
    "show_uncertainty",
    "uncertainty_type",
    "resample",
    "aggregate_dim",
    "label_sites",
    "site_label_var",
    "city_labels",
    "show_density",
    "density_cmap",
    "alpha",
    "marker_size",
    "obs_edgecolor",
    "obs_linewidth",
    "plot_type",
    "cmap",
    "show_surface_map",
    "surface_map_resolution",
    "land_color",
    "ocean_color",
    "show_var",
    "elev",
    "azim",
    "show_coastlines",
    "show_borders",
    "show_projection",
    "alt_scale",
}


def single_source_plot_kwargs(plot_spec: dict[str, Any]) -> dict[str, Any]:
    """Return renderer kwargs for a single-source plot spec."""
    return {k: v for k, v in plot_spec.items() if k not in SINGLE_SOURCE_SCHEMA_KEYS}


def build_comparison_plot_options(
    plot_type: str,
    plot_spec: dict[str, Any],
    analysis_config: dict[str, Any],
    *,
    nlevels: int | None = None,
) -> dict[str, Any]:
    """Return renderer options for paired comparison plots."""
    options = {k: plot_spec[k] for k in FORWARDED_COMPARISON_OPTION_KEYS if k in plot_spec}

    if (
        plot_type.startswith("spatial") or plot_type == "track_map_3d"
    ) and "city_labels" not in options:
        city_labels = analysis_config.get("city_labels")
        if city_labels:
            options["city_labels"] = city_labels

    if plot_type.startswith("spatial") and nlevels is not None:
        options.setdefault("n_levels", nlevels)

    return options


def build_plot_subtitle(
    analysis_config: dict[str, Any],
    *,
    snapshot_timestamp: str | None = None,
) -> str:
    """Return the subtitle line for a plot."""
    if snapshot_timestamp:
        return snapshot_timestamp

    start_time = analysis_config.get("start_time", "")
    end_time = analysis_config.get("end_time", "")
    if not start_time:
        return ""

    start_date = str(start_time).split(" ")[0]
    end_date = str(end_time).split(" ")[0] if end_time else start_date
    return start_date if start_date == end_date else f"{start_date} - {end_date}"


def timestamp_from_field(field: Any, time_index: int = 0) -> str:
    """Return a display timestamp from a time-indexed xarray field."""
    if "time" not in field.dims or len(field["time"]) <= time_index:
        return ""
    ts = field["time"].values[time_index]
    try:
        import pandas as pd

        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(ts)[:16] + " UTC"
