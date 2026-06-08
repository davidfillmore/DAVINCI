"""Plotting stage.

Generates comparison plots from paired data or single-source plots from
unpaired data through the unified renderer ``render(series)`` contract.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.core.base import iter_paired_variable_pairs
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import (
    iter_single_source_datasets,
    resolve_paired_var_names,
    tag_source_roles,
)


class PlottingStage(BaseStage):
    """Stage for generating plots from paired or single-source data."""

    def __init__(self) -> None:
        super().__init__("plotting")

    @staticmethod
    def _config_dict(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return dict(value.model_dump(exclude_none=True))
        if isinstance(value, dict):
            return value
        return {}

    @classmethod
    def _source_var_config(
        cls,
        config: dict[str, Any],
        source_label: str,
        variable_name: str,
    ) -> dict[str, Any]:
        """Return variable plot config for a source label from unified or legacy config."""
        for block in ("sources", "model", "obs"):
            block_cfg = config.get(block, {})
            if not isinstance(block_cfg, dict):
                continue
            source_cfg = cls._config_dict(block_cfg.get(source_label, {}))
            variables = cls._config_dict(source_cfg.get("variables", {}))
            if variable_name in variables:
                return cls._config_dict(variables[variable_name])
        return {}

    def validate(self, context: PipelineContext) -> bool:
        """Run for paired comparisons or single-source plots."""
        return bool(context.paired) or bool(iter_single_source_datasets(context))

    def _execute_single_source(self, context: PipelineContext) -> StageResult:
        """Single-source plotting.

        Renders each plot spec against its single configured source, auto-splitting
        on a ``flight`` coord with >1 flight. Multi-figure renderers (e.g. hourly
        LMA density) return a list of ``(fig, suffix)``.
        """
        import logging
        import time
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np

        from davinci_monet.plots.base import build_series
        from davinci_monet.plots.registry import get_plotter

        _logger = logging.getLogger(__name__)
        _SCHEMA_KEYS = {
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

        start = time.time()
        plots_config = context.config.get("plots", {})
        output_dir = Path(context.config.get("analysis", {}).get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_count = 0
        plots_generated: list[str] = []
        errors: list[str] = []
        source_map = {
            label: (obj, ds, role) for label, obj, ds, role in iter_single_source_datasets(context)
        }

        for plot_name, plot_spec in plots_config.items():
            plot_type = plot_spec.get("type", "")
            # Single-source specs carry either canonical ``source:`` or the
            # deprecated ``obs:`` key.
            if not plot_type or ("source" not in plot_spec and "obs" not in plot_spec):
                continue

            source_label = str(plot_spec.get("source") or plot_spec.get("obs") or "")
            variable = plot_spec.get("variable", "")

            if source_label not in source_map:
                errors.append(f"Source '{source_label}' not found for plot '{plot_name}'")
                continue

            _source_obj, ds, role = source_map[source_label]

            if variable not in ds.data_vars:
                errors.append(
                    f"Variable '{variable}' not in source '{source_label}' for plot '{plot_name}'"
                )
                continue

            plotter = get_plotter(plot_type)
            plot_kwargs = {k: v for k, v in plot_spec.items() if k not in _SCHEMA_KEYS}
            base_title = plot_kwargs.get("title", f"{variable} {plot_type}")

            has_flights = "flight" in ds.coords
            flight_ids = (
                sorted(set(np.unique(ds["flight"].values).tolist())) if has_flights else [None]
            )

            for fid in flight_ids:
                if fid is not None:
                    mask = ds["flight"].values == fid
                    subset = ds.isel(time=mask)
                    suffix = f"_{fid}"
                    flight_kwargs = {**plot_kwargs, "title": f"{base_title} — {fid}"}
                else:
                    subset = ds
                    suffix = ""
                    flight_kwargs = plot_kwargs

                vals = subset[variable].values
                if not np.any(np.isfinite(vals)):
                    continue

                if "flight_tracks" in plot_spec:
                    flight_kwargs["obs_datasets"] = {
                        label: source_ds for label, (_obj, source_ds, _role) in source_map.items()
                    }

                try:
                    # Tag the single source so build_series picks up role +
                    # source label, then render through the unified contract.
                    tag_source_roles(subset, role=role, source_label=source_label)
                    result = plotter.render(build_series(subset, variable), **flight_kwargs)
                    if isinstance(result, list):
                        for fig, fig_suffix in result:
                            out_path = output_dir / f"{plot_name}{suffix}{fig_suffix}.png"
                            plotter.save(fig, out_path)
                            plt.close(fig)
                            plot_count += 1
                            plots_generated.append(str(out_path))
                            _logger.info(f"Saved source plot: {out_path}")
                    else:
                        fig = result
                        out_path = output_dir / f"{plot_name}{suffix}.png"
                        plotter.save(fig, out_path)
                        plt.close(fig)
                        plot_count += 1
                        plots_generated.append(str(out_path))
                        _logger.info(f"Saved source plot: {out_path}")
                except Exception as e:
                    label = f"'{plot_name}' (flight {fid})" if fid else f"'{plot_name}'"
                    errors.append(f"Plot {label} failed: {e}")
                    _logger.warning(f"Source plot {label} failed: {e}")

        return self._create_result(
            StageStatus.COMPLETED if plot_count > 0 or not plots_config else StageStatus.SKIPPED,
            data={"plot_count": plot_count, "plots_generated": plots_generated, "errors": errors},
            duration=time.time() - start,
        )

    def execute(self, context: PipelineContext) -> StageResult:
        """Generate comparison plots from paired data, or plots from unpaired sources."""
        if not context.paired and iter_single_source_datasets(context):
            return self._execute_single_source(context)

        import time
        from pathlib import Path

        import matplotlib.pyplot as plt

        from davinci_monet.plots import get_plotter
        from davinci_monet.plots.base import build_series, format_plot_title

        start = time.time()
        plots_generated: list[str] = []

        plot_config = context.config.get("plots", {})

        # Plotting is optional - if no config, skip
        if not plot_config:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"message": "No plot configuration found"},
                duration=time.time() - start,
            )

        # Get output directory
        analysis_config = context.config.get("analysis", {})
        output_dir_str = analysis_config.get("output_dir") or "."
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get pairs config for variable mapping
        pairs_config = context.config.get("pairs", {})
        total_plots = len(plot_config)
        plot_count = 0
        file_index = 0  # Global counter for ordering files in preview

        for plot_name, plot_spec in plot_config.items():
            try:
                plot_count += 1
                plot_type = plot_spec.get("type", "scatter")
                plot_pairs = plot_spec.get("data") or plot_spec.get("pairs", [])
                title = format_plot_title(plot_spec.get("title", plot_name))

                context.log_progress(f"    Plot: {plot_name} ({plot_count}/{total_plots})")
                context.log_progress(f"step: Rendering {plot_type}...")

                for pair_name in plot_pairs:
                    # Get pair configuration
                    pair_spec = pairs_config.get(pair_name, {})
                    if not isinstance(pair_spec, dict):
                        pair_spec = {}

                    sources = [str(src) for src in pair_spec.get("sources", [])]
                    model_label = ""
                    obs_label = ""
                    var_spec: dict[str, Any] = {}

                    # Resolve the paired-data key. Unified ``sources:`` pairs and
                    # implicitly auto-paired pairs are both keyed by the pair name
                    # in ``context.paired``; for unified pairs we also recover the
                    # reference/comparand labels from the spec.
                    pair_key = pair_name
                    if "sources" in pair_spec:
                        reference_label = pair_spec.get("reference")
                        if reference_label is not None and str(reference_label) in sources:
                            obs_label = str(reference_label)
                            model_label = next(
                                (src for src in sources if src != obs_label),
                                "",
                            )
                    if pair_key not in context.paired:
                        continue

                    paired_obj = context.paired[pair_key]
                    paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

                    if "sources" in pair_spec:
                        pair_vars = iter_paired_variable_pairs(paired_data)
                        if not pair_vars:
                            continue
                        fallback_obs_name, fallback_model_name, fallback_var = pair_vars[0]
                        if not obs_label:
                            obs_label = str(
                                paired_data[fallback_obs_name].attrs.get(
                                    "source_label", "reference"
                                )
                            )
                        if not model_label:
                            model_label = str(
                                paired_data[fallback_model_name].attrs.get(
                                    "source_label", "comparand"
                                )
                            )
                        source_vars = pair_spec.get("variables") or {}
                        obs_var = str(source_vars.get(obs_label) or fallback_var)
                        model_var = str(source_vars.get(model_label) or obs_var)
                        var_spec = {"obs_var": obs_var, "model_var": model_var}
                        obs_var_name, model_var_name = resolve_paired_var_names(
                            paired_data, obs_var, obs_label, model_label
                        )
                    else:
                        # No (or empty) pair spec: a plot referencing a pair key in
                        # context.paired directly (implicit auto-pairing or a
                        # pre-paired key). Recover labels/vars from the paired
                        # data's role-tagged source-label attrs.
                        pair_vars = iter_paired_variable_pairs(paired_data)
                        if not pair_vars:
                            continue
                        obs_var_name, model_var_name, obs_var = pair_vars[0]
                        obs_label = str(
                            paired_data[obs_var_name].attrs.get("source_label", "reference")
                        )
                        model_label = str(
                            paired_data[model_var_name].attrs.get("source_label", "comparand")
                        )
                        var_spec = {"obs_var": obs_var, "model_var": obs_var}

                    # Apply per-plot domain filter if requested. domain_type/
                    # domain_name in the plot spec restrict paired_data to a
                    # named lat/lon extent (e.g. 'conus', or 'epa_region' + R5).
                    # When the type is None / "all" or unrecognised, the helper
                    # returns the dataset unchanged.
                    from davinci_monet.util.domain import filter_paired_by_domain

                    paired_data = filter_paired_by_domain(
                        paired_data,
                        plot_spec.get("domain_type"),
                        plot_spec.get("domain_name"),
                    )

                    if obs_var_name not in paired_data or model_var_name not in paired_data:
                        continue

                    # Get plotter config from source variable settings. The
                    # comparand side wins for comparison-specific plot limits;
                    # reference settings are a fallback for same-source-role runs.
                    model_var = var_spec.get("model_var", "")
                    obs_var = var_spec.get("obs_var", "")
                    var_config = self._source_var_config(context.config, model_label, model_var)
                    if not var_config:
                        var_config = self._source_var_config(context.config, obs_label, obs_var)
                    vmin = var_config.get("vmin_plot")
                    vmax = var_config.get("vmax_plot")
                    vdiff = var_config.get("vdiff_plot")
                    nlevels = var_config.get("nlevels_plot")

                    # Build plotter config
                    plotter_config: dict[str, Any] = {"title": title}
                    if plot_type == "spatial_bias":
                        plotter_config["vmin"] = -vdiff if vdiff else None
                        plotter_config["vmax"] = vdiff if vdiff else None
                    elif plot_type == "timeseries":
                        # Timeseries uses smart auto-scaling from data range
                        # Don't pass vmin/vmax unless explicitly set in plot_spec
                        if "vmin" in plot_spec:
                            plotter_config["vmin"] = plot_spec["vmin"]
                        if "vmax" in plot_spec:
                            plotter_config["vmax"] = plot_spec["vmax"]
                    else:
                        plotter_config["vmin"] = vmin
                        plotter_config["vmax"] = vmax

                    # Extract additional plot options from plot_spec
                    plot_options: dict[str, Any] = {}
                    for opt_key in [
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
                        # spatial_overlay obs marker sizing
                        "marker_size",
                        "obs_edgecolor",
                        "obs_linewidth",
                        # spatial plotter rendering mode
                        "plot_type",
                        "cmap",
                        # track_map_3d options
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
                    ]:
                        if opt_key in plot_spec:
                            plot_options[opt_key] = plot_spec[opt_key]

                    # Add city_labels from analysis config for spatial plots and 3D track maps
                    if (
                        plot_type.startswith("spatial") or plot_type == "track_map_3d"
                    ) and "city_labels" not in plot_options:
                        city_labels = analysis_config.get("city_labels")
                        if city_labels:
                            plot_options["city_labels"] = city_labels

                    # Forward nlevels_plot to spatial plotters as n_levels so
                    # configs can pick contour counts that produce nice round
                    # tick values (e.g. 21 levels over 0-1 -> step 0.05).
                    if plot_type.startswith("spatial") and nlevels is not None:
                        plot_options.setdefault("n_levels", nlevels)

                    # spatial_overlay needs the raw gridded model field for the
                    # contour layer; the paired dataset usually carries sampled
                    # values at reference locations only. Keep the renderer option
                    # name `model_field` for compatibility.
                    if plot_type == "spatial_overlay":
                        if "model_field" not in plot_options:
                            source_obj = context.sources.get(model_label) or context.sources.get(
                                obs_label
                            )
                            if source_obj is not None:
                                source_ds = (
                                    source_obj.data if hasattr(source_obj, "data") else source_obj
                                )
                                source_vars = getattr(source_ds, "data_vars", {})
                                field_var = model_var if model_var in source_vars else obs_var
                                if source_ds is not None and field_var in source_vars:
                                    plot_options["model_field"] = source_ds[field_var]
                        # Observation readers differ on coord naming
                        # (`latitude`/`longitude` vs `lat`/`lon`). Pick whichever
                        # the paired dataset actually carries.
                        if "lat_var" not in plot_options:
                            for cand in ("latitude", "lat"):
                                if cand in paired_data.coords or cand in paired_data:
                                    plot_options["lat_var"] = cand
                                    break
                        if "lon_var" not in plot_options:
                            for cand in ("longitude", "lon"):
                                if cand in paired_data.coords or cand in paired_data:
                                    plot_options["lon_var"] = cand
                                    break

                    # Build subtitle: date range (or snapshot timestamp for
                    # spatial_overlay plots).  The source-pair prefix ("Model vs
                    # Obs") has been removed — plot titles already name the
                    # sources, and the separator rendered as a missing-glyph box
                    # in the Poppins font.
                    start_time = analysis_config.get("start_time", "")
                    end_time = analysis_config.get("end_time", "")
                    date_str = ""
                    if start_time:
                        start_date = str(start_time).split(" ")[0]
                        end_date = str(end_time).split(" ")[0] if end_time else start_date
                        date_str = (
                            start_date if start_date == end_date else f"{start_date} - {end_date}"
                        )
                    snapshot_str = ""
                    if plot_type == "spatial_overlay" and "model_field" in plot_options:
                        mf = plot_options["model_field"]
                        time_idx = plot_options.get("time_index", 0)
                        if "time" in mf.dims and len(mf["time"]) > time_idx:
                            ts = mf["time"].values[time_idx]
                            try:
                                import pandas as pd

                                snapshot_str = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M UTC")
                            except Exception:
                                snapshot_str = str(ts)[:16] + " UTC"
                    when = snapshot_str or date_str
                    subtitle = when
                    # Forward per-plot axis label overrides to the plotter config
                    # so renderers like scatter can display source-named labels
                    # (e.g. "MODIS Terra AOD") instead of "Observed AOD (550 nm)".
                    label_aliases = {
                        "reference_label": "obs_label",
                        "comparand_label": "model_label",
                        "obs_label": "obs_label",
                        "model_label": "model_label",
                    }
                    for input_key, plotter_key in label_aliases.items():
                        if input_key in plot_spec:
                            plotter_config[plotter_key] = plot_spec[input_key]
                    if subtitle:
                        plotter_config["title"] = f"{title}\n{subtitle}"

                    # Get plotter
                    plotter = get_plotter(plot_type, config=plotter_config)

                    # Create subdirectory by reference source.
                    reference_output_dir = output_dir / obs_label
                    reference_output_dir.mkdir(parents=True, exist_ok=True)

                    # Check for per-flight splitting
                    split_by_flight = plot_spec.get("split_by_flight", False)
                    # Check for per-site splitting
                    split_by_site = plot_spec.get("split_by_site", False)

                    if split_by_flight and hasattr(plotter, "plot_per_flight"):
                        # Generate separate plot for each flight
                        flight_coord = plot_spec.get("flight_coord", "flight")
                        min_points = plot_spec.get("min_points", 10)

                        flight_count = 0
                        for flight_id, fig in plotter.plot_per_flight(
                            paired_data,
                            obs_var_name,
                            model_var_name,
                            flight_coord=flight_coord,
                            min_points=min_points,
                            **plot_options,
                        ):
                            # Save plot with flight ID first for grouping by flight in slideshows
                            output_path = (
                                reference_output_dir
                                / f"{flight_id}_{file_index:02d}_{plot_name}.png"
                            )
                            plotter.save(fig, output_path, dpi=300)
                            plots_generated.append(str(output_path))

                            pdf_path = (
                                reference_output_dir
                                / f"{flight_id}_{file_index:02d}_{plot_name}.pdf"
                            )
                            plotter.save(fig, pdf_path)
                            plots_generated.append(str(pdf_path))

                            plt.close(fig)
                            flight_count += 1
                            file_index += 1

                        context.log_progress(f"done: saved {flight_count} flights to {obs_label}/")

                    elif split_by_site and hasattr(plotter, "plot_per_site"):
                        # Generate separate plot for each site
                        site_dim = plot_spec.get("site_dim", "site")
                        min_points = plot_spec.get("min_points", 20)

                        site_count = 0
                        for site_id, fig in plotter.plot_per_site(
                            paired_data,
                            obs_var_name,
                            model_var_name,
                            site_dim=site_dim,
                            min_points=min_points,
                            **plot_options,
                        ):
                            output_path = (
                                reference_output_dir
                                / f"site_{site_id}_{file_index:02d}_{plot_name}.png"
                            )
                            plotter.save(fig, output_path, dpi=300)
                            plots_generated.append(str(output_path))

                            pdf_path = (
                                reference_output_dir
                                / f"site_{site_id}_{file_index:02d}_{plot_name}.pdf"
                            )
                            plotter.save(fig, pdf_path)
                            plots_generated.append(str(pdf_path))

                            plt.close(fig)
                            site_count += 1
                            file_index += 1

                        context.log_progress(f"done: saved {site_count} sites to {obs_label}/")
                    else:
                        # Generate single plot via the unified render contract.
                        fig = plotter.render(
                            build_series(paired_data, obs_var_name, model_var_name), **plot_options
                        )

                        # Save plot (prefixed for ordering)
                        output_path = reference_output_dir / f"{file_index:02d}_{plot_name}.png"
                        plotter.save(fig, output_path, dpi=300)
                        plots_generated.append(str(output_path))

                        # Also save PDF
                        pdf_path = reference_output_dir / f"{file_index:02d}_{plot_name}.pdf"
                        plotter.save(fig, pdf_path)
                        plots_generated.append(str(pdf_path))

                        plt.close(fig)
                        file_index += 1

                        context.log_progress(f"done: saved to {obs_label}/")

            except Exception as e:
                context.metadata.setdefault("plot_errors", []).append(f"{plot_name}: {e}")
                context.log_progress(f"warning: plot failed for {plot_name}: {e}")

        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": plots_generated},
            duration=time.time() - start,
        )
