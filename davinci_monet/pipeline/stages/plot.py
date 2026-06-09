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
from davinci_monet.pipeline.stages.plot_options import (
    build_comparison_plot_options,
    build_plot_subtitle,
    single_source_plot_kwargs,
    timestamp_from_field,
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
            plot_kwargs = single_source_plot_kwargs(plot_spec)
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

    def _resolve_pair_labels_and_vars(
        self,
        pair_name: str,
        pair_spec: dict[str, Any],
        context: PipelineContext,
    ) -> tuple[Any, str, str, dict[str, Any], str, str] | None:
        """Resolve the paired dataset, source labels, and variable names for a pair.

        Returns ``(paired_data, obs_label, model_label, var_spec, obs_var_name,
        model_var_name)`` or ``None`` when the pair should be skipped (mirrors the
        ``continue`` branches of the original loop body).
        """
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
            return None

        paired_obj = context.paired[pair_key]
        paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

        if "sources" in pair_spec:
            pair_vars = iter_paired_variable_pairs(paired_data)
            if not pair_vars:
                return None
            fallback_obs_name, fallback_model_name, fallback_var = pair_vars[0]
            if not obs_label:
                obs_label = str(
                    paired_data[fallback_obs_name].attrs.get("source_label", "reference")
                )
            if not model_label:
                model_label = str(
                    paired_data[fallback_model_name].attrs.get("source_label", "comparand")
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
                return None
            obs_var_name, model_var_name, obs_var = pair_vars[0]
            obs_label = str(paired_data[obs_var_name].attrs.get("source_label", "reference"))
            model_label = str(paired_data[model_var_name].attrs.get("source_label", "comparand"))
            var_spec = {"obs_var": obs_var, "model_var": obs_var}

        return paired_data, obs_label, model_label, var_spec, obs_var_name, model_var_name

    @staticmethod
    def _apply_domain_filter(paired_data: Any, plot_spec: dict[str, Any]) -> Any:
        """Apply a per-plot domain filter to the paired data, if requested.

        ``domain_type``/``domain_name`` in the plot spec restrict ``paired_data``
        to a named lat/lon extent (e.g. 'conus', or 'epa_region' + R5). When the
        type is None / "all" or unrecognised, the helper returns the dataset
        unchanged.
        """
        from davinci_monet.util.domain import filter_paired_by_domain

        return filter_paired_by_domain(
            paired_data,
            plot_spec.get("domain_type"),
            plot_spec.get("domain_name"),
        )

    def _resolve_plot_options(
        self,
        *,
        context: PipelineContext,
        plot_type: str,
        plot_spec: dict[str, Any],
        analysis_config: dict[str, Any],
        title: str,
        paired_data: Any,
        var_spec: dict[str, Any],
        obs_label: str,
        model_label: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Assemble the ``(plotter_config, plot_options)`` for a comparison plot.

        Resolves per-source variable limits, builds the renderer option dict
        (including ``spatial_overlay`` model-field/coord wiring), the subtitle, and
        the axis-label overrides.
        """
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

        plot_options = build_comparison_plot_options(
            plot_type,
            plot_spec,
            analysis_config,
            nlevels=nlevels,
        )

        # spatial_overlay needs the raw gridded model field for the
        # contour layer; the paired dataset usually carries sampled
        # values at reference locations only. Keep the renderer option
        # name `model_field` for compatibility.
        if plot_type == "spatial_overlay":
            if "model_field" not in plot_options:
                source_obj = context.sources.get(model_label) or context.sources.get(obs_label)
                if source_obj is not None:
                    source_ds = source_obj.data if hasattr(source_obj, "data") else source_obj
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
        snapshot_str = ""
        if plot_type == "spatial_overlay" and "model_field" in plot_options:
            mf = plot_options["model_field"]
            time_idx = plot_options.get("time_index", 0)
            snapshot_str = timestamp_from_field(mf, time_idx)
        subtitle = build_plot_subtitle(
            analysis_config,
            snapshot_timestamp=snapshot_str,
        )
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

        return plotter_config, plot_options

    @staticmethod
    def _save_per_flight(
        *,
        plotter: Any,
        paired_data: Any,
        obs_var_name: str,
        model_var_name: str,
        plot_spec: dict[str, Any],
        plot_options: dict[str, Any],
        reference_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        obs_label: str,
    ) -> int:
        """Render and save one figure per flight; return the advanced file_index."""
        import matplotlib.pyplot as plt

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
            output_path = reference_output_dir / f"{flight_id}_{file_index:02d}_{plot_name}.png"
            plotter.save(fig, output_path, dpi=300)
            plots_generated.append(str(output_path))

            pdf_path = reference_output_dir / f"{flight_id}_{file_index:02d}_{plot_name}.pdf"
            plotter.save(fig, pdf_path)
            plots_generated.append(str(pdf_path))

            plt.close(fig)
            flight_count += 1
            file_index += 1

        context.log_progress(f"done: saved {flight_count} flights to {obs_label}/")
        return file_index

    @staticmethod
    def _save_per_site(
        *,
        plotter: Any,
        paired_data: Any,
        obs_var_name: str,
        model_var_name: str,
        plot_spec: dict[str, Any],
        plot_options: dict[str, Any],
        reference_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        obs_label: str,
    ) -> int:
        """Render and save one figure per site; return the advanced file_index."""
        import matplotlib.pyplot as plt

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
            output_path = reference_output_dir / f"site_{site_id}_{file_index:02d}_{plot_name}.png"
            plotter.save(fig, output_path, dpi=300)
            plots_generated.append(str(output_path))

            pdf_path = reference_output_dir / f"site_{site_id}_{file_index:02d}_{plot_name}.pdf"
            plotter.save(fig, pdf_path)
            plots_generated.append(str(pdf_path))

            plt.close(fig)
            site_count += 1
            file_index += 1

        context.log_progress(f"done: saved {site_count} sites to {obs_label}/")
        return file_index

    @staticmethod
    def _save_single(
        *,
        plotter: Any,
        paired_data: Any,
        obs_var_name: str,
        model_var_name: str,
        plot_options: dict[str, Any],
        reference_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        obs_label: str,
    ) -> int:
        """Render and save a single figure via the unified render contract.

        Returns the advanced file_index.
        """
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import build_series

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
        return file_index

    def _render_pair(
        self,
        *,
        context: PipelineContext,
        plot_name: str,
        pair_name: str,
        pairs_config: dict[str, Any],
        plot_type: str,
        plot_spec: dict[str, Any],
        analysis_config: dict[str, Any],
        title: str,
        output_dir: Any,
        file_index: int,
        plots_generated: list[str],
    ) -> int:
        """Render and save all figures for one (plot, pair) combination.

        Returns the advanced ``file_index``. Returns it unchanged when the pair is
        skipped (mirrors the ``continue`` branches of the original loop body).
        """
        from davinci_monet.plots import get_plotter

        # Get pair configuration
        pair_spec = pairs_config.get(pair_name, {})
        if not isinstance(pair_spec, dict):
            pair_spec = {}

        resolved = self._resolve_pair_labels_and_vars(pair_name, pair_spec, context)
        if resolved is None:
            return file_index
        (
            paired_data,
            obs_label,
            model_label,
            var_spec,
            obs_var_name,
            model_var_name,
        ) = resolved

        paired_data = self._apply_domain_filter(paired_data, plot_spec)

        if obs_var_name not in paired_data or model_var_name not in paired_data:
            return file_index

        plotter_config, plot_options = self._resolve_plot_options(
            context=context,
            plot_type=plot_type,
            plot_spec=plot_spec,
            analysis_config=analysis_config,
            title=title,
            paired_data=paired_data,
            var_spec=var_spec,
            obs_label=obs_label,
            model_label=model_label,
        )

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
            return self._save_per_flight(
                plotter=plotter,
                paired_data=paired_data,
                obs_var_name=obs_var_name,
                model_var_name=model_var_name,
                plot_spec=plot_spec,
                plot_options=plot_options,
                reference_output_dir=reference_output_dir,
                plot_name=plot_name,
                file_index=file_index,
                plots_generated=plots_generated,
                context=context,
                obs_label=obs_label,
            )
        elif split_by_site and hasattr(plotter, "plot_per_site"):
            # Generate separate plot for each site
            return self._save_per_site(
                plotter=plotter,
                paired_data=paired_data,
                obs_var_name=obs_var_name,
                model_var_name=model_var_name,
                plot_spec=plot_spec,
                plot_options=plot_options,
                reference_output_dir=reference_output_dir,
                plot_name=plot_name,
                file_index=file_index,
                plots_generated=plots_generated,
                context=context,
                obs_label=obs_label,
            )
        else:
            return self._save_single(
                plotter=plotter,
                paired_data=paired_data,
                obs_var_name=obs_var_name,
                model_var_name=model_var_name,
                plot_options=plot_options,
                reference_output_dir=reference_output_dir,
                plot_name=plot_name,
                file_index=file_index,
                plots_generated=plots_generated,
                context=context,
                obs_label=obs_label,
            )

    def execute(self, context: PipelineContext) -> StageResult:
        """Generate comparison plots from paired data, or plots from unpaired sources."""
        if not context.paired and iter_single_source_datasets(context):
            return self._execute_single_source(context)

        import time
        from pathlib import Path

        from davinci_monet.plots.base import format_plot_title

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
                    file_index = self._render_pair(
                        context=context,
                        pair_name=pair_name,
                        pairs_config=pairs_config,
                        plot_type=plot_type,
                        plot_spec=plot_spec,
                        analysis_config=analysis_config,
                        title=title,
                        output_dir=output_dir,
                        file_index=file_index,
                        plots_generated=plots_generated,
                        plot_name=plot_name,
                    )

            except Exception as e:
                context.metadata.setdefault("plot_errors", []).append(f"{plot_name}: {e}")
                context.log_progress(f"warning: plot failed for {plot_name}: {e}")

        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": plots_generated},
            duration=time.time() - start,
        )
