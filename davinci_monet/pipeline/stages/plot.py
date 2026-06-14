"""Plotting stage.

Generates comparison plots from paired data or single-source plots from
unpaired data through the unified renderer ``render(series)`` contract.
"""

from __future__ import annotations

from typing import Any

from davinci_monet.core.base import iter_paired_variable_xy, paired_canonical_name
from davinci_monet.core.schema_utils import dump_schema, is_schema_object
from davinci_monet.pipeline.stages.base import (
    BaseStage,
    PipelineContext,
    StageResult,
    StageStatus,
)
from davinci_monet.pipeline.stages.helpers import (
    iter_single_source_datasets,
    tag_source_label,
)
from davinci_monet.pipeline.stages.plot_options import (
    build_comparison_plot_options,
    build_plot_subtitle,
    single_source_flight_plot_kwargs,
    single_source_plot_kwargs,
    timestamp_from_field,
)


class PlottingStage(BaseStage):
    """Stage for generating plots from paired or single-source data."""

    def __init__(self) -> None:
        super().__init__("plotting")

    @staticmethod
    def _config_dict(value: Any) -> dict[str, Any]:
        if is_schema_object(value):
            return dump_schema(value, exclude_none=True)
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
        """Return variable plot config for a source label."""
        sources_cfg = config.get("sources", {})
        if not isinstance(sources_cfg, dict):
            return {}
        source_cfg = cls._config_dict(sources_cfg.get(source_label, {}))
        variables = cls._config_dict(source_cfg.get("variables", {}))
        if variable_name in variables:
            return cls._config_dict(variables[variable_name])
        return {}

    @staticmethod
    def _resolve_paired_dataset_variable(
        paired_data: Any,
        *,
        source_label: str,
        y_variable: str,
        axis: str,
        fallback_name: str | None = None,
    ) -> str | None:
        """Resolve a variable from unified paired metadata."""
        candidates = list(paired_data.data_vars)
        if fallback_name in paired_data.data_vars:
            candidates.insert(0, str(fallback_name))

        seen: set[str] = set()
        requested = str(y_variable)
        requested_lower = requested.lower()
        for name in candidates:
            name = str(name)
            if name in seen:
                continue
            seen.add(name)
            attrs = paired_data[name].attrs
            if attrs.get("axis") != axis:
                continue
            if attrs.get("source_label") != source_label:
                continue
            actual_source_var = str(attrs.get("dataset_variable") or "")
            if actual_source_var and actual_source_var.lower() == requested_lower:
                return name
            if paired_canonical_name(paired_data, name).lower() == requested_lower:
                return name
            if name.lower() == requested_lower:
                return name
        return None

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
        config = context.config_dict()
        plots_config = config.get("plots", {})
        analysis_config = config.get("analysis", {})
        output_dir = Path(analysis_config.get("output_dir", "."))
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_count = 0
        plots_generated: list[str] = []
        errors: list[str] = []
        source_map = {label: (obj, ds) for label, obj, ds in iter_single_source_datasets(context)}

        for plot_name, plot_spec in plots_config.items():
            plot_type = plot_spec.get("type", "")
            # Single-source specs carry a ``source:`` key.
            if not plot_type or "source" not in plot_spec:
                continue

            source_label = str(plot_spec.get("source") or "")
            variable = plot_spec.get("variable", "")

            if source_label not in source_map:
                errors.append(f"Source '{source_label}' not found for plot '{plot_name}'")
                continue

            _source_obj, ds = source_map[source_label]

            if variable not in ds.data_vars:
                errors.append(
                    f"Variable '{variable}' not in source '{source_label}' for plot '{plot_name}'"
                )
                continue

            plotter = get_plotter(plot_type)
            plot_kwargs = single_source_plot_kwargs(
                plot_spec,
                analysis_config=analysis_config,
            )

            has_flights = "flight" in ds.coords
            flight_ids = (
                sorted(set(np.unique(ds["flight"].values).tolist())) if has_flights else [None]
            )

            for fid in flight_ids:
                if fid is not None:
                    mask = ds["flight"].values == fid
                    subset = ds.isel(time=mask)
                    suffix = f"_{fid}"
                    flight_base_kwargs = (
                        plot_kwargs
                        if plot_kwargs.get("title")
                        else {**plot_kwargs, "title": f"{variable} {plot_type}"}
                    )
                    flight_kwargs = single_source_flight_plot_kwargs(
                        flight_base_kwargs,
                        flight_id=fid,
                    )
                else:
                    subset = ds
                    suffix = ""
                    flight_kwargs = plot_kwargs

                vals = subset[variable].values
                if not np.any(np.isfinite(vals)):
                    continue

                if "flight_tracks" in plot_spec:
                    flight_kwargs["x_datasets"] = {
                        label: source_ds for label, (_obj, source_ds) in source_map.items()
                    }

                try:
                    # Tag the single source so build_series picks up its source label.
                    tag_source_label(subset, source_label=source_label)
                    render_kwargs = dict(flight_kwargs)
                    plotter.config.subtitle = render_kwargs.pop("subtitle", None)
                    result = plotter.render(build_series(subset, variable), **render_kwargs)
                    if isinstance(result, list):
                        for fig, fig_suffix in result:
                            out_path = output_dir / f"{plot_name}{suffix}{fig_suffix}.png"
                            plotter.save(fig, out_path)
                            plots_generated.append(str(out_path))
                            # Also save PDF (parity with the comparison path)
                            pdf_path = output_dir / f"{plot_name}{suffix}{fig_suffix}.pdf"
                            plotter.save(fig, pdf_path)
                            plots_generated.append(str(pdf_path))
                            plt.close(fig)
                            plot_count += 1
                            _logger.info(f"Saved source plot: {out_path}")
                    else:
                        fig = result
                        out_path = output_dir / f"{plot_name}{suffix}.png"
                        plotter.save(fig, out_path)
                        plots_generated.append(str(out_path))
                        # Also save PDF (parity with the comparison path)
                        pdf_path = output_dir / f"{plot_name}{suffix}.pdf"
                        plotter.save(fig, pdf_path)
                        plots_generated.append(str(pdf_path))
                        plt.close(fig)
                        plot_count += 1
                        _logger.info(f"Saved source plot: {out_path}")
                except Exception as e:
                    label = f"'{plot_name}' (flight {fid})" if fid else f"'{plot_name}'"
                    errors.append(f"Plot {label} failed: {e}")
                    _logger.warning(f"Source plot {label} failed: {e}")

        if errors:
            context.metadata.setdefault("plot_errors", []).extend(errors)
            return self._create_result(
                StageStatus.FAILED,
                data={
                    "plot_count": plot_count,
                    "plots_generated": plots_generated,
                    "errors": errors,
                },
                error="Plotting failed: " + "; ".join(errors),
                duration=time.time() - start,
            )

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

        Returns ``(paired_data, x_source, y_source, var_spec, geometry_var_name,
        dataset_var_name)`` or ``None`` when the pair should be skipped (mirrors the
        ``continue`` branches of the original loop body).
        """
        x_axis = pair_spec.get("x") if isinstance(pair_spec.get("x"), dict) else None
        y_axis = pair_spec.get("y") if isinstance(pair_spec.get("y"), dict) else None
        has_xy = x_axis is not None and y_axis is not None
        y_source = ""
        x_source = ""
        var_spec: dict[str, Any] = {}

        # Resolve the paired-data key. Unified ``x:``/``y:`` pairs and
        # implicitly auto-paired pairs are both keyed by the pair name
        # in ``context.paired``; for unified pairs we also recover the
        # x (reference) / y (sampled) labels from the spec.
        pair_key = pair_name
        if has_xy:
            assert x_axis is not None and y_axis is not None  # narrow for mypy
            x_source = str(x_axis.get("source") or "")
            y_source = str(y_axis.get("source") or "")
        if pair_key not in context.paired:
            return None

        paired_obj = context.paired[pair_key]
        paired_data = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

        if has_xy:
            assert x_axis is not None and y_axis is not None  # narrow for mypy
            pair_vars = iter_paired_variable_xy(paired_data)
            if not pair_vars:
                return None
            fallback_geometry_name, fallback_dataset_name, fallback_var = pair_vars[0]
            if not x_source:
                x_source = str(
                    paired_data[fallback_geometry_name].attrs.get("source_label", "geometry")
                )
            if not y_source:
                y_source = str(
                    paired_data[fallback_dataset_name].attrs.get("source_label", "dataset")
                )
            x_var = str(
                x_axis.get("variable")
                or paired_data[fallback_geometry_name].attrs.get("dataset_variable")
                or fallback_var
            )
            y_var = str(
                y_axis.get("variable")
                or paired_data[fallback_dataset_name].attrs.get("dataset_variable")
                or fallback_var
            )
            var_spec = {
                "x_var": x_var,
                "y_var": y_var,
            }
            x_var_name = self._resolve_paired_dataset_variable(
                paired_data,
                source_label=x_source,
                y_variable=x_var,
                axis="x",
                fallback_name=fallback_geometry_name,
            )
            y_var_name = self._resolve_paired_dataset_variable(
                paired_data,
                source_label=y_source,
                y_variable=y_var,
                axis="y",
                fallback_name=fallback_dataset_name,
            )
            if x_var_name is None or y_var_name is None:
                return None
        else:
            # No pair spec: a plot can name a key already present in
            # context.paired. Recover labels and variables from paired attrs.
            pair_vars = iter_paired_variable_xy(paired_data)
            if not pair_vars:
                return None
            x_var_name, y_var_name, x_var = pair_vars[0]
            x_source = str(paired_data[x_var_name].attrs.get("source_label", "x"))
            y_source = str(paired_data[y_var_name].attrs.get("source_label", "y"))
            var_spec = {"x_var": x_var, "y_var": x_var}

        return (
            paired_data,
            x_source,
            y_source,
            var_spec,
            x_var_name,
            y_var_name,
        )

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
        x_source: str,
        y_source: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Assemble the ``(plotter_config, plot_options)`` for a comparison plot.

        Resolves per-source variable limits, builds the renderer option dict
        (including ``spatial_overlay`` y-field/coord wiring), the title
        subtitle, and the axis-label overrides.
        """
        # Get plotter config from source variable settings. The y side
        # wins for comparison-specific plot limits; x settings are a
        # fallback.
        config = context.config_dict()
        y_var = var_spec.get("y_var", "")
        x_var = var_spec.get("x_var", "")
        var_config = self._source_var_config(config, y_source, y_var)
        if not var_config:
            var_config = self._source_var_config(config, x_source, x_var)
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

        # spatial_overlay needs the raw gridded y field for the
        # contour layer; the paired dataset usually carries sampled
        # values at x locations only. The renderer option name is
        # `y_field`.
        if plot_type == "spatial_overlay":
            if "y_field" not in plot_options:
                source_obj = context.sources.get(y_source) or context.sources.get(x_source)
                if source_obj is not None:
                    source_ds = source_obj.data if hasattr(source_obj, "data") else source_obj
                    source_vars = getattr(source_ds, "data_vars", {})
                    field_var = y_var if y_var in source_vars else x_var
                    if source_ds is not None and field_var in source_vars:
                        plot_options["y_field"] = source_ds[field_var]
            # Dataset readers differ on coord naming
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

        snapshot_str = ""
        if plot_type == "spatial_overlay" and "y_field" in plot_options:
            mf = plot_options["y_field"]
            time_idx = plot_options.get("time_index", 0)
            snapshot_str = timestamp_from_field(mf, time_idx)
        subtitle = build_plot_subtitle(
            analysis_config,
            snapshot_timestamp=snapshot_str,
        )
        if subtitle:
            plotter_config["subtitle"] = subtitle

        # Forward per-plot axis label overrides to the plotter config
        # so renderers like scatter can display source-named labels
        # (e.g. "MODIS Terra AOD") instead of "Dataset AOD (550 nm)".
        label_aliases = {
            "x_label": "x_label",
            "y_label": "y_label",
        }
        for input_key, plotter_key in label_aliases.items():
            if input_key in plot_spec:
                plotter_config[plotter_key] = plot_spec[input_key]

        return plotter_config, plot_options

    @staticmethod
    def _save_per_flight(
        *,
        plotter: Any,
        paired_data: Any,
        x_var_name: str,
        y_var_name: str,
        plot_spec: dict[str, Any],
        plot_options: dict[str, Any],
        x_source_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        x_source: str,
    ) -> int:
        """Render and save one figure per flight; return the advanced file_index."""
        import matplotlib.pyplot as plt

        flight_coord = plot_spec.get("flight_coord", "flight")
        min_points = plot_spec.get("min_points", 10)

        flight_count = 0
        for flight_id, fig in plotter.plot_per_flight(
            paired_data,
            x_var_name,
            y_var_name,
            flight_coord=flight_coord,
            min_points=min_points,
            **plot_options,
        ):
            # Save plot with flight ID first for grouping by flight in slideshows
            output_path = x_source_output_dir / f"{flight_id}_{file_index:02d}_{plot_name}.png"
            plotter.save(fig, output_path, dpi=300)
            plots_generated.append(str(output_path))

            pdf_path = x_source_output_dir / f"{flight_id}_{file_index:02d}_{plot_name}.pdf"
            plotter.save(fig, pdf_path)
            plots_generated.append(str(pdf_path))

            plt.close(fig)
            flight_count += 1
            file_index += 1

        context.log_progress(f"done: saved {flight_count} flights to {x_source}/")
        return file_index

    @staticmethod
    def _save_per_site(
        *,
        plotter: Any,
        paired_data: Any,
        x_var_name: str,
        y_var_name: str,
        plot_spec: dict[str, Any],
        plot_options: dict[str, Any],
        x_source_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        x_source: str,
    ) -> int:
        """Render and save one figure per site; return the advanced file_index."""
        import matplotlib.pyplot as plt

        site_dim = plot_spec.get("site_dim", "site")
        min_points = plot_spec.get("min_points", 20)

        site_count = 0
        for site_id, fig in plotter.plot_per_site(
            paired_data,
            x_var_name,
            y_var_name,
            site_dim=site_dim,
            min_points=min_points,
            **plot_options,
        ):
            output_path = x_source_output_dir / f"site_{site_id}_{file_index:02d}_{plot_name}.png"
            plotter.save(fig, output_path, dpi=300)
            plots_generated.append(str(output_path))

            pdf_path = x_source_output_dir / f"site_{site_id}_{file_index:02d}_{plot_name}.pdf"
            plotter.save(fig, pdf_path)
            plots_generated.append(str(pdf_path))

            plt.close(fig)
            site_count += 1
            file_index += 1

        context.log_progress(f"done: saved {site_count} sites to {x_source}/")
        return file_index

    @staticmethod
    def _save_single(
        *,
        plotter: Any,
        paired_data: Any,
        x_var_name: str,
        y_var_name: str,
        plot_options: dict[str, Any],
        x_source_output_dir: Any,
        plot_name: str,
        file_index: int,
        plots_generated: list[str],
        context: PipelineContext,
        x_source: str,
    ) -> int:
        """Render and save a single figure via the unified render contract.

        Returns the advanced file_index.
        """
        import matplotlib.pyplot as plt

        from davinci_monet.plots.base import build_series

        # Generate single plot via the unified render contract.
        fig = plotter.render(build_series(paired_data, x_var_name, y_var_name), **plot_options)

        # Save plot (prefixed for ordering)
        output_path = x_source_output_dir / f"{file_index:02d}_{plot_name}.png"
        plotter.save(fig, output_path, dpi=300)
        plots_generated.append(str(output_path))

        # Also save PDF
        pdf_path = x_source_output_dir / f"{file_index:02d}_{plot_name}.pdf"
        plotter.save(fig, pdf_path)
        plots_generated.append(str(pdf_path))

        plt.close(fig)
        file_index += 1

        context.log_progress(f"done: saved to {x_source}/")
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
            raise ValueError(f"Pair '{pair_name}' not found or has no matching x/y variables")
        (
            paired_data,
            x_source,
            y_source,
            var_spec,
            x_var_name,
            y_var_name,
        ) = resolved

        paired_data = self._apply_domain_filter(paired_data, plot_spec)

        if x_var_name not in paired_data or y_var_name not in paired_data:
            raise ValueError(
                f"Pair '{pair_name}' missing resolved variables "
                f"'{x_var_name}' and/or '{y_var_name}'"
            )

        plotter_config, plot_options = self._resolve_plot_options(
            context=context,
            plot_type=plot_type,
            plot_spec=plot_spec,
            analysis_config=analysis_config,
            title=title,
            paired_data=paired_data,
            var_spec=var_spec,
            x_source=x_source,
            y_source=y_source,
        )

        # Get plotter
        plotter = get_plotter(plot_type, config=plotter_config)

        # Create subdirectory by geometry source.
        x_source_output_dir = output_dir / x_source
        x_source_output_dir.mkdir(parents=True, exist_ok=True)

        # Check for per-flight splitting
        split_by_flight = plot_spec.get("split_by_flight", False)
        # Check for per-site splitting
        split_by_site = plot_spec.get("split_by_site", False)

        if split_by_flight and hasattr(plotter, "plot_per_flight"):
            # Generate separate plot for each flight
            return self._save_per_flight(
                plotter=plotter,
                paired_data=paired_data,
                x_var_name=x_var_name,
                y_var_name=y_var_name,
                plot_spec=plot_spec,
                plot_options=plot_options,
                x_source_output_dir=x_source_output_dir,
                plot_name=plot_name,
                file_index=file_index,
                plots_generated=plots_generated,
                context=context,
                x_source=x_source,
            )
        elif split_by_site and hasattr(plotter, "plot_per_site"):
            # Generate separate plot for each site
            return self._save_per_site(
                plotter=plotter,
                paired_data=paired_data,
                x_var_name=x_var_name,
                y_var_name=y_var_name,
                plot_spec=plot_spec,
                plot_options=plot_options,
                x_source_output_dir=x_source_output_dir,
                plot_name=plot_name,
                file_index=file_index,
                plots_generated=plots_generated,
                context=context,
                x_source=x_source,
            )
        else:
            return self._save_single(
                plotter=plotter,
                paired_data=paired_data,
                x_var_name=x_var_name,
                y_var_name=y_var_name,
                plot_options=plot_options,
                x_source_output_dir=x_source_output_dir,
                plot_name=plot_name,
                file_index=file_index,
                plots_generated=plots_generated,
                context=context,
                x_source=x_source,
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

        config = context.config_dict()
        plot_config = config.get("plots", {})

        # Plotting is optional - if no config, skip
        if not plot_config:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"message": "No plot configuration found"},
                duration=time.time() - start,
            )

        # Get output directory
        analysis_config = config.get("analysis", {})
        output_dir_str = analysis_config.get("output_dir") or "."
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get pairs config for variable mapping
        pairs_config = config.get("pairs", {})
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

        errors = context.metadata.get("plot_errors") or []
        if errors:
            return self._create_result(
                StageStatus.FAILED,
                data={"plots_generated": plots_generated},
                error="Plotting failed: " + "; ".join(str(e) for e in errors),
                duration=time.time() - start,
            )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": plots_generated},
            duration=time.time() - start,
        )
