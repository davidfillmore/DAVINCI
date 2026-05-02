"""Run command for DAVINCI CLI.

This module implements the main analysis execution command.
"""

from __future__ import annotations

from pathlib import Path

import typer

from davinci_monet.cli.app import (
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    display_error,
)
from davinci_monet.core.exceptions import ConfigurationError, PipelineError


def _is_plume_sentinel_workflow(control_path: str) -> bool:
    """Peek at the control file to see if it declares the plume_sentinel workflow.

    Used to decide whether the plume_sentinel-specific flags
    (``--emit-metrics-json``, ``--run-id``, etc.) are applicable.
    """
    try:
        import yaml

        with open(control_path) as f:
            doc = yaml.safe_load(f)
        analysis = (doc or {}).get("analysis") or {}
        return analysis.get("workflow") == "plume_sentinel"
    except Exception:
        return False


def run_analysis(
    control_path: str,
    debug: bool = False,
    show_plots: bool = False,
    preview_format: str = "pdf",
    open_plots: bool = False,
    output_dir: str | None = None,
    emit_metrics_json: str | None = None,
    run_id: str | None = None,
    region: str | None = None,
    config_slug: str | None = None,
    event_date: str | None = None,
) -> None:
    """Execute DAVINCI analysis from a control file.

    Parameters
    ----------
    control_path
        Path to the YAML control file.
    debug
        If True, show full tracebacks on error.
    show_plots
        If True, display interactive plot preview after completion.
    preview_format
        Format for plot preview: "pdf" or "png".
    open_plots
        If True, open generated plot files after the run.
    output_dir
        If provided, override analysis.output_dir (plume_sentinel only).
    emit_metrics_json
        If provided, write a plumesentinel.metrics.v1 JSON sidecar
        (plume_sentinel only).
    run_id
        Run identifier embedded in the metrics payload.
    region, config_slug
        Provenance tags included in the metrics payload.
    event_date
        If provided (YYYY-MM-DD), overrides the config's
        ``analysis.start_time`` / ``analysis.end_time`` to span the
        given UTC day before the workflow runs (plume_sentinel only).
        Lets one config drive multiple manifest dates.
    """
    # Update global debug flag
    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    p = Path(control_path)
    if not p.is_file():
        typer.secho(f"Error: control file {control_path!r} does not exist", fg=ERROR_COLOR)
        raise typer.Exit(2)

    plume_sentinel = _is_plume_sentinel_workflow(str(p))

    # Validate plume_sentinel-only flag usage early so we don't silently
    # ignore them on non-matching configs.
    ps_flags = {
        "--emit-metrics-json": emit_metrics_json,
        "--run-id": run_id,
        "--region": region,
        "--config-slug": config_slug,
        "--output-dir": output_dir,
        "--event-date": event_date,
    }
    if not plume_sentinel:
        provided = [name for name, value in ps_flags.items() if value is not None]
        if provided:
            typer.secho(
                "Warning: "
                + ", ".join(provided)
                + " only apply to plume_sentinel workflows; ignoring.",
                fg=WARNING_COLOR,
            )

    try:
        if plume_sentinel and emit_metrics_json is not None:
            # Use the addon's high-level entry point so we can capture
            # stage outputs and serialize the metrics sidecar.
            if not run_id:
                typer.secho(
                    "Warning: --emit-metrics-json was set without --run-id; "
                    "the metrics payload will use a synthesized run_id.",
                    fg=WARNING_COLOR,
                )
            from davinci_monet.addons.plume_sentinel import workflow as ps_workflow

            result = ps_workflow.run(
                str(p),
                output_dir=output_dir,
                emit_metrics_json=emit_metrics_json,
                run_id=run_id,
                region=region,
                config_slug=config_slug,
                event_date=event_date,
            )
        else:
            from davinci_monet.pipeline.runner import run_analysis as pipeline_run

            # If the user passed --output-dir or --event-date for a
            # plume_sentinel workflow without --emit-metrics-json, still
            # honor them by editing the config dict before run.
            if plume_sentinel and (output_dir is not None or event_date is not None):
                from davinci_monet.config import load_config

                cfg = load_config(str(p)).model_dump()
                cfg.setdefault("analysis", {})
                if output_dir is not None:
                    cfg["analysis"]["output_dir"] = str(output_dir)
                if event_date is not None:
                    from davinci_monet.addons.plume_sentinel.workflow import (
                        _apply_event_date_override,
                    )

                    _apply_event_date_override(cfg, event_date)
                from davinci_monet.pipeline.runner import PipelineRunner

                runner = PipelineRunner(
                    show_progress=True,
                    show_plots=show_plots,
                    preview_format=preview_format,  # type: ignore[arg-type]
                    open_plots=open_plots,
                )
                result = runner.run_from_config(cfg)
            else:
                result = pipeline_run(
                    str(p), show_progress=True, show_plots=show_plots,
                    preview_format=preview_format, open_plots=open_plots,  # type: ignore[arg-type]
                )
    except ConfigurationError as e:
        # Styled display for configuration/YAML errors (before pipeline starts)
        display_error("Configuration Error", str(e), config_path=control_path)
        if debug:
            raise
        raise typer.Exit(1)
    except PipelineError as e:
        # Styled display for pipeline errors
        display_error("Pipeline Error", str(e), config_path=control_path)
        if debug:
            raise
        raise typer.Exit(1)
    except Exception as e:
        # Styled display for unexpected errors
        display_error("Error", str(e), config_path=control_path)
        if debug:
            raise
        raise typer.Exit(1)

    # Report results (success case only - failure shown by pipeline footer)
    if result.success:
        typer.echo()
        typer.secho(
            f"Analysis complete! ({result.total_duration_seconds:.1f}s)",
            fg=SUCCESS_COLOR,
        )
        typer.secho(
            f"Stages: {', '.join(result.completed_stages)}",
            fg=INFO_COLOR,
        )
        if emit_metrics_json:
            typer.secho(
                f"Metrics sidecar: {emit_metrics_json}",
                fg=INFO_COLOR,
            )
    else:
        # Pipeline footer already shows failure - just exit with error code
        raise typer.Exit(1)
