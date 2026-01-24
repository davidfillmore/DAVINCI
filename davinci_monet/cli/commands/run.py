"""Run command for DAVINCI-MONET CLI.

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
)


def run_analysis(
    control_path: str,
    debug: bool = False,
    show_plots: bool = False,
    preview_format: str = "pdf",
) -> None:
    """Execute DAVINCI-MONET analysis from a control file.

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
    """
    # Update global debug flag
    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    p = Path(control_path)
    if not p.is_file():
        typer.secho(f"Error: control file {control_path!r} does not exist", fg=ERROR_COLOR)
        raise typer.Exit(2)

    # Run the full pipeline with progress bars and logging
    # Note: ProgressFormatter.header() displays the config path
    from davinci_monet.pipeline.runner import run_analysis as pipeline_run

    try:
        result = pipeline_run(
            str(p), show_progress=True, show_plots=show_plots, preview_format=preview_format
        )
    except Exception as e:
        typer.secho(f"Pipeline error: {e}", fg=ERROR_COLOR)
        if debug:
            raise
        raise typer.Exit(1)

    # Report results
    typer.echo()
    if result.success:
        typer.secho(
            f"Analysis complete! ({result.total_duration_seconds:.1f}s)",
            fg=SUCCESS_COLOR,
        )
        typer.secho(
            f"Stages: {', '.join(result.completed_stages)}",
            fg=INFO_COLOR,
        )
    else:
        typer.secho("Analysis failed!", fg=ERROR_COLOR)
        for failed in result.failed_stages:
            typer.secho(f"  {failed.stage_name}: {failed.error}", fg=ERROR_COLOR)
        raise typer.Exit(1)
