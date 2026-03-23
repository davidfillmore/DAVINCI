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


def run_analysis(
    control_path: str,
    debug: bool = False,
    show_plots: bool = False,
    preview_format: str = "pdf",
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
    else:
        # Pipeline footer already shows failure - just exit with error code
        raise typer.Exit(1)
