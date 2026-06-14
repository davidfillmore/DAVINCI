"""Validate command for DAVINCI CLI.

This module implements the configuration validation command.
"""

from __future__ import annotations

from pathlib import Path

import typer

from davinci_monet.cli.app import (
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    display_error,
)
from davinci_monet.core.exceptions import ConfigurationError


def validate_config_command(
    control_path: str, strict: bool = False, show_config: bool = False
) -> None:
    """Validate a DAVINCI configuration file.

    Parameters
    ----------
    control_path
        Path to the YAML control file.
    strict
        If True, use strict validation (no extra fields allowed).
    show_config
        If True, print the parsed configuration.
    """
    p = Path(control_path)
    if not p.is_file():
        typer.secho(f"Error: control file {control_path!r} does not exist", fg=ERROR_COLOR)
        raise typer.Exit(2)

    typer.secho(f"Validating: {control_path!r}", fg=INFO_COLOR)
    typer.secho(f"Full path: {p.absolute().as_posix()}", fg=INFO_COLOR)
    typer.secho(f"Mode: {'strict' if strict else 'flexible'}", fg=INFO_COLOR)
    typer.echo()

    try:
        from davinci_monet.config import load_config

        # Parse and validate.
        config = load_config(p)

        # Report what was found
        typer.echo()
        typer.secho("Configuration summary:", fg=INFO_COLOR)

        # Analysis section
        if config.analysis:
            typer.echo(f"  Analysis:")
            typer.echo(f"    Start: {config.analysis.start_time}")
            typer.echo(f"    End: {config.analysis.end_time}")
            if config.analysis.output_dir:
                typer.echo(f"    Output dir: {config.analysis.output_dir}")

        # Unified sources
        if config.sources:
            typer.echo(f"  Sources: {len(config.sources)} defined")
            for name, source_cfg in config.sources.items():
                typer.echo(f"    - {name}: {source_cfg.type}")

        # Unified pairs
        if config.pairs:
            typer.echo(f"  Pairs: {len(config.pairs)} defined")
            for name, pair_cfg in config.pairs.items():
                axes = (
                    f"x={pair_cfg.x.source}:{pair_cfg.x.variable}, "
                    f"y={pair_cfg.y.source}:{pair_cfg.y.variable}"
                )
                typer.echo(f"    - {name}: {axes}")

        # Plots
        if config.plots:
            typer.echo(f"  Plots: {len(config.plots)} defined")
            for name, plot_cfg in config.plots.items():
                typer.echo(f"    - {name}: {plot_cfg.type}")

        # Stats
        if config.stats:
            typer.echo(f"  Statistics: configured")

        typer.echo()
        typer.secho("Validation passed!", fg=SUCCESS_COLOR)

        # Show full config if requested
        if show_config:
            typer.echo()
            typer.secho("Parsed configuration:", fg=INFO_COLOR)
            typer.echo("-" * 40)

            import json

            # Convert to dict and display
            from davinci_monet.core.schema_utils import dump_schema

            config_dict = dump_schema(config, exclude_none=True)
            typer.echo(json.dumps(config_dict, indent=2, default=str))

    except ConfigurationError as e:
        # Styled display for configuration/YAML errors
        display_error("Validation Error", str(e), config_path=control_path)
        raise typer.Exit(1)
    except Exception as e:
        # Styled display for unexpected errors
        display_error("Error", str(e), config_path=control_path)
        raise typer.Exit(1)
