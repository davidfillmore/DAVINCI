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
    WARNING_COLOR,
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
        from davinci_monet.config import load_config, load_yaml
        from davinci_monet.config.migration import check_deprecated_fields, detect_config_version

        # First, load raw YAML
        raw_config = load_yaml(p)

        # Check for deprecated fields
        deprecations = check_deprecated_fields(raw_config)
        if deprecations:
            typer.secho("Deprecation warnings:", fg=WARNING_COLOR)
            for warning in deprecations:
                typer.echo(f"  - {warning}")
            typer.echo()

        # Detect version
        version = detect_config_version(raw_config)
        typer.secho(f"Detected config version: {version}", fg=INFO_COLOR)

        # Parse and validate
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
                role = f", role={source_cfg.role}" if source_cfg.role else ""
                typer.echo(f"    - {name}: {source_cfg.type}{role}")

        # Unified pairs
        if config.pairs:
            typer.echo(f"  Pairs: {len(config.pairs)} defined")
            for name, pair_cfg in config.pairs.items():
                if pair_cfg.sources:
                    reference = f", reference={pair_cfg.reference}" if pair_cfg.reference else ""
                    typer.echo(f"    - {name}: {', '.join(pair_cfg.sources)}{reference}")
                elif pair_cfg.model and pair_cfg.obs:
                    typer.echo(f"    - {name}: {pair_cfg.model}, {pair_cfg.obs}")

        # Models
        if not config.sources and config.model:
            typer.echo(f"  Models: {len(config.model)} defined")
            for name, model_cfg in config.model.items():
                typer.echo(f"    - {name}: {model_cfg.mod_type}")

        # Observations
        if not config.sources and config.obs:
            typer.echo(f"  Observations: {len(config.obs)} defined")
            for name, obs_cfg in config.obs.items():
                typer.echo(f"    - {name}: {obs_cfg.obs_type}")

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
            config_dict = config.model_dump(exclude_none=True)
            typer.echo(json.dumps(config_dict, indent=2, default=str))

    except ConfigurationError as e:
        # Styled display for configuration/YAML errors
        display_error("Validation Error", str(e), config_path=control_path)
        raise typer.Exit(1)
    except Exception as e:
        # Styled display for unexpected errors
        display_error("Error", str(e), config_path=control_path)
        raise typer.Exit(1)
