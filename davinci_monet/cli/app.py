"""Main CLI application for DAVINCI-MONET.

This module provides the command-line interface using Typer.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

try:
    import typer
except ImportError as e:
    print(
        "The DAVINCI-MONET CLI requires the module 'typer'. "
        "You can install it with `conda install -c conda-forge typer` or "
        "`pip install typer`. "
        f"The error message was: {e}"
    )
    raise SystemExit(1)


# Global debug flag
DEBUG = False

# Color scheme
INFO_COLOR = typer.colors.CYAN
ERROR_COLOR = typer.colors.BRIGHT_RED
SUCCESS_COLOR = typer.colors.GREEN
WARNING_COLOR = typer.colors.YELLOW



def _get_full_name(obj: object) -> str:
    """Get the full name of a function or type, including the module name."""
    import builtins
    import inspect

    mod = inspect.getmodule(obj)
    name = getattr(obj, "__qualname__", str(obj))
    if mod is None or mod is builtins:
        return name
    else:
        return f"{mod.__name__}.{name}"


@contextmanager
def timer(desc: str = "") -> Generator[None, None, None]:
    """Context manager for timing operations with colored output.

    Parameters
    ----------
    desc
        Description of the operation being timed.

    Yields
    ------
    None
    """
    start = time.perf_counter()
    tpl = f"{desc} {{status}} in {{elapsed:.3g}} seconds"

    typer.secho(f"{desc} ...", fg=INFO_COLOR)
    try:
        yield
    except Exception as e:
        typer.secho(
            tpl.format(status="failed", elapsed=time.perf_counter() - start),
            fg=ERROR_COLOR,
        )
        typer.secho(
            f"Error message (type: {_get_full_name(type(e))}): {e}", fg=ERROR_COLOR
        )
        if DEBUG:
            raise
        else:
            typer.echo("(Use the '--debug' flag to see more info.)")
            raise typer.Exit(1)
    else:
        typer.secho(
            tpl.format(status="succeeded", elapsed=time.perf_counter() - start),
            fg=SUCCESS_COLOR,
        )


def _version_callback(value: bool) -> None:
    """Display version information."""
    from davinci_monet import __version__

    if value:
        typer.echo(f"davinci-monet {__version__}")
        raise typer.Exit()


# Create the main application
app = typer.Typer(
    name="davinci-monet",
    help="DAVINCI-MONET: Model and Observation Evaluation Toolkit",
    add_completion=False,
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """DAVINCI-MONET: Model and Observation Evaluation Toolkit.

    A modern tool for evaluating atmospheric chemistry and air quality
    models against observations.
    """


# Import and register commands
def register_commands() -> None:
    """Register all CLI commands."""
    # Import command modules
    from davinci_monet.cli.commands import get_data, run, validate

    # Register subcommands
    app.add_typer(get_data.app, name="get")


# Register commands when module loads
register_commands()


# Direct commands are added via decorators
@app.command()
def run(
    control: str = typer.Argument(
        ...,
        help="Path to the control file (YAML configuration).",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Print more messages (including full tracebacks).",
    ),
) -> None:
    """Run DAVINCI-MONET analysis as described in the control file."""
    from davinci_monet.cli.commands.run import run_analysis

    global DEBUG
    DEBUG = debug

    run_analysis(control, debug=debug)


@app.command()
def validate(
    control: str = typer.Argument(
        ...,
        help="Path to the control file (YAML configuration) to validate.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Use strict validation (no extra fields allowed).",
    ),
    show_config: bool = typer.Option(
        False,
        "--show",
        help="Print the parsed configuration.",
    ),
) -> None:
    """Validate a DAVINCI-MONET configuration file."""
    from davinci_monet.cli.commands.validate import validate_config_command

    validate_config_command(control, strict=strict, show_config=show_config)


# CLI entry point
cli = app

# For sphinx-click documentation
_typer_click_object = typer.main.get_command(app)


if __name__ == "__main__":
    cli()
