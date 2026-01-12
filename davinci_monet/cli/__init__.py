"""Command-line interface for DAVINCI-MONET.

This module provides the CLI using Typer for running analyses,
downloading data, and validating configurations.

Quick Start
-----------
>>> # From command line:
>>> davinci-monet run config.yaml
>>> davinci-monet validate config.yaml
>>> davinci-monet get aeronet -s 2024-01-01 -e 2024-01-31

Programmatic Usage
------------------
>>> from davinci_monet.cli import app, cli
>>> # Run the CLI programmatically
>>> cli(["run", "config.yaml"])
"""

from davinci_monet.cli.app import (
    DEBUG,
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    WARNING_COLOR,
    app,
    cli,
    timer,
)

__all__ = [
    "app",
    "cli",
    "timer",
    "INFO_COLOR",
    "ERROR_COLOR",
    "SUCCESS_COLOR",
    "WARNING_COLOR",
    "DEBUG",
]
