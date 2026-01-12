"""Data download commands for DAVINCI-MONET CLI.

This module implements commands for downloading observation data
from various sources (AERONET, AirNow, AQS, OpenAQ, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import typer

from davinci_monet.cli.app import (
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    timer,
)

# Create sub-application for data commands
app = typer.Typer(
    name="get",
    help="Download observation data from various sources.",
)

_DATE_FMT_NOTE = (
    "Date can be in any format accepted by pandas.Timestamp(), "
    "e.g., 'YYYY-MM-DD', or 'M/D/YYYY'. "
    "Time can be specified by appending ' HH[:MM[:SS]]'."
)
_DATE_END_NOTE = (
    "To get the full last day for hourly data, specify the ending hour "
    "(e.g., append ' 23') or increase end date by one day."
)


def _parse_output_path(
    out_name: str | None, dst: Path, default_prefix: str, start_date: str, end_date: str
) -> tuple[Path, str]:
    """Parse output path and file name.

    Parameters
    ----------
    out_name
        User-specified output name (may include path).
    dst
        Destination directory.
    default_prefix
        Default file name prefix.
    start_date
        Start date string.
    end_date
        End date string.

    Returns
    -------
    tuple
        (destination directory, output filename)
    """
    import pandas as pd

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    fmt = r"%Y%m%d"

    if out_name is None:
        out_name = f"{default_prefix}_{start:{fmt}}_{end:{fmt}}.nc"
        return dst, out_name

    p = Path(out_name)
    if p.name == out_name:
        # Just the file name, no path
        return dst, out_name

    # out_name includes a path
    if dst != Path("."):
        typer.echo(
            f"warning: overriding `dst` setting {dst.as_posix()!r} "
            f"with `out_name` {p.as_posix()!r}"
        )
    return p.parent, p.name


@app.command("aeronet")
def get_aeronet(
    start_date: str = typer.Option(
        ..., "-s", "--start-date", help=f"Start date. {_DATE_FMT_NOTE}"
    ),
    end_date: str = typer.Option(
        ..., "-e", "--end-date", help=f"End date. {_DATE_FMT_NOTE} {_DATE_END_NOTE}"
    ),
    daily: bool = typer.Option(
        False, help="Whether to retrieve the daily averaged data product."
    ),
    freq: str = typer.Option(
        "h",
        "-f",
        "--freq",
        help="Frequency to resample to. Mean is used to reduce time groups.",
    ),
    out_name: str = typer.Option(
        None,
        "-o",
        help=(
            "Output file name (or full/relative path). "
            "By default: 'AERONET_<start-date>_<end-date>.nc'."
        ),
    ),
    dst: Path = typer.Option(
        ".",
        "-d",
        "--dst",
        help="Destination directory for output file.",
    ),
    compress: bool = typer.Option(
        True,
        help="Apply compression to output file.",
    ),
    num_workers: int = typer.Option(
        1, "-n", "--num-workers", help="Number of download workers."
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
    debug: bool = typer.Option(
        False, "--debug", help="Print full tracebacks on error."
    ),
) -> None:
    """Download AERONET data and reformat for DAVINCI-MONET usage."""
    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    
    dst, out_name = _parse_output_path(out_name, dst, "AERONET_L15", start_date, end_date)

    with timer("Fetching AERONET data with monetio"):
        import monetio as mio
        import pandas as pd

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.date_range(start, end, freq="D")

        df = mio.aeronet.add_data(
            dates,
            daily=daily,
            freq=freq,
            n_procs=num_workers,
            verbose=1 if verbose else 0,
        )

    with timer("Forming xarray Dataset"):
        from davinci_monet.observations.surface.aeronet import _dataframe_to_xarray

        ds = _dataframe_to_xarray(df)

    with timer("Writing netCDF file"):
        output_path = dst / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            from davinci_monet.io.writers import write_dataset

            write_dataset(ds, output_path, compress=True)
        else:
            ds.to_netcdf(output_path)

    typer.secho(f"\nOutput written to: {output_path}", fg=SUCCESS_COLOR)


@app.command("airnow")
def get_airnow(
    start_date: str = typer.Option(
        ..., "-s", "--start-date", help=f"Start date. {_DATE_FMT_NOTE}"
    ),
    end_date: str = typer.Option(
        ..., "-e", "--end-date", help=f"End date. {_DATE_FMT_NOTE} {_DATE_END_NOTE}"
    ),
    daily: bool = typer.Option(
        False, help="Whether to retrieve the daily averaged data product."
    ),
    out_name: str = typer.Option(
        None,
        "-o",
        help=(
            "Output file name (or full/relative path). "
            "By default: 'AirNow_<start-date>_<end-date>.nc'."
        ),
    ),
    dst: Path = typer.Option(
        ".",
        "-d",
        "--dst",
        help="Destination directory for output file.",
    ),
    compress: bool = typer.Option(
        True,
        help="Apply compression to output file.",
    ),
    num_workers: int = typer.Option(
        1, "-n", "--num-workers", help="Number of download workers."
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
    debug: bool = typer.Option(
        False, "--debug", help="Print full tracebacks on error."
    ),
) -> None:
    """Download AirNow data and reformat for DAVINCI-MONET usage."""
    import warnings

    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    
    dst, out_name = _parse_output_path(out_name, dst, "AirNow", start_date, end_date)

    with timer("Fetching AirNow data with monetio"):
        import monetio as mio
        import pandas as pd

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.date_range(start, end, freq="h" if not daily else "D")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The (error|warn)_bad_lines argument has been deprecated"
            )
            df = mio.airnow.add_data(
                dates,
                download=False,
                wide_fmt=True,
                n_procs=num_workers,
                daily=daily,
            )

    with timer("Forming xarray Dataset"):
        from davinci_monet.observations.surface.airnow import _dataframe_to_xarray

        ds = _dataframe_to_xarray(df, daily=daily)

    with timer("Writing netCDF file"):
        output_path = dst / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            from davinci_monet.io.writers import write_dataset

            write_dataset(ds, output_path, compress=True)
        else:
            ds.to_netcdf(output_path)

    typer.secho(f"\nOutput written to: {output_path}", fg=SUCCESS_COLOR)


@app.command("aqs")
def get_aqs(
    start_date: str = typer.Option(
        ..., "-s", "--start-date", help=f"Start date. {_DATE_FMT_NOTE}"
    ),
    end_date: str = typer.Option(
        ..., "-e", "--end-date", help=f"End date. {_DATE_FMT_NOTE} {_DATE_END_NOTE}"
    ),
    daily: bool = typer.Option(
        False, help="Whether to retrieve the daily averaged data product."
    ),
    param: List[str] = typer.Option(
        ["O3", "PM2.5", "PM10"],
        "-p",
        "--params",
        help=(
            "Parameter groups. Use '-p' multiple times for multiple groups. "
            "Examples: 'SPEC', 'VOC', 'NONOxNOy', 'SO2', 'NO2', 'CO'."
        ),
    ),
    out_name: str = typer.Option(
        None,
        "-o",
        help=(
            "Output file name (or full/relative path). "
            "By default: 'AQS_<start-date>_<end-date>.nc'."
        ),
    ),
    dst: Path = typer.Option(
        ".",
        "-d",
        "--dst",
        help="Destination directory for output file.",
    ),
    compress: bool = typer.Option(
        True,
        help="Apply compression to output file.",
    ),
    num_workers: int = typer.Option(
        1, "-n", "--num-workers", help="Number of download workers."
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
    debug: bool = typer.Option(
        False, "--debug", help="Print full tracebacks on error."
    ),
) -> None:
    """Download EPA AQS data and reformat for DAVINCI-MONET usage.

    These are archived data from https://aqs.epa.gov/aqsweb/airdata/download_files.html
    Recent-past data may not be available from this source.
    """
    import warnings

    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    
    dst, out_name = _parse_output_path(out_name, dst, "AQS", start_date, end_date)

    with timer("Fetching AQS data with monetio"):
        import monetio as mio
        import pandas as pd

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.date_range(start, end, freq="h" if not daily else "D")

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The (error|warn)_bad_lines argument has been deprecated"
            )
            df = mio.aqs.add_data(
                dates,
                param=param,
                daily=daily,
                network=None,
                download=False,
                local=False,
                wide_fmt=True,
                n_procs=num_workers,
                meta=False,
            )

    with timer("Forming xarray Dataset"):
        from davinci_monet.observations.surface.aqs import _dataframe_to_xarray

        ds = _dataframe_to_xarray(df, daily=daily)

    with timer("Writing netCDF file"):
        output_path = dst / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            from davinci_monet.io.writers import write_dataset

            write_dataset(ds, output_path, compress=True)
        else:
            ds.to_netcdf(output_path)

    typer.secho(f"\nOutput written to: {output_path}", fg=SUCCESS_COLOR)


@app.command("openaq")
def get_openaq(
    start_date: str = typer.Option(
        ..., "-s", "--start-date", help=f"Start date. {_DATE_FMT_NOTE}"
    ),
    end_date: str = typer.Option(
        ..., "-e", "--end-date", help=f"End date. {_DATE_FMT_NOTE} {_DATE_END_NOTE}"
    ),
    param: List[str] = typer.Option(
        ["o3", "pm25", "pm10"],
        "-p",
        "--param",
        help=(
            "Parameters to retrieve. Use '-p' multiple times for multiple params. "
            "Examples: 'no', 'no2', 'nox', 'so2', 'co', 'bc'."
        ),
    ),
    reference_grade: bool = typer.Option(
        True, help="Include reference-grade sensors."
    ),
    low_cost: bool = typer.Option(False, help="Include low-cost sensors."),
    country: List[str] = typer.Option(
        None,
        "-c",
        "--country",
        help="Two-letter country codes (US, CA, MX, ...). Use multiple times.",
    ),
    out_name: str = typer.Option(
        None,
        "-o",
        help=(
            "Output file name (or full/relative path). "
            "By default: 'OpenAQ_<start-date>_<end-date>.nc'."
        ),
    ),
    dst: Path = typer.Option(
        ".",
        "-d",
        "--dst",
        help="Destination directory for output file.",
    ),
    compress: bool = typer.Option(
        True,
        help="Apply compression to output file.",
    ),
    num_workers: int = typer.Option(
        1, "-n", "--num-workers", help="Number of download workers."
    ),
    verbose: bool = typer.Option(False, help="Enable verbose output."),
    debug: bool = typer.Option(
        False, "--debug", help="Print full tracebacks on error."
    ),
) -> None:
    """Download hourly OpenAQ data and reformat for DAVINCI-MONET usage."""
    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    
    dst, out_name = _parse_output_path(out_name, dst, "OpenAQ", start_date, end_date)

    # Validate sensor type selection
    sensor_types = []
    if reference_grade:
        sensor_types.append("reference grade")
    if low_cost:
        sensor_types.append("low-cost sensor")
    if not sensor_types:
        typer.secho(
            "Error: no sensor types selected. "
            "Use --reference-grade and/or --low-cost",
            fg=ERROR_COLOR,
        )
        raise typer.Exit(2)

    if not country:
        country = None

    with timer("Fetching OpenAQ data with monetio"):
        import monetio as mio
        import pandas as pd

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.date_range(start, end, freq="h")

        df = mio.obs.openaq_v3.add_data(
            dates,
            parameters=param,
            sensor_type=sensor_types,
            wide_fmt=True,
            hourly=True,
            country=country,
            timeout=60,
            retry=15,
            threads=num_workers if num_workers > 1 else None,
        )

        if df.empty:
            raise RuntimeError("No data found")

        # Handle duplicates
        dupes = df[df.duplicated(["time", "siteid"], keep=False)]
        if not dupes.empty:
            typer.echo(f"warning: {len(dupes)} duplicate time-siteid rows, keeping first")
            df = df.drop_duplicates(["time", "siteid"])

    with timer("Forming xarray Dataset"):
        from davinci_monet.observations.surface.openaq import _dataframe_to_xarray

        ds = _dataframe_to_xarray(df)

    with timer("Writing netCDF file"):
        output_path = dst / out_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if compress:
            from davinci_monet.io.writers import write_dataset

            write_dataset(ds, output_path, compress=True)
        else:
            ds.to_netcdf(output_path)

    typer.secho(f"\nOutput written to: {output_path}", fg=SUCCESS_COLOR)
