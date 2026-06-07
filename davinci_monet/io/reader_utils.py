"""Shared reader plumbing for source readers.

This module collects the near-identical boilerplate that every model (and,
later, observation) reader repeats: file-list resolution and validation, the
3-attempt transient-NetCDF retry loop, monetio-then-xarray variable selection,
and dimension/coordinate standardization.

The helpers are intentionally small and pure so each reader keeps its
format-specific logic (e.g. WRF-Chem ``Times`` decoding, CESM hybrid levels,
generic alias-based standardization) while sharing the mechanical plumbing.

Behavior note
-------------
These helpers reproduce the existing per-reader behavior exactly. In
particular:

* :func:`validate_file_list` does *not* glob — model readers receive paths
  already expanded by the pipeline load stage. Use :func:`resolve_file_list`
  for the canonical glob+expanduser resolution where globbing is wanted.
* :func:`retry_transient_open` uses the same transient-error detection
  (:func:`~davinci_monet.core.exceptions.is_transient_error`), the same
  ``cleanup_netcdf_state`` recovery, the same warning text, and the same
  ``write_error_log`` + :class:`DataFormatError` final-failure path the
  readers used inline.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

import xarray as xr

from davinci_monet.core.exceptions import (
    DataFormatError,
    DataNotFoundError,
    cleanup_netcdf_state,
    is_transient_error,
    write_error_log,
)

__all__ = [
    "resolve_file_list",
    "validate_file_list",
    "select_available_vars",
    "select_variables",
    "retry_transient_open",
    "standardize_dims",
    "promote_to_coords",
    "alias_coord",
]


def resolve_file_list(files: Any) -> list[str]:
    """Resolve a ``files``/``filename`` config value to a sorted path list.

    This is the canonical glob resolver: each entry is expanded with
    ``~`` user-directory expansion, then glob patterns (``*``/``?``) are
    expanded and sorted while plain paths are passed through verbatim.

    Parameters
    ----------
    files
        A path string, a sequence of path strings, or ``None``.

    Returns
    -------
    list[str]
        Resolved file paths (sorted within each glob expansion). Empty when
        ``files`` is ``None``.
    """
    from glob import glob

    if files is None:
        return []
    if isinstance(files, (list, tuple)):
        values = [str(Path(str(item)).expanduser()) for item in files]
    else:
        values = [str(Path(str(files)).expanduser())]

    expanded: list[str] = []
    for value in values:
        if "*" in value or "?" in value:
            expanded.extend(sorted(glob(value)))
        else:
            expanded.append(value)
    return expanded


def validate_file_list(
    file_paths: Sequence[str | Path],
    *,
    source_label: str,
) -> list[Path]:
    """Convert to ``Path`` objects and verify the files exist.

    Reproduces the preamble each model reader performs at the top of
    ``open()``: convert inputs to :class:`~pathlib.Path`, raise when the list
    is empty, and raise when any path is missing.

    Parameters
    ----------
    file_paths
        Paths provided to the reader's ``open()``.
    source_label
        Human-readable source name used in error messages (e.g. ``"CMAQ"``,
        ``"WRF-Chem"``, ``"CESM"``, ``"UFS"``). For the generic reader an
        empty string yields the historical wording (``"No files provided"`` /
        ``"Files not found: ..."``).

    Returns
    -------
    list[pathlib.Path]
        The validated paths.

    Raises
    ------
    DataNotFoundError
        If no files are provided, or if any file does not exist.
    """
    file_list = [Path(f) for f in file_paths]

    # "No CMAQ files provided" with a label, "No files provided" without.
    if not file_list:
        noun = f"{source_label} files" if source_label else "files"
        raise DataNotFoundError(f"No {noun} provided")

    missing = [f for f in file_list if not f.exists()]
    if missing:
        # "CMAQ files not found: ..." with a label, "Files not found: ..."
        # (leading capital) without — matching the historical wording.
        label = f"{source_label} files" if source_label else "Files"
        raise DataNotFoundError(f"{label} not found: {missing}")

    return file_list


def select_available_vars(
    ds: xr.Dataset,
    variables: Sequence[str] | None,
) -> list[str]:
    """Return the subset of ``variables`` present in ``ds.data_vars``.

    Parameters
    ----------
    ds
        Dataset to inspect.
    variables
        Requested variable names, or ``None``.

    Returns
    -------
    list[str]
        Requested names that exist as data variables. Empty when
        ``variables`` is ``None`` or none are present.
    """
    if variables is None:
        return []
    return [v for v in variables if v in ds.data_vars]


def select_variables(
    ds: xr.Dataset,
    variables: Sequence[str] | None,
) -> xr.Dataset:
    """Subset ``ds`` to the requested variables when any are present.

    Mirrors the common idiom::

        available = [v for v in variables if v in ds.data_vars]
        if available:
            ds = ds[available]

    When ``variables`` is ``None`` or none of the requested names are present,
    the dataset is returned unchanged.
    """
    available = select_available_vars(ds, variables)
    if available:
        return ds[available]
    return ds


def retry_transient_open(
    open_fn: Callable[[], xr.Dataset],
    *,
    context: str,
    attempts: int = 3,
    reraise: tuple[type[BaseException], ...] = (),
    on_failure: Callable[[], None] | None = None,
) -> xr.Dataset:
    """Run ``open_fn`` with retries on transient NetCDF/HDF5 errors.

    Encapsulates the 3-attempt loop the readers wrap around their open call:
    on a transient error (per :func:`is_transient_error`) it warns, runs
    :func:`cleanup_netcdf_state`, and retries; otherwise (or once attempts are
    exhausted) it logs via :func:`write_error_log` and raises
    :class:`DataFormatError`.

    Parameters
    ----------
    open_fn
        Zero-argument callable that opens and returns the dataset. It should
        also perform any per-attempt variable selection so a retry re-runs it.
    context
        Operation description used for warnings, the error log context, and the
        ``DataFormatError`` message — e.g. ``"Opening CMAQ files"``. The
        user-facing message becomes ``"Failed to {context_lower}: {e}"`` where
        ``context_lower`` lowercases the leading ``"Opening"`` to match the
        historical ``"Failed to open ... files"`` wording.
    attempts
        Maximum number of attempts (default 3).
    reraise
        Exception types that should propagate immediately without retry or
        wrapping (e.g. CESM-SE re-raises :class:`DataFormatError` from a bad
        SCRIP file). These bypass the transient-retry and final-wrap logic.
    on_failure
        Optional zero-argument cleanup hook invoked right before raising on the
        terminal failure path (used by the generic reader to suppress noisy
        ``__del__`` tracebacks via stderr redirection).

    Returns
    -------
    xr.Dataset
        The dataset returned by ``open_fn``.

    Raises
    ------
    DataFormatError
        On a non-transient error or once attempts are exhausted.
    """
    # "Opening CMAQ files" -> "open CMAQ files" for the failure message, which
    # historically read "Failed to open <SOURCE> files: ...".
    if context.startswith("Opening "):
        action = "open " + context[len("Opening ") :]
    else:
        action = context

    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            return open_fn()
        except reraise:
            # Caller-designated terminal exceptions propagate verbatim.
            raise
        except Exception as e:
            last_error = e
            if attempt < attempts - 1 and is_transient_error(e):
                warnings.warn(
                    f"Transient NetCDF error (attempt {attempt + 1}/{attempts}), " f"retrying: {e}",
                    UserWarning,
                )
                cleanup_netcdf_state()
                continue
            if on_failure is not None:
                on_failure()
            error_file = write_error_log(e, context)
            msg = f"Failed to {action}: {e}"
            if error_file:
                msg += f" (details: {error_file})"
            raise DataFormatError(msg) from e

    # Defensive: loop always returns or raises above. Mirror the readers'
    # post-loop "after N attempts" message.
    if on_failure is not None:
        on_failure()
    raise DataFormatError(f"Failed to {action} after {attempts} attempts") from last_error


def standardize_dims(ds: xr.Dataset, rename_map: dict[str, str]) -> xr.Dataset:
    """Rename dimensions present in ``ds`` according to ``rename_map``.

    Only the entries whose source dimension is actually present in ``ds.dims``
    are applied, matching the per-reader pattern of building a ``dim_renames``
    dict guarded by ``if "<dim>" in ds.dims`` before calling ``ds.rename``.

    Parameters
    ----------
    ds
        Dataset to standardize.
    rename_map
        Mapping of source dimension name to standardized name.

    Returns
    -------
    xr.Dataset
        Dataset with present dimensions renamed. Returned unchanged when no
        mapped dimension is present.
    """
    present = {src: dst for src, dst in rename_map.items() if src in ds.dims}
    if present:
        return ds.rename(present)
    return ds


def promote_to_coords(ds: xr.Dataset, names: Sequence[str]) -> xr.Dataset:
    """Promote named data variables to coordinates when present.

    Mirrors the repeated idiom::

        if "latitude" in ds.data_vars and "latitude" not in ds.coords:
            ds = ds.set_coords("latitude")

    Parameters
    ----------
    ds
        Dataset to modify.
    names
        Variable names to promote (e.g. ``("latitude", "longitude")``).

    Returns
    -------
    xr.Dataset
        Dataset with the named variables set as coordinates where applicable.
    """
    to_set = [n for n in names if n in ds.data_vars and n not in ds.coords]
    if to_set:
        return ds.set_coords(to_set)
    return ds


def alias_coord(ds: xr.Dataset, source: str, alias: str) -> xr.Dataset:
    """Add ``alias`` as a coordinate pointing at existing coordinate ``source``.

    Mirrors the repeated idiom::

        if "latitude" in ds.coords and "lat" not in ds.coords:
            ds = ds.assign_coords(lat=ds["latitude"])

    Parameters
    ----------
    ds
        Dataset to modify.
    source
        Name of an existing coordinate.
    alias
        New coordinate name to create as an alias.

    Returns
    -------
    xr.Dataset
        Dataset with the alias coordinate added when ``source`` exists and
        ``alias`` does not.
    """
    if source in ds.coords and alias not in ds.coords:
        return ds.assign_coords({alias: ds[source]})
    return ds
