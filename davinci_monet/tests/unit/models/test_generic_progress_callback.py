"""Tests for per-file progress callback in GenericReader.open().

Two tests:
- test_generic_progress_callback_called_per_file: verifies that when a
  progress_callback is supplied to GenericReader.open() for a 3-file list,
  the callback is called exactly 3 times with the correct (i, total, name)
  arguments in file order.
- test_generic_no_callback_uses_default_path: regression guard confirming that
  the default path (no callback) still loads the data correctly without error.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import davinci_monet.models  # noqa: F401  (registers "generic")
from davinci_monet.models.generic import GenericReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_nc(path: Path, value: float, time_offset_days: int = 0) -> None:
    """Write a tiny (time, lat, lon) NetCDF file with non-overlapping times."""
    start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=time_offset_days)
    t = pd.date_range(start, periods=2, freq="D")
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    ds = xr.Dataset(
        {"temp": (("time", "lat", "lon"), np.full((2, 2, 2), value))},
        coords={"time": t, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generic_progress_callback_called_per_file(tmp_path: Path) -> None:
    """progress_callback(i, total, name) is called once per file in order."""
    # Create 3 synthetic NetCDF files with names that sort deterministically.
    # Each file has non-overlapping time values so combine="by_coords" works.
    file_a = tmp_path / "file_a.nc"
    file_b = tmp_path / "file_b.nc"
    file_c = tmp_path / "file_c.nc"
    _make_tiny_nc(file_a, 1.0, time_offset_days=0)
    _make_tiny_nc(file_b, 2.0, time_offset_days=2)
    _make_tiny_nc(file_c, 3.0, time_offset_days=4)

    file_paths = [file_a, file_b, file_c]

    calls: List[Tuple[int, int, str]] = []

    def _cb(i: int, total: int, name: str) -> None:
        calls.append((i, total, name))

    reader = GenericReader()
    ds = reader.open(file_paths, variables=["temp"], progress_callback=_cb)

    # Dataset should still load correctly.
    assert "temp" in ds.data_vars

    # Callback must have been called exactly once per file.
    assert len(calls) == 3, f"Expected 3 callback calls, got {len(calls)}: {calls}"

    # i runs 1-based; total is always 3.
    indices = [c[0] for c in calls]
    totals = [c[1] for c in calls]
    names = [c[2] for c in calls]

    assert indices == [1, 2, 3], f"Expected i=[1,2,3], got {indices}"
    assert totals == [3, 3, 3], f"Expected total=[3,3,3], got {totals}"

    # Names must be the file basenames in the order the files were opened.
    expected_names = [p.name for p in file_paths]
    assert names == expected_names, f"Expected names {expected_names}, got {names}"


def test_generic_no_callback_uses_default_path(tmp_path: Path) -> None:
    """Without a progress_callback, open() still loads all files correctly."""
    file_a = tmp_path / "file_a.nc"
    file_b = tmp_path / "file_b.nc"
    file_c = tmp_path / "file_c.nc"
    _make_tiny_nc(file_a, 1.0, time_offset_days=0)
    _make_tiny_nc(file_b, 2.0, time_offset_days=2)
    _make_tiny_nc(file_c, 3.0, time_offset_days=4)

    reader = GenericReader()
    ds = reader.open([file_a, file_b, file_c], variables=["temp"])

    assert "temp" in ds.data_vars
    # 3 files × 2 time steps each = 6 combined time steps along time dim.
    assert ds.sizes["time"] == 6
