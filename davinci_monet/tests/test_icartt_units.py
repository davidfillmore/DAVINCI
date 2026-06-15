"""Tests for ICARTT header-unit parsing and application.

Covers _parse_header_units(), _apply_header_units(), and their integration
in ICARTTReader.open().  All tests are deterministic and require no monetio
or external data dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid FFI-1001 .ict content
#
# Line numbering (1-indexed per ICARTT spec):
#   L1  : NLHEAD, FFI
#   L2-L7 : org/title/source/mission/volume/date lines  (6 lines)
#   L8  : NAUXV (number of auxiliary variables — here -1 = none)
#   L9  : independent variable def  "name, units, description"
#   L10 : NV (number of dependent variables)
#   L11 : scale factors
#   L12 : missing values
#   L13 : dep var def 1  "shortname, units, longname"
#   L14 : dep var def 2
#   L15 : dep var def 3
#   L16 : NSCOML (0 special comments)
#   L17 : NNCOML (0 normal comments)
#   L18 : column-names line  (= last header line, index NLHEAD-1)
#   --- data rows ---
#   Row 1
#   Row 2
#
# NLHEAD = 18  (18 header lines)

_NLHEAD = 18
_ICT_CONTENT = f"""{_NLHEAD}, 1001
Test Organization
ICARTT Units Test
Test Source
TEST_CAMPAIGN
1, 1
2024, 01, 15, 2024, 1, 15
-1
Time_Start, seconds, start time
3
1, 1, 1
-999999, -999999, -999999
O3_CL, ppbv, Ozone_LaserAbsorption_InSitu
CO_DACOM, ppbv, CarbonMonoxide_InSitu
MSL_GPS_Altitude, m, GPS_Altitude_MSL
0
0
Time_Start, O3_CL, CO_DACOM, MSL_GPS_Altitude
0, 45.2, 120.5, 2500.0
60, 46.1, 118.3, 2510.0
"""

_EXPECTED_UNITS = {
    "Time_Start": "seconds",
    "O3_CL": "ppbv",
    "CO_DACOM": "ppbv",
    "MSL_GPS_Altitude": "m",
}


@pytest.fixture()
def ict_file(tmp_path: Path) -> Path:
    """Write a minimal valid FFI-1001 .ict file and return its path."""
    path = tmp_path / "test_data.ict"
    path.write_text(_ICT_CONTENT)
    return path


@pytest.fixture()
def reader():
    """Return an ICARTTReader instance."""
    from davinci_monet.datasets.aircraft.icartt import ICARTTReader

    return ICARTTReader()


# ---------------------------------------------------------------------------
# _parse_header_units
# ---------------------------------------------------------------------------


class TestParseHeaderUnits:
    """Unit tests for ICARTTReader._parse_header_units()."""

    def test_returns_expected_mapping(self, reader, ict_file: Path):
        result = reader._parse_header_units(ict_file)
        assert result == _EXPECTED_UNITS

    def test_independent_variable_included(self, reader, ict_file: Path):
        result = reader._parse_header_units(ict_file)
        assert result["Time_Start"] == "seconds"

    def test_dependent_variables_included(self, reader, ict_file: Path):
        result = reader._parse_header_units(ict_file)
        assert result["O3_CL"] == "ppbv"
        assert result["CO_DACOM"] == "ppbv"
        assert result["MSL_GPS_Altitude"] == "m"

    def test_malformed_header_returns_empty(self, reader, tmp_path: Path):
        """A truncated / malformed file must return {} without raising."""
        bad = tmp_path / "bad.ict"
        bad.write_text("not a valid header\n")
        result = reader._parse_header_units(bad)
        assert result == {}

    def test_missing_file_returns_empty(self, reader, tmp_path: Path):
        """A non-existent file must return {} without raising."""
        result = reader._parse_header_units(tmp_path / "nonexistent.ict")
        assert result == {}

    def test_empty_file_returns_empty(self, reader, tmp_path: Path):
        """An empty file must return {} without raising."""
        empty = tmp_path / "empty.ict"
        empty.write_text("")
        result = reader._parse_header_units(empty)
        assert result == {}

    def test_missing_value_truncated_header_returns_empty(self, reader, tmp_path: Path):
        """A file whose dep-var lines are missing (NV>0 but lines absent) returns {}."""
        # Only 5 lines; int(lines[9]) will fail (IndexError on lines[9])
        truncated = tmp_path / "trunc.ict"
        truncated.write_text("20, 1001\nOrg\nTitle\nSrc\nMission\n")
        result = reader._parse_header_units(truncated)
        assert result == {}


# ---------------------------------------------------------------------------
# _apply_header_units
# ---------------------------------------------------------------------------


class TestApplyHeaderUnits:
    """Unit tests for ICARTTReader._apply_header_units()."""

    def _make_ds(self) -> xr.Dataset:
        """Tiny dataset with O3_CL and CO_DACOM, no units."""
        time = np.array([0.0, 60.0])
        return xr.Dataset(
            {
                "O3_CL": ("time", np.array([45.2, 46.1])),
                "CO_DACOM": ("time", np.array([120.5, 118.3])),
            },
            coords={"time": time},
        )

    def test_units_set_on_variables(self, reader, ict_file: Path):
        ds = self._make_ds()
        result = reader._apply_header_units(ds, [ict_file])
        assert result["O3_CL"].attrs["units"] == "ppbv"
        assert result["CO_DACOM"].attrs["units"] == "ppbv"

    def test_existing_units_not_overwritten(self, reader, ict_file: Path):
        ds = self._make_ds()
        ds["O3_CL"].attrs["units"] = "ppmv"  # pre-existing attr
        result = reader._apply_header_units(ds, [ict_file])
        # Must not be overwritten
        assert result["O3_CL"].attrs["units"] == "ppmv"
        # The other variable should still get stamped
        assert result["CO_DACOM"].attrs["units"] == "ppbv"

    def test_variable_not_in_map_left_alone(self, reader, ict_file: Path):
        ds = self._make_ds()
        ds["extra_var"] = xr.DataArray([1.0, 2.0], dims=["time"])
        result = reader._apply_header_units(ds, [ict_file])
        assert "units" not in result["extra_var"].attrs

    def test_coords_get_units_stamped(self, reader, ict_file: Path):
        """Coordinates (not just data vars) should receive units."""
        ds = xr.Dataset(
            {"O3_CL": ("time", np.array([45.2, 46.1]))},
            coords={"time": np.array([0.0, 60.0]), "Time_Start": ("time", np.array([0.0, 60.0]))},
        )
        result = reader._apply_header_units(ds, [ict_file])
        # Time_Start is in the header map
        assert result["Time_Start"].attrs.get("units") == "seconds"

    def test_empty_file_list_returns_ds_unchanged(self, reader):
        ds = xr.Dataset({"O3_CL": ("time", np.array([45.2, 46.1]))})
        result = reader._apply_header_units(ds, [])
        assert result is ds
        assert "units" not in result["O3_CL"].attrs

    def test_malformed_header_returns_ds_unchanged(self, reader, tmp_path: Path):
        bad = tmp_path / "bad.ict"
        bad.write_text("not a valid header\n")
        ds = xr.Dataset({"O3_CL": ("time", np.array([45.2, 46.1]))})
        result = reader._apply_header_units(ds, [bad])
        assert "units" not in result["O3_CL"].attrs
