"""Tests for type aliases module.

These tests verify that type aliases are properly defined and usable.
Since TypeAlias is primarily for static type checking, these tests
focus on ensuring the module imports correctly and aliases are defined.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import get_args, get_origin

import numpy as np
import pytest


class TestTypeAliasesImport:
    """Tests for importing type aliases."""

    def test_import_path_types(self) -> None:
        """Test importing path-related type aliases."""
        from davinci_monet.core.types import PathLike, PathSequence

        # Verify they're defined (not None)
        assert PathLike is not None
        assert PathSequence is not None

    def test_import_time_types(self) -> None:
        """Test importing time-related type aliases."""
        from davinci_monet.core.types import TimeDelta, TimeRange, Timestamp

        assert Timestamp is not None
        assert TimeDelta is not None
        assert TimeRange is not None

    def test_import_numeric_types(self) -> None:
        """Test importing numeric type aliases."""
        from davinci_monet.core.types import ArrayLike, BoolArray, FloatArray, IntArray, Number

        assert Number is not None
        assert ArrayLike is not None
        assert FloatArray is not None
        assert IntArray is not None
        assert BoolArray is not None

    def test_import_coordinate_types(self) -> None:
        """Test importing coordinate type aliases."""
        from davinci_monet.core.types import (
            Altitude,
            BoundingBox,
            Coordinate,
            Coordinate3D,
            Latitude,
            Longitude,
        )

        assert Longitude is not None
        assert Latitude is not None
        assert Coordinate is not None
        assert BoundingBox is not None
        assert Altitude is not None
        assert Coordinate3D is not None

    def test_import_variable_types(self) -> None:
        """Test importing variable-related type aliases."""
        from davinci_monet.core.types import (
            AttributeDict,
            DimensionName,
            Dimensions,
            VariableDict,
            VariableMapping,
            VariableName,
        )

        assert VariableName is not None
        assert VariableMapping is not None
        assert VariableDict is not None
        assert DimensionName is not None
        assert Dimensions is not None
        assert AttributeDict is not None

    def test_import_config_types(self) -> None:
        """Test importing configuration type aliases."""
        from davinci_monet.core.types import ConfigDict, NestedConfigDict

        assert ConfigDict is not None
        assert NestedConfigDict is not None

    def test_import_callable_types(self) -> None:
        """Test importing callable type aliases."""
        from davinci_monet.core.types import DataTransformer, Predicate, Transformer

        assert Transformer is not None
        assert Predicate is not None
        assert DataTransformer is not None

    def test_import_statistics_types(self) -> None:
        """Test importing statistics type aliases."""
        from davinci_monet.core.types import GroupBy, MetricDict, MetricName, MetricValue

        assert MetricName is not None
        assert MetricValue is not None
        assert MetricDict is not None
        assert GroupBy is not None

    def test_import_plotting_types(self) -> None:
        """Test importing plotting type aliases."""
        from davinci_monet.core.types import Color, ColorMap, PlotKwargs

        assert Color is not None
        assert ColorMap is not None
        assert PlotKwargs is not None

    def test_import_io_types(self) -> None:
        """Test importing I/O type aliases."""
        from davinci_monet.core.types import CompressionLevel, Encoding, FileFormat

        assert FileFormat is not None
        assert Encoding is not None
        assert CompressionLevel is not None


class TestPathTypes:
    """Tests for path type alias usage."""

    def test_pathlike_accepts_string(self) -> None:
        """Verify strings work with PathLike."""
        from davinci_monet.core.types import PathLike

        def process_path(p: PathLike) -> str:
            return str(p)

        assert process_path("/path/to/file.nc") == "/path/to/file.nc"

    def test_pathlike_accepts_path(self) -> None:
        """Verify Path objects work with PathLike."""
        from davinci_monet.core.types import PathLike

        def process_path(p: PathLike) -> str:
            return str(p)

        assert process_path(Path("/path/to/file.nc")) == "/path/to/file.nc"


class TestTimeTypes:
    """Tests for time type alias usage."""

    def test_timestamp_accepts_datetime(self) -> None:
        """Verify datetime works with Timestamp."""
        from davinci_monet.core.types import Timestamp

        def process_time(t: Timestamp) -> str:
            if isinstance(t, datetime):
                return t.isoformat()
            return str(t)

        dt = datetime(2024, 1, 15, 12, 0)
        assert "2024-01-15" in process_time(dt)

    def test_timestamp_accepts_string(self) -> None:
        """Verify ISO string works with Timestamp."""
        from davinci_monet.core.types import Timestamp

        def process_time(t: Timestamp) -> str:
            return str(t)

        assert process_time("2024-01-15T12:00:00") == "2024-01-15T12:00:00"

    def test_timestamp_accepts_numpy_datetime(self) -> None:
        """Verify numpy datetime64 works with Timestamp."""
        from davinci_monet.core.types import Timestamp

        def process_time(t: Timestamp) -> str:
            return str(t)

        dt = np.datetime64("2024-01-15")
        assert "2024-01-15" in process_time(dt)


class TestNumericTypes:
    """Tests for numeric type alias usage."""

    def test_number_accepts_int(self) -> None:
        """Verify int works with Number."""
        from davinci_monet.core.types import Number

        def double(n: Number) -> float:
            return float(n) * 2

        assert double(5) == 10.0

    def test_number_accepts_float(self) -> None:
        """Verify float works with Number."""
        from davinci_monet.core.types import Number

        def double(n: Number) -> float:
            return float(n) * 2

        assert double(2.5) == 5.0

    def test_number_accepts_numpy_types(self) -> None:
        """Verify numpy numeric types work with Number."""
        from davinci_monet.core.types import Number

        def double(n: Number) -> float:
            return float(n) * 2

        assert double(np.float64(3.0)) == 6.0
        assert double(np.int32(4)) == 8.0


class TestCoordinateTypes:
    """Tests for coordinate type alias usage."""

    def test_coordinate_tuple(self) -> None:
        """Verify Coordinate is a (lon, lat) tuple."""
        from davinci_monet.core.types import Coordinate

        coord: Coordinate = (-105.0, 40.0)
        lon, lat = coord
        assert lon == -105.0
        assert lat == 40.0

    def test_bounding_box_tuple(self) -> None:
        """Verify BoundingBox is a 4-tuple."""
        from davinci_monet.core.types import BoundingBox

        bbox: BoundingBox = (-110.0, -100.0, 35.0, 45.0)
        lon_min, lon_max, lat_min, lat_max = bbox
        assert lon_min == -110.0
        assert lon_max == -100.0
        assert lat_min == 35.0
        assert lat_max == 45.0

    def test_coordinate_3d_tuple(self) -> None:
        """Verify Coordinate3D is a (lon, lat, alt) tuple."""
        from davinci_monet.core.types import Coordinate3D

        coord: Coordinate3D = (-105.0, 40.0, 1500.0)
        lon, lat, alt = coord
        assert lon == -105.0
        assert lat == 40.0
        assert alt == 1500.0


class TestVariableTypes:
    """Tests for variable-related type alias usage."""

    def test_variable_mapping(self) -> None:
        """Verify VariableMapping is a string-to-string mapping."""
        from davinci_monet.core.types import VariableMapping

        mapping: VariableMapping = {
            "ozone": "O3",
            "pm25": "PM2.5",
            "nitrogen_dioxide": "NO2",
        }
        assert mapping["ozone"] == "O3"

    def test_variable_dict(self) -> None:
        """Verify VariableDict is a nested dict structure."""
        from davinci_monet.core.types import VariableDict

        var_config: VariableDict = {
            "O3": {
                "units": "ppb",
                "long_name": "Ozone",
                "obs_min": 0.0,
            }
        }
        assert var_config["O3"]["units"] == "ppb"


class TestStatisticsTypes:
    """Tests for statistics type alias usage."""

    def test_metric_dict(self) -> None:
        """Verify MetricDict is a string-to-float mapping."""
        from davinci_monet.core.types import MetricDict

        stats: MetricDict = {
            "MB": 0.5,
            "RMSE": 2.3,
            "R2": 0.85,
        }
        assert stats["MB"] == 0.5
        assert stats["R2"] == 0.85

    def test_groupby_single(self) -> None:
        """Verify GroupBy accepts single string."""
        from davinci_monet.core.types import GroupBy

        group: GroupBy = "site"
        assert group == "site"

    def test_groupby_multiple(self) -> None:
        """Verify GroupBy accepts sequence of strings."""
        from davinci_monet.core.types import GroupBy

        groups: GroupBy = ["site", "time.month"]
        assert len(groups) == 2


class TestPlottingTypes:
    """Tests for plotting type alias usage."""

    def test_color_string(self) -> None:
        """Verify Color accepts string."""
        from davinci_monet.core.types import Color

        color: Color = "blue"
        assert color == "blue"

    def test_color_rgb_tuple(self) -> None:
        """Verify Color accepts RGB tuple."""
        from davinci_monet.core.types import Color

        color: Color = (0.2, 0.4, 0.6)
        assert len(color) == 3

    def test_color_rgba_tuple(self) -> None:
        """Verify Color accepts RGBA tuple."""
        from davinci_monet.core.types import Color

        color: Color = (0.2, 0.4, 0.6, 0.8)
        assert len(color) == 4
