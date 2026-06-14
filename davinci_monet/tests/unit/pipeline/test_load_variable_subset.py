"""LoadSourcesStage subsets a source to its configured variables.

Regression: a source using ``source_name`` mappings retained ALL raw columns,
because the reader's ``open(variables=)`` subset is keyed on the post-rename
standard names (e.g. ``O3``) which do not match the pre-rename raw columns
(e.g. ``O3_ESRL``) at open time — so ``select_variables`` no-ops and keeps every
column. The load stage must subset to the configured variables AFTER renaming,
preserving coordinates (track/profile plots depend on lat/lon/alt).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.registry import source_registry
from davinci_monet.pipeline.stages import LoadSourcesStage, PipelineContext, StageStatus


class _RawColumnsReader:
    """Test reader returning a raw dataset with extra columns + coords.

    It ignores ``variables`` — exactly as the ICARTT reader effectively does
    when the requested (post-rename) names do not match the raw source columns
    — so the load stage alone is responsible for subsetting.
    """

    @property
    def name(self) -> str:
        return "raw_columns_probe"

    @property
    def geometry(self) -> DataGeometry:
        return DataGeometry.TRACK

    def open(self, file_paths, variables=None, time_range=None, **kwargs):  # noqa: ANN001
        n = 5
        time = np.datetime64("2012-05-29T00") + np.arange(n) * np.timedelta64(1, "h")
        return xr.Dataset(
            {
                "O3_RAW": ("time", np.linspace(30, 80, n)),
                "CO_RAW": ("time", np.linspace(80, 150, n)),
                "JUNK1": ("time", np.zeros(n)),
                "JUNK2": ("time", np.ones(n)),
            },
            coords={
                "time": time,
                "latitude": ("time", np.linspace(35, 40, n)),
                "longitude": ("time", np.linspace(-100, -95, n)),
                "altitude": ("time", np.linspace(0, 10000, n)),
            },
        )


@pytest.fixture
def register_probe():
    source_registry.register("raw_columns_probe", _RawColumnsReader, replace=True)
    try:
        yield
    finally:
        source_registry.unregister("raw_columns_probe")


def test_load_subsets_to_configured_variables_preserving_coords(register_probe) -> None:
    ctx = PipelineContext(
        config={
            "sources": {
                "dc8": {
                    "type": "raw_columns_probe",
                    "filename": "ignored.nc",
                    "variables": {
                        "O3": {"source_name": "O3_RAW"},
                        "CO": {"source_name": "CO_RAW"},
                    },
                }
            }
        }
    )

    result = LoadSourcesStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    ds = ctx.sources["dc8"].data
    # Subset to exactly the configured (renamed) variables — unconfigured raw
    # columns dropped.
    assert set(ds.data_vars) == {"O3", "CO"}
    # Coordinates preserved (track/profile plots depend on them).
    assert {"latitude", "longitude", "altitude"} <= set(ds.coords)


def test_load_without_variable_config_keeps_all(register_probe) -> None:
    """A source that configures no variables loads every column (no subset)."""
    ctx = PipelineContext(
        config={"sources": {"dc8": {"type": "raw_columns_probe", "filename": "ignored.nc"}}}
    )

    result = LoadSourcesStage().execute(ctx)

    assert result.status is StageStatus.COMPLETED
    ds = ctx.sources["dc8"].data
    assert {"O3_RAW", "CO_RAW", "JUNK1", "JUNK2"} == set(ds.data_vars)
