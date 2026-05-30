"""Tests for the unified data-source pipeline plumbing (Phase 3).

Phase 3 is additive: it introduces ``PipelineContext.sources`` + ``get_source``
and a ``LoadSourcesStage`` that unifies model/observation loading into a single
``context.sources`` view (tagging each dataset with role/source_label/geometry),
while leaving the legacy ``LoadModelsStage``/``LoadObservationsStage`` and the
``models``/``observations`` context dicts untouched.

These are unit tests: they construct data containers directly and exercise the
context API and stage logic, per the repo's existing pipeline-stage test pattern
(see test_obs_pipeline.py).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.models.base import ModelData
from davinci_monet.observations.base import ObservationData
from davinci_monet.pipeline.stages import (
    LoadSourcesStage,
    PipelineContext,
    StageStatus,
)


def _point_obs_dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n_t, n_s = 12, 4
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    return xr.Dataset(
        {
            "o3": (("time", "site"), rng.uniform(10, 60, (n_t, n_s)), {"units": "ppb"}),
        },
        coords={
            "time": times,
            "site": np.arange(n_s),
            "latitude": ("site", rng.uniform(0, 40, n_s)),
            "longitude": ("site", rng.uniform(90, 140, n_s)),
        },
    )


def _grid_model_dataset() -> xr.Dataset:
    rng = np.random.default_rng(1)
    n_t, n_lat, n_lon = 12, 5, 6
    times = np.datetime64("2024-02-01") + np.arange(n_t) * np.timedelta64(1, "h")
    return xr.Dataset(
        {"O3": (("time", "lat", "lon"), rng.uniform(10, 60, (n_t, n_lat, n_lon)))},
        coords={
            "time": times,
            "lat": np.linspace(0, 40, n_lat),
            "lon": np.linspace(90, 140, n_lon),
        },
    )


@pytest.fixture
def model_data() -> ModelData:
    return ModelData(data=_grid_model_dataset(), label="cam", mod_type="generic")


@pytest.fixture
def obs_data() -> ObservationData:
    return ObservationData(
        data=_point_obs_dataset(),
        label="airnow",
        obs_type="pt_sfc",
        _geometry=DataGeometry.POINT,
    )


class TestPipelineContextSources:
    def test_sources_defaults_to_empty_dict(self) -> None:
        ctx = PipelineContext()
        assert ctx.sources == {}

    def test_get_source_returns_registered(self, obs_data: ObservationData) -> None:
        ctx = PipelineContext(sources={"airnow": obs_data})
        assert ctx.get_source("airnow") is obs_data

    def test_get_source_missing_raises_keyerror(self) -> None:
        ctx = PipelineContext()
        with pytest.raises(KeyError):
            ctx.get_source("nope")


class TestLoadSourcesStage:
    def test_unifies_models_and_observations_with_tags(
        self, model_data: ModelData, obs_data: ObservationData
    ) -> None:
        # No model/obs config blocks -> delegate loaders are skipped; the stage
        # unifies the already-populated containers into context.sources.
        ctx = PipelineContext(
            models={"cam": model_data},
            observations={"airnow": obs_data},
        )
        result = LoadSourcesStage().execute(ctx)

        assert result.status is StageStatus.COMPLETED
        # Both sources exposed via the unified view.
        assert set(ctx.sources) == {"cam", "airnow"}
        assert ctx.get_source("cam") is model_data
        assert ctx.get_source("airnow") is obs_data

        # Datasets tagged with role / source_label / geometry.
        cam_attrs = ctx.sources["cam"].data.attrs
        assert cam_attrs["role"] == "model"
        assert cam_attrs["source_label"] == "cam"
        assert cam_attrs["geometry"] == "grid"

        air_attrs = ctx.sources["airnow"].data.attrs
        assert air_attrs["role"] == "obs"
        assert air_attrs["source_label"] == "airnow"
        assert air_attrs["geometry"] == "point"

    def test_legacy_context_dicts_preserved(
        self, model_data: ModelData, obs_data: ObservationData
    ) -> None:
        ctx = PipelineContext(
            models={"cam": model_data},
            observations={"airnow": obs_data},
        )
        LoadSourcesStage().execute(ctx)
        # Backward-compatible accessors still work unchanged.
        assert ctx.get_model("cam") is model_data
        assert ctx.get_observation("airnow") is obs_data

    def test_stage_name(self) -> None:
        assert LoadSourcesStage().name == "load_sources"
