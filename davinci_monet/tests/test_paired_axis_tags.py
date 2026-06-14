"""Tests for source-label tagging helpers."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages import tag_source_label


def _source_dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    return xr.Dataset(
        {
            "o3": ("time", rng.uniform(10, 60, n)),
            "pm25": ("time", rng.uniform(1, 12, n)),
        },
        coords={"time": np.arange(n)},
    )


class TestTagDatasetLabel:
    def test_tags_each_data_variable(self) -> None:
        ds = _source_dataset()
        tag_source_label(ds, source_label="airnow")
        assert ds["o3"].attrs["source_label"] == "airnow"
        assert ds["pm25"].attrs["source_label"] == "airnow"

    def test_does_not_overwrite_existing_label(self) -> None:
        ds = _source_dataset()
        ds["o3"].attrs["source_label"] = "existing"
        tag_source_label(ds, source_label="airnow")
        assert ds["o3"].attrs["source_label"] == "existing"
        assert ds["pm25"].attrs["source_label"] == "airnow"

    def test_none_is_safe(self) -> None:
        tag_source_label(None, source_label="airnow")
