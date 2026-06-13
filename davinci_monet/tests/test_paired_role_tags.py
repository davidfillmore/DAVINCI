"""Tests for role/source-label tagging on paired output."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages import tag_paired_roles


def _paired() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    return xr.Dataset(
        {
            "model_o3": ("time", rng.uniform(10, 60, n)),
            "obs_o3": ("time", rng.uniform(10, 60, n)),
        },
        coords={"time": np.arange(n)},
    )


class TestTagPairedRoles:
    def test_tags_role_by_prefix(self) -> None:
        ds = _paired()
        tag_paired_roles(ds)
        assert ds["comparand_o3"].attrs["role"] == "model"
        assert ds["reference_o3"].attrs["role"] == "obs"

    def test_renames_to_neutral_source_labels(self) -> None:
        ds = _paired()
        tag_paired_roles(ds)
        assert set(ds.data_vars) == {"comparand_o3", "reference_o3"}
        assert ds["comparand_o3"].attrs["source_label"] == "comparand"
        assert ds["reference_o3"].attrs["source_label"] == "reference"

    def test_does_not_overwrite_existing_role(self) -> None:
        ds = _paired()
        ds["model_o3"].attrs["role"] = "obs"  # pre-set; must be preserved
        tag_paired_roles(ds)
        assert ds["comparand_o3"].attrs["role"] == "obs"

    def test_none_is_safe(self) -> None:
        # Should not raise on None input.
        tag_paired_roles(None)
