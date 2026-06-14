"""Tests for source-label variable resolution in the plotting stage."""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages import resolve_paired_var_names
from davinci_monet.plots.base import canonical_variable_name, get_variable_label


def _paired_with_labels(geometry: str, dataset: str) -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    ds = xr.Dataset(
        {
            f"{geometry}_o3": ("time", rng.uniform(10, 60, n)),
            f"{dataset}_o3": ("time", rng.uniform(10, 60, n)),
        },
        coords={"time": np.arange(n)},
    )
    ds[f"{geometry}_o3"].attrs.update(
        {"axis": "x", "source_label": geometry, "canonical_name": "o3"}
    )
    ds[f"{dataset}_o3"].attrs.update({"axis": "y", "source_label": dataset, "canonical_name": "o3"})
    return ds


class TestResolvePairedVarNames:
    def test_prefers_dataset_label_names(self) -> None:
        ds = _paired_with_labels(geometry="airnow", dataset="cam")
        x_name, y_name = resolve_paired_var_names(ds, "o3", "airnow", "cam")
        assert x_name == "airnow_o3"
        assert y_name == "cam_o3"

    def test_prefix_names_are_fallbacks(self) -> None:
        ds = xr.Dataset(
            {
                "x_o3": ("time", np.zeros(4)),
                "y_o3": ("time", np.ones(4)),
            },
            coords={"time": np.arange(4)},
        )
        x_name, y_name = resolve_paired_var_names(ds, "o3", "airnow", "cam")
        assert x_name == "x_o3"
        assert y_name == "y_o3"


class TestCanonicalVariableName:
    def test_strips_dataset_label_prefix(self) -> None:
        ds = _paired_with_labels(geometry="airnow", dataset="cam")
        assert canonical_variable_name(ds, "airnow_o3") == "o3"
        assert canonical_variable_name(ds, "cam_o3") == "o3"

    def test_unprefixed_name_unchanged(self) -> None:
        ds = _paired_with_labels(geometry="airnow", dataset="cam")
        assert canonical_variable_name(ds, "o3") == "o3"


class TestGetVariableLabelPreserved:
    """Source labels must not leak dataset/geometry axis words into labels."""

    def test_alias_label_formats_canonical_name(self) -> None:
        ds = _paired_with_labels(geometry="airnow", dataset="cam")
        assert get_variable_label(ds, "airnow_o3") == r"O$_3$"
        assert get_variable_label(ds, "cam_o3") == r"O$_3$"

    def test_explicit_attrs_still_win(self) -> None:
        ds = _paired_with_labels(geometry="airnow", dataset="cam")
        ds["airnow_o3"].attrs["long_name"] = "Surface Ozone"
        assert get_variable_label(ds, "airnow_o3") == "Surface Ozone"

    def test_label_uses_canonical_name_attr(self) -> None:
        ds = xr.Dataset(
            {
                "merra2_LWTUP": (
                    "time",
                    np.array([1.0, 2.0]),
                    {
                        "axis": "y",
                        "source_label": "merra2",
                        "dataset_variable": "LWTUP",
                        "canonical_name": "toa_lw_up",
                    },
                )
            },
            coords={"time": np.arange(2)},
        )
        assert get_variable_label(ds, "merra2_LWTUP") == "TOA LW Up"
