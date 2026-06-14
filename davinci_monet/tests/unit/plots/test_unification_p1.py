"""P1 of the renderer unification: single-source tagging + count-aware color rule.

Pure additions consumed later (P2 wires tagging into the stage; P3 wires colors
into render()). No behavior change to existing renderers here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries, iter_canonical_variable_series
from davinci_monet.plots.style import DATASET_A_COLOR, DATASET_B_COLOR, NCAR_PALETTE, NCAR_PRIMARY


def _series(pair_axis, label, index):
    ds = xr.Dataset({"o3": ("time", [1.0, 2.0])}, coords={"time": [0, 1]})
    return PlotSeries(
        dataset=ds,
        var_name="o3",
        canonical="o3",
        pair_axis=pair_axis,
        dataset_label=label,
        index=index,
    )


class TestTagDatasetLabels:
    def test_tags_each_var_and_series_resolver_sees_them(self) -> None:
        from davinci_monet.pipeline.stages import tag_source_label

        ds = xr.Dataset(
            {"O3": ("time", [1.0, 2.0, 3.0]), "NO2": ("time", [1.0, 2.0, 3.0])},
            coords={"time": np.arange(3)},
        )
        ds.attrs["dataset_label"] = "airnow"
        tag_source_label(ds, dataset_label="airnow")
        assert ds["O3"].attrs["dataset_label"] == "airnow"
        # The N-capable resolver now picks up the (previously invisible) single source.
        groups = iter_canonical_variable_series(ds)
        assert set(groups) == {"O3", "NO2"}
        assert groups["O3"][0].pair_axis is None
        assert groups["O3"][0].dataset_label == "airnow"

    def test_does_not_overwrite_existing_pair_axis(self) -> None:
        from davinci_monet.pipeline.stages import tag_source_label

        ds = xr.Dataset({"O3": ("time", [1.0])}, coords={"time": [0]})
        ds["O3"].attrs["pair_axis"] = "dataset"
        tag_source_label(ds, dataset_label="cam")
        assert ds["O3"].attrs["pair_axis"] == "dataset"  # preserved


class TestSeriesColors:
    def test_single_geometry_source_is_brand_blue_not_gray(self) -> None:
        from davinci_monet.plots.base import series_colors

        assert series_colors([_series("geometry", "airnow", 0)]) == [NCAR_PRIMARY]

    def test_single_dataset_source_is_blue(self) -> None:
        from davinci_monet.plots.base import series_colors

        assert series_colors([_series("dataset", "cam", 0)]) == [NCAR_PRIMARY]

    def test_single_roleless_source_is_brand_blue(self) -> None:
        from davinci_monet.plots.base import series_colors

        assert series_colors([_series(None, None, 0)]) == [NCAR_PRIMARY]

    def test_paired_geometry_gray_dataset_blue(self) -> None:
        from davinci_monet.plots.base import series_colors

        series = [
            _series("geometry", "airnow", 0),
            _series("dataset", "cam", 1),
        ]
        assert series_colors(series) == [DATASET_A_COLOR, DATASET_B_COLOR]

    def test_n_source_distinct_palette(self) -> None:
        from davinci_monet.plots.base import series_colors

        series = [
            _series("geometry", "airnow", 0),
            _series("dataset", "cam", 1),
            _series("dataset", "cam2", 2),
        ]
        colors = series_colors(series)
        assert len(set(colors)) == 3  # all distinct
        assert colors[0] == NCAR_PALETTE[0]
