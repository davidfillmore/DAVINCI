"""P1 of the renderer unification: single-source tagging + count-aware color rule.

Pure additions consumed later (P2 wires tagging into the stage; P3 wires colors
into render()). No behavior change to existing renderers here.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from davinci_monet.core.base import PlotSeries, iter_canonical_variable_series
from davinci_monet.plots.style import MODEL_COLOR, NCAR_PALETTE, NCAR_PRIMARY, OBS_COLOR


def _series(role, pair_role, label, index):
    ds = xr.Dataset({"o3": ("time", [1.0, 2.0])}, coords={"time": [0, 1]})
    return PlotSeries(
        dataset=ds,
        var_name="o3",
        canonical="o3",
        role=role,
        pair_role=pair_role,
        source_label=label,
        index=index,
    )


class TestTagSourceRoles:
    def test_tags_each_var_and_series_resolver_sees_them(self) -> None:
        from davinci_monet.pipeline.stages import tag_source_roles

        ds = xr.Dataset(
            {"O3": ("time", [1.0, 2.0, 3.0]), "NO2": ("time", [1.0, 2.0, 3.0])},
            coords={"time": np.arange(3)},
        )
        ds.attrs["source_label"] = "airnow"
        tag_source_roles(ds, role="obs", source_label="airnow")
        assert ds["O3"].attrs["role"] == "obs"
        assert ds["O3"].attrs["source_label"] == "airnow"
        # The N-capable resolver now picks up the (previously invisible) single source.
        groups = iter_canonical_variable_series(ds)
        assert set(groups) == {"O3", "NO2"}
        assert groups["O3"][0].role == "obs"
        assert groups["O3"][0].source_label == "airnow"

    def test_does_not_overwrite_existing_role(self) -> None:
        from davinci_monet.pipeline.stages import tag_source_roles

        ds = xr.Dataset({"O3": ("time", [1.0])}, coords={"time": [0]})
        ds["O3"].attrs["role"] = "model"
        tag_source_roles(ds, role="obs", source_label="cam")
        assert ds["O3"].attrs["role"] == "model"  # preserved


class TestSeriesColors:
    def test_single_obs_source_is_brand_blue_not_gray(self) -> None:
        from davinci_monet.plots.base import series_colors

        assert series_colors([_series("obs", "reference", "airnow", 0)]) == [NCAR_PRIMARY]

    def test_single_model_source_is_blue(self) -> None:
        from davinci_monet.plots.base import series_colors

        assert series_colors([_series("model", "comparand", "cam", 0)]) == [MODEL_COLOR]

    def test_single_roleless_source_is_brand_blue(self) -> None:
        from davinci_monet.plots.base import series_colors

        assert series_colors([_series(None, None, None, 0)]) == [NCAR_PRIMARY]

    def test_paired_obs_gray_model_blue(self) -> None:
        from davinci_monet.plots.base import series_colors

        series = [
            _series("obs", "reference", "airnow", 0),
            _series("model", "comparand", "cam", 1),
        ]
        assert series_colors(series) == [OBS_COLOR, MODEL_COLOR]

    def test_n_source_distinct_palette(self) -> None:
        from davinci_monet.plots.base import series_colors

        series = [
            _series("obs", "reference", "airnow", 0),
            _series("model", "comparand", "cam", 1),
            _series("model", "comparand", "cam2", 2),
        ]
        colors = series_colors(series)
        assert len(set(colors)) == 3  # all distinct
        assert colors[0] == NCAR_PALETTE[0]
