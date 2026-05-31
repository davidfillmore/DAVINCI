"""Tests for source-label variable resolution in the plotting stage (R-2).

R-2 of the renderer rewire: PlottingStage resolves which paired variables to
hand to a renderer via the source labels rather than hard-coding the legacy
``obs_``/``model_`` prefixes. It prefers the ``<source_label>_<var>`` aliases
added by tag_paired_roles (R-1) and falls back to the legacy prefix when no
alias exists (older paired data, or a label in the reserved namespace). This
keeps plotting green today and future-proofs the clean break in R-5.

obs is the reference and model the comparand, and the pairing engine names both
paired variables off the *obs* canonical name (``model_<obs_var>``), so both
resolutions key off ``obs_var``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import xarray as xr
from matplotlib.text import Text

from davinci_monet.pipeline.stages import resolve_paired_var_names, tag_paired_roles
from davinci_monet.plots.base import canonical_variable_name, get_variable_label


def _legacy_paired() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    return xr.Dataset(
        {
            "model_o3": ("time", rng.uniform(10, 60, n)),
            "obs_o3": ("time", rng.uniform(10, 60, n)),
        },
        coords={"time": np.arange(n)},
    )


def _paired_with_aliases(reference: str, comparand: str) -> xr.Dataset:
    ds = _legacy_paired()
    tag_paired_roles(ds, reference_label=reference, comparand_label=comparand)
    return ds


class TestResolvePairedVarNames:
    def test_prefers_source_label_aliases(self) -> None:
        ds = _paired_with_aliases(reference="airnow", comparand="cam")
        obs_name, model_name = resolve_paired_var_names(ds, "o3", "airnow", "cam")
        assert obs_name == "airnow_o3"
        assert model_name == "cam_o3"

    def test_falls_back_to_legacy_prefixes_when_no_alias(self) -> None:
        # Legacy-only paired data (no aliases tagged) must still resolve.
        ds = _legacy_paired()
        obs_name, model_name = resolve_paired_var_names(ds, "o3", "airnow", "cam")
        assert obs_name == "obs_o3"
        assert model_name == "model_o3"

    def test_reserved_prefix_labels_resolve_to_legacy(self) -> None:
        # Labels that collide with the reserved namespace get no alias (R-1), so
        # resolution lands on the legacy vars, which carry the same data.
        ds = _paired_with_aliases(reference="obs", comparand="model")
        obs_name, model_name = resolve_paired_var_names(ds, "o3", "obs", "model")
        assert obs_name == "obs_o3"
        assert model_name == "model_o3"

    def test_returns_legacy_strings_even_when_absent(self) -> None:
        # When neither alias nor legacy var is present, fall back to the legacy
        # prefix strings; the caller guards on membership before plotting.
        ds = xr.Dataset({"unrelated": ("time", np.zeros(4))}, coords={"time": np.arange(4)})
        obs_name, model_name = resolve_paired_var_names(ds, "o3", "airnow", "cam")
        assert obs_name == "obs_o3"
        assert model_name == "model_o3"


class TestCanonicalVariableName:
    def test_strips_source_label_prefix(self) -> None:
        ds = _paired_with_aliases(reference="airnow", comparand="cam")
        assert canonical_variable_name(ds, "airnow_o3") == "o3"
        assert canonical_variable_name(ds, "cam_o3") == "o3"

    def test_strips_legacy_prefix(self) -> None:
        ds = _legacy_paired()
        assert canonical_variable_name(ds, "obs_o3") == "o3"
        assert canonical_variable_name(ds, "model_o3") == "o3"

    def test_unprefixed_name_unchanged(self) -> None:
        ds = _legacy_paired()
        assert canonical_variable_name(ds, "o3") == "o3"


class TestGetVariableLabelPreserved:
    """R-2 must not change rendered labels when var names switch to source labels."""

    def test_alias_label_matches_legacy_with_prefix(self) -> None:
        ds = _paired_with_aliases(reference="airnow", comparand="cam")
        # Observed/Modeled prefix + canonical lookup preserved despite the
        # source-label variable name.
        assert get_variable_label(ds, "airnow_o3") == get_variable_label(ds, "obs_o3")
        assert get_variable_label(ds, "cam_o3") == get_variable_label(ds, "model_o3")

    def test_alias_label_matches_legacy_without_prefix(self) -> None:
        ds = _paired_with_aliases(reference="airnow", comparand="cam")
        assert get_variable_label(ds, "airnow_o3", include_prefix=False) == get_variable_label(
            ds, "obs_o3", include_prefix=False
        )

    def test_explicit_attrs_still_win(self) -> None:
        # display_name/long_name override formatting for both alias and legacy.
        ds = _legacy_paired()
        ds["obs_o3"].attrs["long_name"] = "Surface Ozone"
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        assert get_variable_label(ds, "airnow_o3") == "Surface Ozone"


class TestScorecardCanonicalRowLabel:
    def test_scorecard_row_label_is_canonical(self) -> None:
        from davinci_monet.plots.renderers.scorecard import ScorecardPlotter

        ds = _paired_with_aliases(reference="airnow", comparand="cam")
        fig = ScorecardPlotter().plot(ds, "airnow_o3", "cam_o3")
        text_artists = cast(list[Text], fig.findobj(match=lambda o: hasattr(o, "get_text")))
        texts = {t.get_text() for t in text_artists}
        assert "o3" in texts
        assert "airnow_o3" not in texts
