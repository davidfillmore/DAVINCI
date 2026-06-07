"""Tests for the unified sources config schema (Phase 6, CFG-1).

The unified ``sources:`` block plus binary ``SourcePairConfig`` is the only
supported data-source schema; the legacy ``model:``/``obs:`` blocks were removed
and are now rejected at config load.
"""

from __future__ import annotations

import pytest

from davinci_monet.config.parser import validate_config
from davinci_monet.config.schema import MonetConfig, SourceConfig, SourcePairConfig
from davinci_monet.core.exceptions import ConfigurationError


class TestSourceConfig:
    def test_parses_sources_block(self) -> None:
        cfg = MonetConfig.model_validate(
            {
                "sources": {
                    "cam": {
                        "type": "cesm_fv",
                        "role": "model",
                        "files": "/data/cam/*.nc",
                        "radius_of_influence": 15000,
                        "variables": {"O3": {"unit_scale": 1.0e9}},
                    },
                    "airnow": {
                        "type": "pt_sfc",
                        "role": "obs",
                        "filename": "/data/airnow.nc",
                        "variables": {"o3": {"obs_min": 0, "obs_max": 500}},
                    },
                }
            }
        )
        assert set(cfg.sources) == {"cam", "airnow"}
        cam = cfg.sources["cam"]
        assert isinstance(cam, SourceConfig)
        assert cam.type == "cesm_fv"
        assert cam.role == "model"
        assert cam.radius_of_influence == 15000
        # variables parsed into VariableConfig with unit_scale preserved.
        assert cam.variables["O3"].unit_scale == 1.0e9
        assert cfg.sources["airnow"].role == "obs"

    def test_role_is_optional(self) -> None:
        cfg = MonetConfig.model_validate(
            {"sources": {"x": {"type": "pt_sfc", "filename": "/d.nc"}}}
        )
        assert cfg.sources["x"].role is None

    def test_sources_default_empty(self) -> None:
        assert MonetConfig().sources == {}


class TestSourcePairConfig:
    def test_binary_pair_parses(self) -> None:
        pair = SourcePairConfig(
            sources=["cam", "airnow"],
            reference="airnow",
            variables={"cam": "O3", "airnow": "o3"},
        )
        assert pair.sources == ["cam", "airnow"]
        assert pair.reference == "airnow"
        assert pair.variables == {"cam": "O3", "airnow": "o3"}

    def test_reference_optional(self) -> None:
        pair = SourcePairConfig(sources=["a", "b"], variables={"a": "v", "b": "v"})
        assert pair.reference is None


class TestLegacyRejected:
    def test_model_obs_blocks_rejected_at_load(self) -> None:
        """Legacy model:/obs: blocks are a hard error pointing at migrate-config."""
        with pytest.raises(ConfigurationError, match="migrate-config"):
            validate_config(
                {
                    "model": {"cam": {"mod_type": "cesm_fv", "files": "/d/*.nc"}},
                    "obs": {"airnow": {"obs_type": "pt_sfc", "filename": "/a.nc"}},
                }
            )

    def test_schema_has_no_model_obs_fields(self) -> None:
        """The root schema no longer defines model:/obs: fields."""
        assert "model" not in MonetConfig.model_fields
        assert "obs" not in MonetConfig.model_fields
