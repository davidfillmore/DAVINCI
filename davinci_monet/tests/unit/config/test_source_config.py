"""Tests for the unified sources config schema (Phase 6, CFG-1).

Additive: introduces SourceConfig + a `sources:` block and a binary
SourcePairConfig alongside the existing model:/obs: schema, which remains
supported (deprecated) and is left untouched here.
"""

from __future__ import annotations

from davinci_monet.config.schema import MonetConfig, SourceConfig, SourcePairConfig


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


class TestLegacyStillParses:
    def test_model_obs_blocks_unchanged(self) -> None:
        cfg = MonetConfig.model_validate(
            {
                "model": {"cam": {"mod_type": "cesm_fv", "files": "/d/*.nc"}},
                "obs": {"airnow": {"obs_type": "pt_sfc", "filename": "/a.nc"}},
            }
        )
        assert "cam" in cfg.model
        assert "airnow" in cfg.obs
        # sources stays empty for legacy configs.
        assert cfg.sources == {}
