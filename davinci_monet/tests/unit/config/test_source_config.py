"""Tests for the unified sources config schema.

The unified ``sources:`` block plus binary ``SourcePairConfig`` is the only
supported data-source schema; top-level ``dataset:``/``geometry:`` blocks were removed
and are now rejected at config load.
"""

from __future__ import annotations

import pytest

from davinci_monet.config.parser import validate_config
from davinci_monet.config.schema import MonetConfig, SourceConfig, SourcePairConfig
from davinci_monet.core.exceptions import ConfigurationError
from davinci_monet.core.schema_utils import validate_schema


class TestSourceConfig:
    def test_parses_sources_block(self) -> None:
        cfg = validate_schema(
            MonetConfig,
            {
                "sources": {
                    "cam": {
                        "type": "cesm_fv",
                        "files": "/data/cam/*.nc",
                        "radius_of_influence": 15000,
                        "variables": {"O3": {"unit_scale": 1.0e9}},
                    },
                    "airnow": {
                        "type": "pt_sfc",
                        "filename": "/data/airnow.nc",
                        "variables": {"o3": {"valid_min": 0, "valid_max": 500}},
                    },
                }
            },
        )
        assert set(cfg.sources) == {"cam", "airnow"}
        cam = cfg.sources["cam"]
        assert isinstance(cam, SourceConfig)
        assert cam.type == "cesm_fv"
        assert cam.radius_of_influence == 15000
        # variables parsed into VariableConfig with unit_scale preserved.
        assert cam.variables["O3"].unit_scale == 1.0e9

    def test_pair_direction_is_not_a_source_field(self) -> None:
        cfg = validate_schema(
            MonetConfig, {"sources": {"x": {"type": "pt_sfc", "filename": "/d.nc"}}}
        )
        assert "axis" not in SourceConfig.__pydantic_fields__
        assert "axis" not in (cfg.sources["x"].__pydantic_extra__ or {})

    def test_sources_default_empty(self) -> None:
        assert MonetConfig().sources == {}


class TestSourcePairConfig:
    def test_binary_pair_parses(self) -> None:
        pair = SourcePairConfig(
            sources=["cam", "airnow"],
            geometry="airnow",
            variables={"cam": "O3", "airnow": "o3"},
        )
        assert pair.sources == ["cam", "airnow"]
        assert pair.geometry == "airnow"
        assert pair.variables == {"cam": "O3", "airnow": "o3"}

    def test_geometry_optional(self) -> None:
        pair = SourcePairConfig(sources=["a", "b"], variables={"a": "v", "b": "v"})
        assert pair.geometry is None


class TestRootConfigShape:
    def test_unknown_top_level_section_rejected_at_load(self) -> None:
        """Unknown top-level sections are rejected."""
        with pytest.raises(ConfigurationError, match="Extra inputs"):
            validate_config(
                {
                    "unsupported_group": {"cam": {"type": "cesm_fv", "files": "/d/*.nc"}},
                }
            )

    def test_schema_has_no_single_source_fields(self) -> None:
        """The root schema keeps all sources under ``sources``."""
        assert "dataset" not in MonetConfig.__pydantic_fields__
        assert "geometry" not in MonetConfig.__pydantic_fields__
