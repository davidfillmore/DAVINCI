"""Tests for sources -> legacy expansion making sources: configs runnable (CFG-3)."""

from __future__ import annotations

from typing import Any

from davinci_monet.config.migration import expand_sources_to_legacy


def _unified() -> dict[str, Any]:
    return {
        "analysis": {"start_time": "2024-02-01", "end_time": "2024-02-29"},
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
                "variables": {"o3": {"obs_min": 0}},
            },
        },
        "pairs": {
            "cam_airnow_o3": {
                "sources": ["cam", "airnow"],
                "reference": "airnow",
                "variables": {"cam": "O3", "airnow": "o3"},
                "title": "O3",
            }
        },
        "plots": {"s": {"type": "scatter", "pairs": ["cam_airnow_o3"]}},
    }


class TestExpandSourcesToLegacy:
    def test_sources_split_into_model_and_obs(self) -> None:
        out = expand_sources_to_legacy(_unified())
        assert "sources" not in out
        assert out["model"]["cam"]["mod_type"] == "cesm_fv"
        assert "role" not in out["model"]["cam"]
        assert out["model"]["cam"]["radius_of_influence"] == 15000
        assert out["obs"]["airnow"]["obs_type"] == "pt_sfc"
        assert out["obs"]["airnow"]["filename"] == "/data/airnow.nc"

    def test_binary_pair_becomes_legacy(self) -> None:
        out = expand_sources_to_legacy(_unified())
        p = out["pairs"]["cam_airnow_o3"]
        assert p["model"] == "cam"
        assert p["obs"] == "airnow"
        assert p["variable"] == {"model_var": "O3", "obs_var": "o3"}
        assert p["title"] == "O3"

    def test_role_less_inference_by_type(self) -> None:
        cfg = {
            "sources": {
                "cam": {"type": "cesm_fv", "files": "/d/*.nc"},  # model type
                "site": {"type": "pt_sfc", "filename": "/o.nc"},  # obs type
            }
        }
        out = expand_sources_to_legacy(cfg)
        assert "cam" in out["model"]
        assert "site" in out["obs"]

    def test_non_sources_config_unchanged(self) -> None:
        legacy = {"model": {"m": {"mod_type": "cmaq"}}, "obs": {"o": {"obs_type": "pt_sfc"}}}
        assert expand_sources_to_legacy(legacy) == legacy

    def test_roundtrip_with_migrate(self) -> None:
        from davinci_monet.config.migration import migrate_to_sources

        legacy = {
            "model": {"cam": {"mod_type": "cesm_fv", "files": "/d/*.nc"}},
            "obs": {"airnow": {"obs_type": "pt_sfc", "filename": "/a.nc"}},
            "pairs": {
                "p": {
                    "model": "cam",
                    "obs": "airnow",
                    "variable": {"model_var": "O3", "obs_var": "o3"},
                }
            },
        }
        back = expand_sources_to_legacy(migrate_to_sources(legacy))
        assert back["model"]["cam"]["mod_type"] == "cesm_fv"
        assert back["obs"]["airnow"]["obs_type"] == "pt_sfc"
        assert back["pairs"]["p"]["model"] == "cam"
        assert back["pairs"]["p"]["variable"] == {"model_var": "O3", "obs_var": "o3"}


class TestRunnerAcceptsSourcesConfig:
    def test_sources_only_config_not_rejected_as_empty(self, tmp_path: Any) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner

        config = {
            "sources": {"airnow": {"type": "pt_sfc", "role": "obs", "filename": "/fake/path.nc"}},
            "analysis": {"output_dir": str(tmp_path / "out")},
        }
        runner = PipelineRunner(show_progress=False)
        # Should expand sources -> obs and run the pipeline (load fails on the
        # fake path, but it must NOT raise the "empty configuration" error).
        result = runner.run_from_config(config)
        assert result is not None
        stage_names = [s.name for s in runner.stages]
        assert "load_sources" in stage_names
