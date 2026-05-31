"""Tests for model/obs -> sources config migration (Phase 6, CFG-2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from davinci_monet.config.migration import migrate_to_sources
from davinci_monet.config.schema import MonetConfig


def _legacy() -> dict[str, Any]:
    return {
        "analysis": {"start_time": "2024-02-01", "end_time": "2024-02-29"},
        "model": {
            "cam": {
                "mod_type": "cesm_fv",
                "files": "/data/cam/*.nc",
                "radius_of_influence": 15000,
                "variables": {"O3": {"unit_scale": 1.0e9}},
            }
        },
        "obs": {
            "airnow": {
                "obs_type": "pt_sfc",
                "filename": "/data/airnow.nc",
                "variables": {"o3": {"obs_min": 0}},
            }
        },
        "pairs": {
            "cam_airnow_o3": {
                "model": "cam",
                "obs": "airnow",
                "variable": {"model_var": "O3", "obs_var": "o3"},
                "title": "O3",
            }
        },
        "plots": {"s": {"type": "scatter", "pairs": ["cam_airnow_o3"]}},
    }


class TestMigrateToSources:
    def test_model_and_obs_become_sources_with_roles(self) -> None:
        out = migrate_to_sources(_legacy())
        assert "model" not in out and "obs" not in out
        assert out["sources"]["cam"]["type"] == "cesm_fv"
        assert out["sources"]["cam"]["role"] == "model"
        assert out["sources"]["cam"]["radius_of_influence"] == 15000
        assert out["sources"]["cam"]["variables"] == {"O3": {"unit_scale": 1.0e9}}
        assert out["sources"]["airnow"]["type"] == "pt_sfc"
        assert out["sources"]["airnow"]["role"] == "obs"
        assert out["sources"]["airnow"]["filename"] == "/data/airnow.nc"

    def test_pairs_become_binary_with_reference_obs(self) -> None:
        out = migrate_to_sources(_legacy())
        p = out["pairs"]["cam_airnow_o3"]
        assert p["sources"] == ["cam", "airnow"]
        assert p["reference"] == "airnow"
        assert p["variables"] == {"cam": "O3", "airnow": "o3"}
        # Non-structural keys preserved.
        assert p["title"] == "O3"

    def test_analysis_and_plots_preserved(self) -> None:
        out = migrate_to_sources(_legacy())
        assert out["analysis"]["start_time"] == "2024-02-01"
        assert out["plots"]["s"]["pairs"] == ["cam_airnow_o3"]

    def test_migrated_config_parses_with_sources(self) -> None:
        out = migrate_to_sources(_legacy())
        cfg = MonetConfig(**out)
        assert set(cfg.sources) == {"cam", "airnow"}
        assert cfg.model == {} and cfg.obs == {}

    def test_idempotent_on_already_unified(self) -> None:
        unified = migrate_to_sources(_legacy())
        again = migrate_to_sources(unified)
        assert again["sources"].keys() == unified["sources"].keys()
        assert "model" not in again and "obs" not in again


class TestMigrateCLI:
    def test_cli_writes_migrated_yaml(self, tmp_path: Path) -> None:
        from davinci_monet.cli.commands.migrate import migrate_config_command

        src = tmp_path / "legacy.yaml"
        dst = tmp_path / "unified.yaml"
        src.write_text(yaml.dump(_legacy()))
        migrate_config_command(str(src), str(dst))
        written = yaml.safe_load(dst.read_text())
        assert "sources" in written and "model" not in written
        assert written["sources"]["cam"]["role"] == "model"
