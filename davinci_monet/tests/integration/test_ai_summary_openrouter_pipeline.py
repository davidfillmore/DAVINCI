"""Integration: OpenRouter summary path through PipelineRunner.run_from_config.

The pipeline runs for real on synthetic data; only the OpenRouter HTTP send is
stubbed so no network call is made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import davinci_monet.ai.openrouter as orouter
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.datasets import create_dataset_dataset
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario, sample_geometry_from

pytestmark = pytest.mark.integration


def _build_config(tmp_path: Path) -> dict:
    domain = Domain(lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12)
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

    y_ds = create_dataset_dataset(variables=["O3"], domain=domain, time_config=time_cfg, seed=42)
    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_geometry=10,
        noise_level=0.0,
        seed=42,
    )
    x_ds = sample_geometry_from(y_ds, "point", scenario=scenario)

    rng = np.random.default_rng(42)
    y_ds["O3"] = y_ds["O3"] + 5.0 + rng.normal(0, 3.0, size=y_ds["O3"].shape)

    y_path = tmp_path / "dataset.nc"
    x_path = tmp_path / "geometry.nc"
    y_ds.to_netcdf(y_path)
    x_ds.to_netcdf(x_path)

    return {
        "analysis": {
            "start_time": "2024-01-15 00:00",
            "end_time": "2024-01-17 00:00",
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
        },
        "sources": {
            "synthetic": {
                "type": "generic",
                "files": str(y_path),
                "radius_of_influence": 50000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "surface": {
                "type": "pt_sfc",
                "filename": str(x_path),
                "variables": {"O3": {"valid_min": 0, "valid_max": 200, "units": "ppb"}},
            },
        },
        "pairs": {
            "synthetic_surface": {
                "x": {"source": "surface", "variable": "O3"},
                "y": {"source": "synthetic", "variable": "O3"},
            },
        },
        "plots": {
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Y vs X",
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]},
    }


def test_openrouter_summary_writes_file(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    calls = {}

    def _fake_send(cfg, key, body):
        calls["key"] = key
        calls["body"] = body
        return {
            "model": body["model"],
            "choices": [
                {"message": {"content": "## What this run is\nx\n## Caveats\nsynthetic\n"}}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda cfg, key: None)

    keyfile = tmp_path / "OpenRouter.api"
    keyfile.write_text("sk-or-fake")

    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(keyfile),
    }

    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(config)

    assert result.success
    summary_file = tmp_path / "output" / "AI_summary.md"
    assert summary_file.exists()
    assert "## Caveats" in summary_file.read_text()
    # key came from the file; the request carried an image_url vision block
    assert calls["key"] == "sk-or-fake"
    user_content = calls["body"]["messages"][1]["content"]
    assert any(block.get("type") == "image_url" for block in user_content)


def test_openrouter_summary_skips_without_key(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(tmp_path / "missing.api"),  # does not exist
    }

    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(config)

    assert result.success  # non-fatal: run still succeeds
    assert not (tmp_path / "output" / "AI_summary.md").exists()


def test_openrouter_summary_displays_tokens_and_credits(monkeypatch, tmp_path: Path) -> None:
    import davinci_monet.pipeline.runner as runner_module
    from davinci_monet.pipeline.runner import PipelineRunner

    def _fake_send(cfg, key, body):
        return {
            "model": body["model"],
            "choices": [{"message": {"content": "## Caveats\n- only point\n"}}],
            "usage": {"prompt_tokens": 123, "completion_tokens": 45},
        }

    monkeypatch.setattr(orouter, "_send_openrouter_request", _fake_send)
    monkeypatch.setattr(orouter, "_fetch_credits_remaining", lambda cfg, key: 88.5)

    captured: list[dict] = []
    monkeypatch.setattr(
        runner_module.ProgressFormatter,
        "print_summary",
        lambda self, items, summary_file=None, usage=None, credits_remaining=None: captured.append(
            {"items": items, "usage": usage, "credits_remaining": credits_remaining}
        ),
    )

    keyfile = tmp_path / "OpenRouter.api"
    keyfile.write_text("sk-or-fake")
    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(keyfile),
    }

    runner = PipelineRunner(show_progress=True)
    result = runner.run_from_config(config)

    assert result.success
    assert captured, "summary was not displayed"
    call = captured[0]
    assert call["usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert call["credits_remaining"] == 88.5
    # full brief still on disk
    assert "## Caveats" in (tmp_path / "output" / "AI_summary.md").read_text()
