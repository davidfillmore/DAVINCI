"""Integration: OpenRouter summary path through PipelineRunner.run_from_config.

The pipeline runs for real on synthetic data; only the OpenRouter HTTP send is
stubbed so no network call is made.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import davinci_monet.ai.openrouter as orouter
from davinci_monet.config.migration import LegacyConfigWarning
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


def _build_config(tmp_path: Path) -> dict:
    domain = Domain(lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12)
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

    model_ds = create_model_dataset(variables=["O3"], domain=domain, time_config=time_cfg, seed=42)
    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_obs=10,
        noise_level=0.0,
        seed=42,
    )
    obs_ds = scenario._generate_point_obs(model_ds)

    rng = np.random.default_rng(42)
    model_ds["O3"] = model_ds["O3"] + 5.0 + rng.normal(0, 3.0, size=model_ds["O3"].shape)

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    model_ds.to_netcdf(model_path)
    obs_ds.to_netcdf(obs_path)

    return {
        "analysis": {
            "start_time": "2024-01-15 00:00",
            "end_time": "2024-01-17 00:00",
            "output_dir": str(tmp_path / "output"),
            "log_dir": str(tmp_path / "logs"),
        },
        "model": {
            "synthetic": {
                "mod_type": "generic",
                "files": str(model_path),
                "radius_of_influence": 50000,
                "mapping": {"surface": {"O3": "O3"}},
                "variables": {"O3": {"units": "ppb"}},
            },
        },
        "obs": {
            "surface": {
                "obs_type": "pt_sfc",
                "filename": str(obs_path),
                "variables": {"O3": {"obs_min": 0, "obs_max": 200, "units": "ppb"}},
            },
        },
        "pairs": {
            "synthetic_surface": {
                "model": "synthetic",
                "obs": "surface",
                "variable": {"model_var": "O3", "obs_var": "O3"},
            },
        },
        "plots": {
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Model vs Observations",
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

    keyfile = tmp_path / "OpenRouter.api"
    keyfile.write_text("sk-or-fake")

    config = _build_config(tmp_path)
    config["summary"] = {
        "enabled": True,
        "provider": "openrouter",
        "api_key_file": str(keyfile),
    }

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success  # non-fatal: run still succeeds
    assert not (tmp_path / "output" / "AI_summary.md").exists()
