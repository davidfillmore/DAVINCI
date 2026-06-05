"""Integration: AI summary stage runs through PipelineRunner.run_from_config.

The pipeline runs for real on synthetic data; only the Anthropic client is
stubbed so no network call is made.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import davinci_monet.ai.summarizer as summarizer_mod
from davinci_monet.config.migration import LegacyConfigWarning
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


class _StubClient:
    """Returns a fixed markdown brief; records calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        outer = self

        class _Msgs:
            def create(self, **kwargs):
                outer.calls.append(kwargs)

                class _Block:
                    text = (
                        "## What this run is\nSynthetic O3 run.\n"
                        "## Headline metrics\nN counted.\n"
                        "## Interpretation\nClose match.\n"
                        "## Caveats\nSynthetic data.\n"
                    )

                class _Usage:
                    input_tokens = 100
                    output_tokens = 50

                class _Resp:
                    content = [_Block()]
                    usage = _Usage()
                    model = kwargs["model"]

                return _Resp()

        self.messages = _Msgs()


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


def test_summary_stage_writes_file_through_pipeline(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_mod, "_build_client", lambda cfg: stub)

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success, "pipeline run failed"
    summary_file = tmp_path / "output" / "AI_summary.md"
    assert summary_file.exists(), "AI_summary.md was not written"
    assert "## Caveats" in summary_file.read_text()
    # the stubbed client actually received the scatter plot image
    assert stub.calls, "Anthropic client was not called"
    user_content = stub.calls[0]["messages"][0]["content"]
    assert any(block["type"] == "image" for block in user_content)


def test_summary_stage_skips_without_api_key(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True}

    runner = PipelineRunner(show_progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    # run still succeeds; summary file is absent (stage skipped non-fatally)
    assert result.success, "pipeline run should still succeed without a key"
    assert not (tmp_path / "output" / "AI_summary.md").exists()


def test_summary_displayed_to_terminal_at_end_of_run(monkeypatch, tmp_path: Path) -> None:
    """Per spec, the brief must be printed to the terminal (not just the file)."""
    import davinci_monet.pipeline.runner as runner_mod
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_mod, "_build_client", lambda cfg: stub)

    displayed: list[str] = []
    monkeypatch.setattr(
        runner_mod.ProgressFormatter,
        "print_summary",
        lambda self, markdown: displayed.append(markdown),
    )

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LegacyConfigWarning)
        result = runner.run_from_config(config)

    assert result.success
    assert displayed, "summary brief was not displayed at end of run"
    assert "## Caveats" in displayed[0]
