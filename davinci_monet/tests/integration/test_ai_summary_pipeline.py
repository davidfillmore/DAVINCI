"""Integration: AI summary stage runs through PipelineRunner.run_from_config.

The pipeline runs for real on synthetic data; only the Anthropic client is
stubbed so no network call is made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import davinci_monet.ai.summarizer as summarizer_module
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.datasets import create_dataset_dataset
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario, sample_geometry_from

pytestmark = pytest.mark.integration


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

    dataset_ds = create_dataset_dataset(
        variables=["O3"], domain=domain, time_config=time_cfg, seed=42
    )
    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_geometry=10,
        noise_level=0.0,
        seed=42,
    )
    geometry_ds = sample_geometry_from(dataset_ds, "point", scenario=scenario)

    rng = np.random.default_rng(42)
    dataset_ds["O3"] = dataset_ds["O3"] + 5.0 + rng.normal(0, 3.0, size=dataset_ds["O3"].shape)

    dataset_path = tmp_path / "dataset.nc"
    geometry_path = tmp_path / "geometry.nc"
    dataset_ds.to_netcdf(dataset_path)
    geometry_ds.to_netcdf(geometry_path)

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
                "files": str(dataset_path),
                "radius_of_influence": 50000,
                "variables": {"O3": {"units": "ppb"}},
            },
            "surface": {
                "type": "pt_sfc",
                "filename": str(geometry_path),
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
                "title": "O3: Dataset vs Datasets",
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]},
    }


def test_summary_stage_writes_file_through_pipeline(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_module, "_build_client", lambda cfg: stub)

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=False)
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
    result = runner.run_from_config(config)

    # run still succeeds; summary file is absent (stage skipped non-fatally)
    assert result.success, "pipeline run should still succeed without a key"
    assert not (tmp_path / "output" / "AI_summary.md").exists()


def test_summary_displayed_to_terminal_at_end_of_run(monkeypatch, tmp_path: Path) -> None:
    """Terminal gets the itemized bullets; AI_summary.md keeps the full brief."""
    import davinci_monet.pipeline.runner as runner_module
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_module, "_build_client", lambda cfg: stub)

    displayed: list[tuple] = []
    monkeypatch.setattr(
        runner_module.ProgressFormatter,
        "print_summary",
        lambda self, items, summary_file=None, usage=None, credits_remaining=None: displayed.append(
            (items, summary_file)
        ),
    )

    config = _build_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=True)
    result = runner.run_from_config(config)

    assert result.success
    assert displayed, "summary was not displayed at end of run"
    items, summary_file = displayed[0]
    # the display got an itemized list (not the raw full markdown)
    assert isinstance(items, list) and items
    assert summary_file is not None and summary_file.endswith("AI_summary.md")
    # the file on disk still holds the full brief
    assert "## Caveats" in (tmp_path / "output" / "AI_summary.md").read_text()


def _build_two_species_config(tmp_path: Path) -> dict:
    domain = Domain(lon_min=-105.0, lon_max=-95.0, lat_min=35.0, lat_max=45.0, n_lon=12, n_lat=12)
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

    dataset_ds = create_dataset_dataset(
        variables=["O3", "PM25"], domain=domain, time_config=time_cfg, seed=7
    )
    scenario = PerfectMatchScenario(
        variables=["O3", "PM25"],
        domain=domain,
        time_config=time_cfg,
        geometry=DataGeometry.POINT,
        n_geometry=10,
        noise_level=0.0,
        seed=7,
    )
    geometry_ds = sample_geometry_from(dataset_ds, "point", scenario=scenario)

    dataset_path = tmp_path / "dataset2.nc"
    geometry_path = tmp_path / "geometry2.nc"
    dataset_ds.to_netcdf(dataset_path)
    geometry_ds.to_netcdf(geometry_path)

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
                "files": str(dataset_path),
                "radius_of_influence": 50000,
                "variables": {"O3": {"units": "ppb"}, "PM25": {"units": "ug/m3"}},
            },
            "surface": {
                "type": "pt_sfc",
                "filename": str(geometry_path),
                "variables": {"O3": {"units": "ppb"}, "PM25": {"units": "ug/m3"}},
            },
        },
        "pairs": {
            "o3_pair": {
                "x": {"source": "surface", "variable": "O3"},
                "y": {"source": "synthetic", "variable": "O3"},
            },
            "pm_pair": {
                "x": {"source": "surface", "variable": "PM25"},
                "y": {"source": "synthetic", "variable": "PM25"},
            },
        },
        "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
    }


def test_two_species_prompt_carries_distinct_templates(monkeypatch, tmp_path: Path) -> None:
    from davinci_monet.pipeline.runner import PipelineRunner

    stub = _StubClient()
    monkeypatch.setattr(summarizer_module, "_build_client", lambda cfg: stub)

    config = _build_two_species_config(tmp_path)
    config["summary"] = {"enabled": True, "model": "claude-haiku-4-5"}

    runner = PipelineRunner(show_progress=False)
    result = runner.run_from_config(config)

    assert result.success, "pipeline run failed"
    assert stub.calls, "client was not called"
    user_text = stub.calls[0]["messages"][0]["content"][0]["text"]
    assert "## o3_pair — O3" in user_text
    assert "## pm_pair — PM25" in user_text
    assert "Bias and timing" in user_text  # an ozone_eval-only section heading
    assert "Bias and episodes" in user_text  # a pm_eval-only section heading
