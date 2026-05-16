"""Integration test for the PlumeSentinel add-on workflow.

Runs through PipelineRunner.run_from_config() with synthetic GOES-like
data and verifies the full load -> prepare -> plot path produces output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pytest
import xarray as xr

from davinci_monet.addons.plume_sentinel import workflow as ps_workflow
from davinci_monet.pipeline.runner import PipelineRunner


class TestPlumeSentinelWorkflow:
    def _create_synthetic_goes(self, path: Path) -> None:
        ds = xr.Dataset(
            {
                "CMI_C01": (["y", "x"], np.random.rand(20, 20).astype(np.float32)),
                "CMI_C02": (["y", "x"], np.random.rand(20, 20).astype(np.float32)),
                "CMI_C03": (["y", "x"], np.random.rand(20, 20).astype(np.float32)),
                "goes_imager_projection": (
                    [],
                    0,
                    {
                        "perspective_point_height": 35786023.0,
                        "longitude_of_projection_origin": -75.0,
                        "sweep_angle_axis": "x",
                        "semi_major_axis": 6378137.0,
                        "semi_minor_axis": 6356752.31414,
                    },
                ),
            },
            coords={
                "x": np.linspace(-0.1, 0.1, 20),
                "y": np.linspace(-0.1, 0.1, 20),
            },
        )
        ds.to_netcdf(path)

    def test_goes_only_workflow(self, tmp_path):
        """Minimal workflow with GOES input only (no HMS overlay)."""
        goes_path = tmp_path / "goes_test.nc"
        self._create_synthetic_goes(goes_path)
        output_dir = tmp_path / "output"

        config = {
            "analysis": {
                "workflow": "plume_sentinel",
                "output_dir": str(output_dir),
                "style": {"theme": "ncar", "context": "default"},
            },
            "plume_sentinel": {
                "inputs": {
                    "goes_event": {
                        "type": "goes_truecolor",
                        "file": str(goes_path),
                        "gamma": 1.8,
                    },
                },
                "plots": {
                    "test_goes": {
                        "type": "truecolor_contour",
                        "background": "goes_event",
                        "overlays": [],
                        "extent": [-130, -110, 30, 52],
                        "projection": {
                            "type": "lambert_conformal",
                            "central_longitude": -120,
                        },
                        "title": "Test GOES Plot",
                    },
                },
            },
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)

        assert result.success, (
            f"Pipeline failed: {[s.error for s in result.stage_results if s.error]}"
        )
        assert len(result.completed_stages) == 4

        # Check plots_generated
        plotting_result = result.context.results["plotting"]
        assert plotting_result.data is not None
        plots = plotting_result.data.get("plots_generated", [])
        assert len(plots) >= 1
        assert Path(plots[0]).exists()

    def test_workflow_emits_metrics_json(self, tmp_path):
        """`workflow.run` with `emit_metrics_json` writes a v1-conformant sidecar."""
        goes_path = tmp_path / "goes_test.nc"
        self._create_synthetic_goes(goes_path)
        output_dir = tmp_path / "output"
        metrics_path = tmp_path / "metrics.json"

        config = {
            "analysis": {
                "workflow": "plume_sentinel",
                "start_time": "2020-09-09",
                "end_time": "2020-09-09",
                "output_dir": str(output_dir),
                "style": {"theme": "ncar", "context": "default"},
            },
            "plume_sentinel": {
                "inputs": {
                    "goes_event": {
                        "type": "goes_truecolor",
                        "file": str(goes_path),
                        "gamma": 1.8,
                    },
                },
                "plots": {
                    "test_goes": {
                        "type": "truecolor_contour",
                        "background": "goes_event",
                        "overlays": [],
                        "extent": [-130, -110, 30, 52],
                        "projection": {
                            "type": "lambert_conformal",
                            "central_longitude": -120,
                        },
                        "title": "Test GOES Plot",
                    },
                },
            },
        }

        result = ps_workflow.run(
            config,
            emit_metrics_json=metrics_path,
            run_id="test-westcoast-modisaod",
            region="westcoast",
            config_slug="modisaod",
        )

        assert result.success, (
            f"Pipeline failed: {[s.error for s in result.stage_results if s.error]}"
        )
        assert metrics_path.exists()

        payload = json.loads(metrics_path.read_text())
        assert payload["schema"] == "plumesentinel.metrics.v1"
        assert payload["run_id"] == "test-westcoast-modisaod"
        assert payload["region"] == "westcoast"
        assert payload["config_slug"] == "modisaod"
        assert "metrics" in payload
        assert "pipeline_version" in payload
        assert "davinci_monet" in payload["pipeline_version"]
        assert "plume_sentinel_addon" in payload["pipeline_version"]
        assert "input_datasets" in payload
        assert "quality_flags" in payload
        assert "plot_urls" in payload
        assert "wallclock_s" in payload
        assert "produced_at" in payload
        assert "valid_time" in payload
        assert "config_files" in payload

    def test_workflow_event_date_override(self, tmp_path):
        """`--event-date` overrides config start/end_time and propagates to metrics.json.

        The fixture config declares start_time/end_time = 2020-09-09; we invoke
        ``workflow.run`` with ``event_date='2020-09-10'`` and verify the emitted
        metrics payload reflects the overridden date in ``event_date``,
        ``valid_time``, and any config-derived ``input_datasets[*].valid_time``.
        """
        goes_path = tmp_path / "goes_test.nc"
        self._create_synthetic_goes(goes_path)
        output_dir = tmp_path / "output"
        metrics_path = tmp_path / "metrics.json"

        config = {
            "analysis": {
                "workflow": "plume_sentinel",
                "start_time": "2020-09-09",
                "end_time": "2020-09-09",
                "output_dir": str(output_dir),
                "style": {"theme": "ncar", "context": "default"},
            },
            "plume_sentinel": {
                "inputs": {
                    "goes_event": {
                        "type": "goes_truecolor",
                        "file": str(goes_path),
                        "gamma": 1.8,
                    },
                },
                "plots": {
                    "test_goes": {
                        "type": "truecolor_contour",
                        "background": "goes_event",
                        "overlays": [],
                        "extent": [-130, -110, 30, 52],
                        "projection": {
                            "type": "lambert_conformal",
                            "central_longitude": -120,
                        },
                        "title": "Test GOES Plot",
                    },
                },
            },
        }

        result = ps_workflow.run(
            config,
            emit_metrics_json=metrics_path,
            run_id="test-westcoast-modisaod-override",
            region="westcoast",
            config_slug="modisaod",
            event_date="2020-09-10",
        )

        assert result.success, (
            f"Pipeline failed: {[s.error for s in result.stage_results if s.error]}"
        )
        assert metrics_path.exists()

        payload = json.loads(metrics_path.read_text())
        assert payload["event_date"] == "2020-09-10", payload["event_date"]
        assert payload["valid_time"].startswith("2020-09-10"), payload["valid_time"]
        # Any config-derived input_datasets should also carry the overridden valid_time
        # (loader-provided structured provenance, when present, is allowed to differ).
        for ds in payload.get("input_datasets", []):
            assert ds["valid_time"].startswith("2020-09-10"), (
                f"input_dataset {ds.get('name')} valid_time = {ds.get('valid_time')!r}"
            )
