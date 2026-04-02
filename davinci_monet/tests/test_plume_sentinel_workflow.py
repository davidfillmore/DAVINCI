"""Integration test for the PlumeSentinel add-on workflow.

Runs through PipelineRunner.run_from_config() with synthetic GOES-like
data and verifies the full load -> prepare -> plot path produces output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import numpy as np
import pytest
import xarray as xr

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
        assert len(result.completed_stages) == 3

        # Check plots_generated
        plotting_result = result.context.results["plotting"]
        assert plotting_result.data is not None
        plots = plotting_result.data.get("plots_generated", [])
        assert len(plots) >= 1
        assert Path(plots[0]).exists()
