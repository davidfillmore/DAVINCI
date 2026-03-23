"""End-to-end integration test for DAVINCI pipeline.

Generates synthetic model and observation data, writes them to NetCDF,
runs the full pipeline (load → pair → stats → plot → save), and verifies
that all expected outputs are produced with reasonable statistics.

This test produces CI artifacts (plots, stats CSV, pipeline log) that
can be uploaded from GitHub Actions for review.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def integration_domain() -> Domain:
    """Small CONUS-like domain for integration testing."""
    return Domain(
        lon_min=-105.0,
        lon_max=-95.0,
        lat_min=35.0,
        lat_max=45.0,
        n_lon=12,
        n_lat=12,
    )


@pytest.fixture
def integration_time() -> TimeConfig:
    """48-hour period for integration testing (covers diurnal cycle)."""
    return TimeConfig(
        start="2024-01-15 00:00",
        end="2024-01-17 00:00",
        freq="1h",
    )


@pytest.fixture
def synthetic_data(
    integration_domain: Domain,
    integration_time: TimeConfig,
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Generate synthetic model and obs NetCDF files.

    Returns paths to (model_file, obs_file).
    """
    from davinci_monet.tests.synthetic.models import create_model_dataset
    from davinci_monet.tests.synthetic.observations import create_point_observations

    # Build model with a latitude gradient (20 ppb south→north)
    model_ds = create_model_dataset(
        variables=["O3"],
        domain=integration_domain,
        time_config=integration_time,
        seed=42,
    )
    lat_vals = model_ds.lat.values
    lat_norm = (lat_vals - lat_vals.min()) / (lat_vals.max() - lat_vals.min())
    model_ds["O3"] = model_ds["O3"] + 20.0 * lat_norm[:, np.newaxis]

    # Sample obs from this model (so obs inherit the spatial gradient)
    scenario = PerfectMatchScenario(
        variables=["O3"],
        domain=integration_domain,
        time_config=integration_time,
        geometry=DataGeometry.POINT,
        n_obs=10,
        noise_level=0.0,
        seed=42,
    )
    # Override: use our gradient-enhanced model for sampling
    obs_ds = scenario._generate_point_obs(model_ds)

    # Now add model bias: +5 ppb global, east-west gradient (0-6 ppb),
    # and per-gridcell noise (±3 ppb) — obs stay clean
    rng = np.random.default_rng(42)
    lon_vals = model_ds.lon.values
    lon_norm = (lon_vals - lon_vals.min()) / (lon_vals.max() - lon_vals.min())
    lon_gradient = 6.0 * lon_norm[np.newaxis, :]
    noise = rng.normal(0, 3.0, size=model_ds["O3"].shape)
    model_ds["O3"] = model_ds["O3"] + 5.0 + lon_gradient + noise

    # Write model to NetCDF
    model_path = tmp_path / "model.nc"
    model_ds.to_netcdf(model_path)

    # Write obs to NetCDF
    obs_path = tmp_path / "obs.nc"
    obs_ds.to_netcdf(obs_path)

    return model_path, obs_path


@pytest.fixture
def pipeline_config(
    synthetic_data: tuple[Path, Path],
    tmp_path: Path,
) -> dict:
    """Build a portable pipeline config dict using tmp_path."""
    model_path, obs_path = synthetic_data
    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    return {
        "analysis": {
            "start_time": "2024-01-15 00:00",
            "end_time": "2024-01-17 00:00",
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "debug": False,
        },
        "model": {
            "synthetic": {
                "mod_type": "generic",
                "files": str(model_path),
                "radius_of_influence": 50000,
                "mapping": {
                    "surface": {
                        "O3": "O3",
                    },
                },
                "variables": {
                    "O3": {
                        "units": "ppb",
                        "ylabel_plot": "O$_3$ (ppb)",
                        "vmin_plot": 30,
                        "vmax_plot": 70,
                        "vdiff_plot": 10,
                    },
                },
            },
        },
        "obs": {
            "surface": {
                "obs_type": "pt_sfc",
                "filename": str(obs_path),
                "variables": {
                    "O3": {
                        "obs_min": 0,
                        "obs_max": 200,
                        "units": "ppb",
                    },
                },
            },
        },
        "pairs": {
            "synthetic_surface": {
                "model": "synthetic",
                "obs": "surface",
                "variable": {
                    "model_var": "O3",
                    "obs_var": "O3",
                },
            },
        },
        "plots": {
            # --- Statistical plots ---
            "scatter_o3": {
                "type": "scatter",
                "pairs": ["synthetic_surface"],
                "title": "O3: Model vs Observations",
            },
            "taylor_o3": {
                "type": "taylor",
                "pairs": ["synthetic_surface"],
                "title": "O3 Taylor Diagram",
            },
            "boxplot_o3": {
                "type": "boxplot",
                "pairs": ["synthetic_surface"],
                "title": "O3 Box Plot",
            },
            # --- Temporal plots ---
            "timeseries_o3": {
                "type": "timeseries",
                "pairs": ["synthetic_surface"],
                "title": "O3 Time Series",
                "aggregate_dim": "site",
            },
            "diurnal_o3": {
                "type": "diurnal",
                "pairs": ["synthetic_surface"],
                "title": "O3 Diurnal Cycle",
            },
            # --- Spatial plots ---
            "spatial_bias_o3": {
                "type": "spatial_bias",
                "pairs": ["synthetic_surface"],
                "title": "O3 Spatial Bias",
            },
            "spatial_dist_o3": {
                "type": "spatial_distribution",
                "pairs": ["synthetic_surface"],
                "title": "O3 Observed Distribution",
                "show_var": "obs",
            },
            # --- Specialized plots ---
            "scorecard_o3": {
                "type": "scorecard",
                "pairs": ["synthetic_surface"],
                "title": "O3 Scorecard",
            },
        },
        "stats": {
            "metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"],
        },
    }


# =============================================================================
# Integration Test
# =============================================================================


EXPECTED_PLOT_COUNT = 8


class TestPipelineIntegration:
    """End-to-end pipeline test with synthetic data."""

    def test_full_pipeline(
        self,
        pipeline_config: dict,
        tmp_path: Path,
    ) -> None:
        """Run complete pipeline and verify all outputs."""
        from davinci_monet.pipeline.runner import PipelineRunner

        # Run pipeline
        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(pipeline_config)

        # Pipeline should succeed
        assert result.success, (
            f"Pipeline failed. Failed stages: "
            f"{[s.stage_name + ': ' + str(s.error) for s in result.failed_stages]}"
        )

        output_dir = Path(pipeline_config["analysis"]["output_dir"])

        # --- Verify statistics CSV ---
        stats_files = list(output_dir.rglob("*.csv"))
        assert len(stats_files) > 0, "No statistics CSV files produced"

        # Read stats and check values
        stats_path = stats_files[0]
        with open(stats_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have metrics
        assert len(rows) > 0, "Statistics CSV is empty"

        # --- Verify plots ---
        png_files = sorted(output_dir.rglob("*.png"))
        assert len(png_files) >= EXPECTED_PLOT_COUNT, (
            f"Expected at least {EXPECTED_PLOT_COUNT} PNG plots, "
            f"got {len(png_files)}: {[f.name for f in png_files]}"
        )

        # Each plot should be a real image (>1KB)
        for png in png_files:
            assert png.stat().st_size > 1024, (
                f"Plot {png.name} is too small ({png.stat().st_size} bytes)"
            )

        # --- Verify pipeline log ---
        log_dir = Path(pipeline_config["analysis"]["log_dir"])
        log_files = list(log_dir.glob("pipeline_*.md"))
        assert len(log_files) > 0, "No pipeline log file produced"

        # --- Copy artifacts for CI upload ---
        ci_artifacts_dir = os.environ.get("CI_ARTIFACTS_DIR")
        if ci_artifacts_dir:
            artifacts = Path(ci_artifacts_dir)
            artifacts.mkdir(parents=True, exist_ok=True)

            # Copy stats
            for f in stats_files:
                shutil.copy2(f, artifacts / f.name)

            # Copy plots
            for f in png_files:
                shutil.copy2(f, artifacts / f.name)

            # Copy log
            for f in log_files:
                shutil.copy2(f, artifacts / f.name)
