"""Pipeline integration tests for DAVINCI.

All tests run through PipelineRunner.run_from_config() with a Python
config dict. This exercises the pipeline core (loading, pairing,
statistics, plotting, saving) but not the CLI or YAML parsing path.
For CLI end-to-end tests, see test_cli_e2e.py. Synthetic data is
written to NetCDF, a config dict is constructed, and the pipeline handles
loading, pairing (or geometry-only detection), statistics, plotting, and saving.

Three test classes cover different workflow types:
  - TestPointPipeline: paired point (surface) — 8 plot types
  - TestTrackPipeline: paired track (aircraft) — curtain, track_map_3d, flight_timeseries
  - TestGeometryOnlyPipeline: geometry-only (aircraft) — 4 geometry plot types

CI artifacts (plots, stats CSV, pipeline log) are copied to
CI_ARTIFACTS_DIR when set by GitHub Actions.
"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario, sample_geometry_from

# =============================================================================
# Helpers
# =============================================================================


def _copy_artifacts(output_dir: Path, log_dir: Path, prefix: str = "") -> None:
    """Copy pipeline outputs to CI artifacts directory if enabled."""
    ci_dir = os.environ.get("CI_ARTIFACTS_DIR")
    if not ci_dir:
        return
    artifacts = Path(ci_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    for f in sorted(output_dir.rglob("*.png")):
        shutil.copy2(f, artifacts / f"{prefix}{f.name}")
    for f in sorted(output_dir.rglob("*.csv")):
        shutil.copy2(f, artifacts / f"{prefix}{f.name}")
    for f in sorted(log_dir.glob("pipeline_*.md")):
        shutil.copy2(f, artifacts / f"{prefix}{f.name}")


def _assert_pipeline_success(result: object) -> None:
    """Assert pipeline succeeded with useful error message."""
    assert result.success, (  # type: ignore[attr-defined]
        f"Pipeline failed. Failed stages: "
        f"{[s.stage_name + ': ' + str(s.error) for s in result.failed_stages]}"  # type: ignore[attr-defined]
    )


def _assert_plots(output_dir: Path, min_count: int) -> list[Path]:
    """Assert expected number of PNG plots, each >1KB."""
    png_files = sorted(output_dir.rglob("*.png"))
    assert len(png_files) >= min_count, (
        f"Expected at least {min_count} PNG plots, "
        f"got {len(png_files)}: {[f.name for f in png_files]}"
    )
    for png in png_files:
        assert (
            png.stat().st_size > 1024
        ), f"Plot {png.name} is too small ({png.stat().st_size} bytes)"
    return png_files


# =============================================================================
# 1. Paired Point Pipeline (surface stations)
# =============================================================================


@pytest.mark.integration
class TestPointPipeline:
    """End-to-end paired pipeline: dataset + surface geometry → 8 plot types."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.datasets import create_dataset_dataset

        domain = Domain(
            lon_min=-105.0,
            lon_max=-95.0,
            lat_min=35.0,
            lat_max=45.0,
            n_lon=12,
            n_lat=12,
        )
        time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

        # Build dataset with latitude gradient
        y_ds = create_dataset_dataset(
            variables=["O3"], domain=domain, time_config=time_cfg, seed=42
        )
        lat_vals = y_ds.lat.values
        lat_norm = (lat_vals - lat_vals.min()) / (lat_vals.max() - lat_vals.min())
        y_ds["O3"] = y_ds["O3"] + 20.0 * lat_norm[:, np.newaxis]

        # Sample geometry from gradient-enhanced dataset
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

        # Add dataset bias + noise (geometry stay clean)
        rng = np.random.default_rng(42)
        lon_vals = y_ds.lon.values
        lon_norm = (lon_vals - lon_vals.min()) / (lon_vals.max() - lon_vals.min())
        y_ds["O3"] = (
            y_ds["O3"]
            + 5.0
            + 6.0 * lon_norm[np.newaxis, :]
            + rng.normal(0, 3.0, size=y_ds["O3"].shape)
        )

        # Write to NetCDF
        y_path = tmp_path / "dataset.nc"
        x_path = tmp_path / "geometry.nc"
        y_ds.to_netcdf(y_path)
        x_ds.to_netcdf(x_path)

        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        config = {
            "analysis": {
                "start_time": "2024-01-15 00:00",
                "end_time": "2024-01-17 00:00",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "sources": {
                "synthetic": {
                    "type": "generic",
                    "files": str(y_path),
                    "radius_of_influence": 50000,
                    "variables": {
                        "O3": {
                            "units": "ppb",
                            "vmin_plot": 30,
                            "vmax_plot": 70,
                            "vdiff_plot": 10,
                        },
                    },
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
                "spatial_bias_o3": {
                    "type": "spatial_bias",
                    "pairs": ["synthetic_surface"],
                    "title": "O3 Spatial Bias",
                },
                "spatial_dist_o3": {
                    "type": "spatial_distribution",
                    "pairs": ["synthetic_surface"],
                    "title": "O3 Distribution",
                    "show_var": "x",
                },
                "scorecard_o3": {
                    "type": "scorecard",
                    "pairs": ["synthetic_surface"],
                    "title": "O3 Scorecard",
                },
            },
            "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME", "IOA"]},
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
        _assert_pipeline_success(result)

        # Verify stats
        stats_files = list(output_dir.rglob("*.csv"))
        assert len(stats_files) > 0, "No statistics CSV files produced"

        # Verify plots
        _assert_plots(output_dir, min_count=8)

        # Verify log
        assert list(log_dir.glob("pipeline_*.md")), "No pipeline log"

        _copy_artifacts(output_dir, log_dir, prefix="point_")


# =============================================================================
# 2. Paired Track Pipeline (aircraft)
# =============================================================================


@pytest.mark.integration
class TestTrackPipeline:
    """End-to-end paired pipeline: dataset + aircraft track → 3 plot types."""

    def test_track_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.datasets import create_dataset_dataset

        domain = Domain(
            lon_min=-105.0,
            lon_max=-95.0,
            lat_min=35.0,
            lat_max=42.0,
            n_lon=12,
            n_lat=10,
        )
        time_cfg = TimeConfig(start="2024-01-15 14:00", end="2024-01-15 17:00", freq="1h")
        rng = np.random.default_rng(42)
        n = 200

        # Build 2D dataset (surface only — track strategy falls back to surface extraction)
        y_ds = create_dataset_dataset(
            variables=["O3"], domain=domain, time_config=time_cfg, seed=42
        )
        y_ds["O3"] = y_ds["O3"] + rng.normal(0, 2.0, size=y_ds["O3"].shape)

        # Build synthetic track geometry
        t = np.linspace(0, 4 * np.pi, n)
        times = np.datetime64("2024-01-15T14:00") + np.arange(n) * np.timedelta64(30, "s")
        lats = 38.0 + 3.0 * np.sin(t / 2)
        lons = -100.0 + 4.0 * np.cos(t / 3)
        alts = 1000 + 8000 * (0.5 + 0.5 * np.sin(t / 2))
        flight_ids = np.where(np.arange(n) < 100, "F01", "F02")

        x_ds = xr.Dataset(
            {"O3": ("time", 30.0 + 5.0 * (alts / 1000) + rng.normal(0, 3, n))},
            coords={
                "time": times,
                "latitude": ("time", lats),
                "longitude": ("time", lons),
                "altitude": ("time", alts),
                "flight": ("time", flight_ids),
            },
            attrs={"geometry": "track"},
        )

        y_path = tmp_path / "dataset_track.nc"
        x_path = tmp_path / "geometry_track.nc"
        y_ds.to_netcdf(y_path)
        x_ds.to_netcdf(x_path)

        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        config = {
            "analysis": {
                "start_time": "2024-01-15 14:00",
                "end_time": "2024-01-15 17:00",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "sources": {
                "synthetic": {
                    "type": "generic",
                    "files": str(y_path),
                    "radius_of_influence": 100000,
                    "variables": {"O3": {"units": "ppb"}},
                },
                "aircraft": {
                    "type": "aircraft",
                    "filename": str(x_path),
                    "variables": {"O3": {"units": "ppb"}},
                },
            },
            "pairs": {
                "synthetic_aircraft": {
                    "x": {"source": "aircraft", "variable": "O3"},
                    "y": {"source": "synthetic", "variable": "O3"},
                },
            },
            "plots": {
                "curtain_o3": {
                    "type": "curtain",
                    "pairs": ["synthetic_aircraft"],
                    "title": "O3 Curtain",
                },
                "track_3d_o3": {
                    "type": "track_map_3d",
                    "pairs": ["synthetic_aircraft"],
                    "title": "O3 3D Track",
                    "show_var": "x",
                    "show_coastlines": False,
                },
                "flight_ts_o3": {
                    "type": "flight_timeseries",
                    "pairs": ["synthetic_aircraft"],
                    "title": "O3 Flight Time Series",
                },
            },
            "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
        _assert_pipeline_success(result)
        _assert_plots(output_dir, min_count=3)
        assert list(log_dir.glob("pipeline_*.md")), "No pipeline log"
        _copy_artifacts(output_dir, log_dir, prefix="track_")


# =============================================================================
# 3. Geometry-Only Pipeline (no dataset)
# =============================================================================


@pytest.mark.integration
class TestGeometryOnlyPipeline:
    """End-to-end geometry-only pipeline: aircraft data → 4 geometry plot types."""

    def test_geometry_only_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner

        rng = np.random.default_rng(99)
        n = 300

        # Build synthetic aircraft geometry
        t = np.linspace(0, 3 * np.pi, n)
        times = np.datetime64("2012-05-29T14:00") + np.arange(n) * np.timedelta64(20, "s")
        lats = 37.0 + 2.0 * np.sin(t)
        lons = -97.0 + 3.0 * np.cos(t / 2)
        alts = 500 + 10000 * (0.5 + 0.5 * np.sin(t))
        flight_ids = np.where(np.arange(n) < 150, "2012-05-29", "2012-05-30")

        x_ds = xr.Dataset(
            {
                "O3": (
                    "time",
                    30.0 + 6.0 * (alts / 1000) + rng.normal(0, 5, n),
                    {"units": "ppbv", "long_name": "Ozone"},
                ),
                "CO": (
                    "time",
                    80.0 + 20.0 * np.exp(-alts / 5000) + rng.normal(0, 10, n),
                    {"units": "ppbv", "long_name": "Carbon Monoxide"},
                ),
            },
            coords={
                "time": times,
                "latitude": ("time", lats),
                "longitude": ("time", lons),
                "altitude": ("time", alts),
                "flight": ("time", flight_ids),
            },
            attrs={"geometry": "track"},
        )

        x_path = tmp_path / "geometry_aircraft.nc"
        x_ds.to_netcdf(x_path)

        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        # A single geometry source with no pairs triggers the geometry-only pipeline.
        config = {
            "analysis": {
                "start_time": "2012-05-29",
                "end_time": "2012-05-31",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "sources": {
                "dc8": {
                    "type": "aircraft",
                    "filename": str(x_path),
                    "variables": {
                        "O3": {"units": "ppbv"},
                        "CO": {"units": "ppbv"},
                    },
                },
            },
            "plots": {
                "geometry_ts_o3": {
                    "type": "timeseries",
                    "source": "dc8",
                    "variable": "O3",
                    "title": "O3 Time Series",
                },
                "geometry_hist_o3": {
                    "type": "histogram",
                    "source": "dc8",
                    "variable": "O3",
                    "title": "O3 Distribution",
                },
                "geometry_profile_o3": {
                    "type": "vertical_profile",
                    "source": "dc8",
                    "variable": "O3",
                    "title": "O3 Vertical Profile",
                },
                "geometry_track_o3": {
                    "type": "flight_track",
                    "source": "dc8",
                    "variable": "O3",
                    "title": "O3 Flight Track",
                },
            },
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
        _assert_pipeline_success(result)
        _assert_plots(output_dir, min_count=4)
        assert list(log_dir.glob("pipeline_*.md")), "No pipeline log"
        _copy_artifacts(output_dir, log_dir, prefix="geometryonly_")
