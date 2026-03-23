"""Pipeline integration tests for DAVINCI.

All tests run through PipelineRunner.run_from_config() with a Python
config dict. This exercises the pipeline core (loading, pairing,
statistics, plotting, saving) but not the CLI or YAML parsing path.
For CLI end-to-end tests, see test_cli_e2e.py. Synthetic data is
written to NetCDF, a config dict is constructed, and the pipeline handles
loading, pairing (or obs-only detection), statistics, plotting, and saving.

Three test classes cover different workflow types:
  - TestPointPipeline: paired point (surface) — 8 plot types
  - TestTrackPipeline: paired track (aircraft) — curtain, track_map_3d, flight_timeseries
  - TestObsOnlyPipeline: obs-only (aircraft) — 4 obs plot types

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
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


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
    assert result.success, (
        f"Pipeline failed. Failed stages: "
        f"{[s.stage_name + ': ' + str(s.error) for s in result.failed_stages]}"
    )


def _assert_plots(output_dir: Path, min_count: int) -> list[Path]:
    """Assert expected number of PNG plots, each >1KB."""
    png_files = sorted(output_dir.rglob("*.png"))
    assert len(png_files) >= min_count, (
        f"Expected at least {min_count} PNG plots, "
        f"got {len(png_files)}: {[f.name for f in png_files]}"
    )
    for png in png_files:
        assert png.stat().st_size > 1024, (
            f"Plot {png.name} is too small ({png.stat().st_size} bytes)"
        )
    return png_files


# =============================================================================
# 1. Paired Point Pipeline (surface stations)
# =============================================================================


class TestPointPipeline:
    """End-to-end paired pipeline: model + surface obs → 8 plot types."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.models import create_model_dataset

        domain = Domain(
            lon_min=-105.0, lon_max=-95.0,
            lat_min=35.0, lat_max=45.0,
            n_lon=12, n_lat=12,
        )
        time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-17 00:00", freq="1h")

        # Build model with latitude gradient
        model_ds = create_model_dataset(variables=["O3"], domain=domain, time_config=time_cfg, seed=42)
        lat_vals = model_ds.lat.values
        lat_norm = (lat_vals - lat_vals.min()) / (lat_vals.max() - lat_vals.min())
        model_ds["O3"] = model_ds["O3"] + 20.0 * lat_norm[:, np.newaxis]

        # Sample obs from gradient-enhanced model
        scenario = PerfectMatchScenario(
            variables=["O3"], domain=domain, time_config=time_cfg,
            geometry=DataGeometry.POINT, n_obs=10, noise_level=0.0, seed=42,
        )
        obs_ds = scenario._generate_point_obs(model_ds)

        # Add model bias + noise (obs stay clean)
        rng = np.random.default_rng(42)
        lon_vals = model_ds.lon.values
        lon_norm = (lon_vals - lon_vals.min()) / (lon_vals.max() - lon_vals.min())
        model_ds["O3"] = (
            model_ds["O3"] + 5.0
            + 6.0 * lon_norm[np.newaxis, :]
            + rng.normal(0, 3.0, size=model_ds["O3"].shape)
        )

        # Write to NetCDF
        model_path = tmp_path / "model.nc"
        obs_path = tmp_path / "obs.nc"
        model_ds.to_netcdf(model_path)
        obs_ds.to_netcdf(obs_path)

        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        config = {
            "analysis": {
                "start_time": "2024-01-15 00:00",
                "end_time": "2024-01-17 00:00",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "model": {
                "synthetic": {
                    "mod_type": "generic",
                    "files": str(model_path),
                    "radius_of_influence": 50000,
                    "mapping": {"surface": {"O3": "O3"}},
                    "variables": {
                        "O3": {
                            "units": "ppb",
                            "vmin_plot": 30, "vmax_plot": 70, "vdiff_plot": 10,
                        },
                    },
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
                    "model": "synthetic", "obs": "surface",
                    "variable": {"model_var": "O3", "obs_var": "O3"},
                },
            },
            "plots": {
                "scatter_o3": {"type": "scatter", "pairs": ["synthetic_surface"], "title": "O3: Model vs Observations"},
                "taylor_o3": {"type": "taylor", "pairs": ["synthetic_surface"], "title": "O3 Taylor Diagram"},
                "boxplot_o3": {"type": "boxplot", "pairs": ["synthetic_surface"], "title": "O3 Box Plot"},
                "timeseries_o3": {"type": "timeseries", "pairs": ["synthetic_surface"], "title": "O3 Time Series", "aggregate_dim": "site"},
                "diurnal_o3": {"type": "diurnal", "pairs": ["synthetic_surface"], "title": "O3 Diurnal Cycle"},
                "spatial_bias_o3": {"type": "spatial_bias", "pairs": ["synthetic_surface"], "title": "O3 Spatial Bias"},
                "spatial_dist_o3": {"type": "spatial_distribution", "pairs": ["synthetic_surface"], "title": "O3 Observed Distribution", "show_var": "obs"},
                "scorecard_o3": {"type": "scorecard", "pairs": ["synthetic_surface"], "title": "O3 Scorecard"},
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


class TestTrackPipeline:
    """End-to-end paired pipeline: model + aircraft track → 3 plot types."""

    def test_track_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.models import create_model_dataset

        domain = Domain(
            lon_min=-105.0, lon_max=-95.0,
            lat_min=35.0, lat_max=42.0,
            n_lon=12, n_lat=10,
        )
        time_cfg = TimeConfig(start="2024-01-15 14:00", end="2024-01-15 17:00", freq="1h")
        rng = np.random.default_rng(42)
        n = 200

        # Build 2D model (surface only — track strategy falls back to surface extraction)
        model_ds = create_model_dataset(variables=["O3"], domain=domain, time_config=time_cfg, seed=42)
        model_ds["O3"] = model_ds["O3"] + rng.normal(0, 2.0, size=model_ds["O3"].shape)

        # Build synthetic track obs
        t = np.linspace(0, 4 * np.pi, n)
        times = np.datetime64("2024-01-15T14:00") + np.arange(n) * np.timedelta64(30, "s")
        lats = 38.0 + 3.0 * np.sin(t / 2)
        lons = -100.0 + 4.0 * np.cos(t / 3)
        alts = 1000 + 8000 * (0.5 + 0.5 * np.sin(t / 2))
        flight_ids = np.where(np.arange(n) < 100, "F01", "F02")

        obs_ds = xr.Dataset(
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

        model_path = tmp_path / "model_track.nc"
        obs_path = tmp_path / "obs_track.nc"
        model_ds.to_netcdf(model_path)
        obs_ds.to_netcdf(obs_path)

        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        config = {
            "analysis": {
                "start_time": "2024-01-15 14:00",
                "end_time": "2024-01-15 17:00",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "model": {
                "synthetic": {
                    "mod_type": "generic",
                    "files": str(model_path),
                    "radius_of_influence": 100000,
                    "mapping": {"aircraft": {"O3": "O3"}},
                    "variables": {"O3": {"units": "ppb"}},
                },
            },
            "obs": {
                "aircraft": {
                    "obs_type": "aircraft",
                    "filename": str(obs_path),
                    "variables": {"O3": {"units": "ppb"}},
                },
            },
            "pairs": {
                "synthetic_aircraft": {
                    "model": "synthetic", "obs": "aircraft",
                    "variable": {"model_var": "O3", "obs_var": "O3"},
                },
            },
            "plots": {
                "curtain_o3": {"type": "curtain", "pairs": ["synthetic_aircraft"], "title": "O3 Curtain"},
                "track_3d_o3": {
                    "type": "track_map_3d", "pairs": ["synthetic_aircraft"],
                    "title": "O3 3D Track", "show_var": "obs", "show_coastlines": False,
                },
                "flight_ts_o3": {
                    "type": "flight_timeseries", "pairs": ["synthetic_aircraft"],
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
# 3. Obs-Only Pipeline (no model)
# =============================================================================


class TestObsOnlyPipeline:
    """End-to-end obs-only pipeline: aircraft data → 4 obs plot types."""

    def test_obs_only_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pipeline.runner import PipelineRunner

        rng = np.random.default_rng(99)
        n = 300

        # Build synthetic aircraft obs
        t = np.linspace(0, 3 * np.pi, n)
        times = np.datetime64("2012-05-29T14:00") + np.arange(n) * np.timedelta64(20, "s")
        lats = 37.0 + 2.0 * np.sin(t)
        lons = -97.0 + 3.0 * np.cos(t / 2)
        alts = 500 + 10000 * (0.5 + 0.5 * np.sin(t))
        flight_ids = np.where(np.arange(n) < 150, "2012-05-29", "2012-05-30")

        obs_ds = xr.Dataset(
            {
                "O3": ("time", 30.0 + 6.0 * (alts / 1000) + rng.normal(0, 5, n),
                       {"units": "ppbv", "long_name": "Ozone"}),
                "CO": ("time", 80.0 + 20.0 * np.exp(-alts / 5000) + rng.normal(0, 10, n),
                       {"units": "ppbv", "long_name": "Carbon Monoxide"}),
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

        obs_path = tmp_path / "obs_aircraft.nc"
        obs_ds.to_netcdf(obs_path)

        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        # No model section → triggers obs-only pipeline
        config = {
            "analysis": {
                "start_time": "2012-05-29",
                "end_time": "2012-05-31",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "obs": {
                "dc8": {
                    "obs_type": "aircraft",
                    "filename": str(obs_path),
                    "variables": {
                        "O3": {"units": "ppbv"},
                        "CO": {"units": "ppbv"},
                    },
                },
            },
            "plots": {
                "obs_ts_o3": {"type": "obs_timeseries", "obs": "dc8", "variable": "O3", "title": "O3 Time Series"},
                "obs_hist_o3": {"type": "obs_histogram", "obs": "dc8", "variable": "O3", "title": "O3 Distribution"},
                "obs_profile_o3": {"type": "obs_vertical_profile", "obs": "dc8", "variable": "O3", "title": "O3 Vertical Profile"},
                "obs_track_o3": {"type": "obs_flight_track", "obs": "dc8", "variable": "O3", "title": "O3 Flight Track"},
            },
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
        _assert_pipeline_success(result)
        _assert_plots(output_dir, min_count=4)
        assert list(log_dir.glob("pipeline_*.md")), "No pipeline log"
        _copy_artifacts(output_dir, log_dir, prefix="obsonly_")


# =============================================================================
# 4. Swath-to-Grid Pipeline (satellite L2)
# =============================================================================


class TestSwathGridPipeline:
    """End-to-end satellite evaluation: swath binning → grid pairing → pcolormesh plots.

    Tests the numba-accelerated binning of synthetic swath pixels onto a model
    grid, then feeds the binned result through the pipeline via load_binned.
    Skips HDF4 I/O (no pyhdf dependency needed).
    """

    def test_swath_pipeline(self, tmp_path: Path) -> None:
        from davinci_monet.pairing.grid_binning import (
            bin_swath_to_grid,
            edges_from_centers,
            normalize_grid,
        )
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.models import create_model_dataset

        rng = np.random.default_rng(77)

        # --- Model: small global-ish grid with AOD-like values ---
        domain = Domain(
            lon_min=100.0, lon_max=160.0,
            lat_min=-50.0, lat_max=0.0,
            n_lon=24, n_lat=20,
        )
        time_cfg = TimeConfig(start="2019-12-21", end="2019-12-22", freq="1D")

        # Build model AOD field directly: low background + Gaussian hotspot
        # Model overestimates the plume but underestimates background → spatial bias pattern
        lat_centers = np.linspace(domain.lat_min, domain.lat_max, domain.n_lat)
        lon_centers = np.linspace(domain.lon_min, domain.lon_max, domain.n_lon)
        time_vals = np.array(["2019-12-21"], dtype="datetime64[ns]")

        aod_field = np.full((1, domain.n_lat, domain.n_lon), 0.08)  # low background
        for i, la in enumerate(lat_centers):
            for j, lo in enumerate(lon_centers):
                dist = np.sqrt((la - (-25.0))**2 + (lo - 120.0)**2)
                aod_field[0, i, j] += 2.0 * np.exp(-dist**2 / 150)  # stronger, tighter plume
        aod_field += rng.normal(0, 0.03, aod_field.shape)
        aod_field = np.clip(aod_field, 0, 5)

        model_ds = xr.Dataset(
            {"AOD": (["time", "lat", "lon"], aod_field)},
            coords={"time": time_vals, "lat": lat_centers, "lon": lon_centers},
        )

        model_path = tmp_path / "model_aod.nc"
        model_ds.to_netcdf(model_path)

        # --- Synthetic swath: 500 scanlines × 20 pixels ---
        n_scan, n_pix = 500, 20
        scan_lats = np.linspace(-48.0, -2.0, n_scan)
        pix_offsets = np.linspace(-3.0, 3.0, n_pix)
        swath_lat = scan_lats[:, np.newaxis] + pix_offsets[np.newaxis, :] * 0.1
        swath_lon = np.linspace(102.0, 158.0, n_scan)[:, np.newaxis] + pix_offsets[np.newaxis, :]

        # AOD values: base field + hotspot + noise
        swath_aod = np.full((n_scan, n_pix), 0.15)
        for i in range(n_scan):
            for j in range(n_pix):
                dist = np.sqrt((swath_lat[i, j] - (-25.0))**2 + (swath_lon[i, j] - 120.0)**2)
                swath_aod[i, j] += 1.2 * np.exp(-dist**2 / 200)
        swath_aod += rng.normal(0, 0.05, (n_scan, n_pix))
        swath_aod = np.clip(swath_aod, 0, 5)

        # --- Bin swath onto model grid using numba ---
        lat_centers = model_ds.lat.values.astype(np.float64)
        lon_centers = model_ds.lon.values.astype(np.float64)
        lat_edges = edges_from_centers(lat_centers)
        lon_edges = edges_from_centers(lon_centers)

        # Single time step
        time_center = np.array([model_ds.time.values[0].astype("datetime64[s]").astype(np.float64)])
        time_edges = edges_from_centers(time_center)

        ntime, nlon, nlat = 1, len(lon_centers), len(lat_centers)
        count_grid = np.zeros((ntime, nlon, nlat), dtype=np.int64)
        data_grid = np.zeros((ntime, nlon, nlat), dtype=np.float64)

        # Flatten swath and assign all pixels to the single time bin
        flat_aod = swath_aod.ravel().astype(np.float64)
        flat_lat = swath_lat.ravel().astype(np.float64)
        flat_lon = swath_lon.ravel().astype(np.float64)
        flat_time = np.full(len(flat_aod), time_center[0], dtype=np.float64)

        bin_swath_to_grid(
            time_edges, lon_edges, lat_edges,
            flat_time, flat_lon, flat_lat, flat_aod,
            count_grid, data_grid,
        )
        normalize_grid(count_grid, data_grid)

        # Verify binning produced data
        assert np.any(count_grid > 0), "Binning produced no data"
        assert np.nanmax(data_grid) > 0.5, "Binning lost the hotspot signal"

        # --- Write binned result as NetCDF (same format as MODIS cache) ---
        binned_ds = xr.Dataset(
            {
                "AOD": (["time", "lon", "lat"], data_grid),
                "pixel_count": (["time", "lon", "lat"], count_grid.astype(np.float64)),
            },
            coords={
                "time": model_ds.time.values[:1],
                "lon": lon_centers,
                "lat": lat_centers,
            },
        )
        binned_path = tmp_path / "swath_binned.nc"
        binned_ds.to_netcdf(binned_path)

        # --- Run pipeline with load_binned ---
        output_dir = tmp_path / "output"
        log_dir = tmp_path / "logs"

        config = {
            "analysis": {
                "start_time": "2019-12-21",
                "end_time": "2019-12-22",
                "output_dir": str(output_dir),
                "log_dir": str(log_dir),
            },
            "model": {
                "cam6": {
                    "mod_type": "generic",
                    "files": str(model_path),
                    "radius_of_influence": 50000,
                    "mapping": {"satellite": {"AOD": "AOD"}},
                    "variables": {
                        "AOD": {
                            "units": "1",
                            "ylabel_plot": "AOD (550 nm)",
                            "vmin_plot": 0, "vmax_plot": 2.0, "vdiff_plot": 0.5,
                        },
                    },
                },
            },
            "obs": {
                "satellite": {
                    "obs_type": "sat_swath_clm",
                    "sat_type": "modis_l2",
                    "grid_source": "cam6",
                    "load_binned": True,
                    "binned_file": str(binned_path),
                    "variables": {
                        "AOD": {"units": "1"},
                    },
                },
            },
            "pairs": {
                "cam6_satellite": {
                    "model": "cam6", "obs": "satellite",
                    "variable": {"model_var": "AOD", "obs_var": "AOD"},
                },
            },
            "plots": {
                "aod_scatter": {
                    "type": "scatter", "pairs": ["cam6_satellite"],
                    "title": "AOD: Model vs Satellite", "show_density": True,
                },
                "aod_obs_map": {
                    "type": "spatial_distribution", "pairs": ["cam6_satellite"],
                    "title": "Satellite AOD", "show_var": "obs",
                    "plot_type": "pcolormesh", "cmap": "turbo",
                },
                "aod_bias_map": {
                    "type": "spatial_bias", "pairs": ["cam6_satellite"],
                    "title": "AOD Bias (Model − Satellite)",
                    "plot_type": "pcolormesh",
                },
            },
            "stats": {"metrics": ["N", "MB", "RMSE", "R", "NMB", "NME"]},
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
        _assert_pipeline_success(result)
        _assert_plots(output_dir, min_count=3)
        assert list(log_dir.glob("pipeline_*.md")), "No pipeline log"
        _copy_artifacts(output_dir, log_dir, prefix="swath_")
