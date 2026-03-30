"""Tests for radiative analysis runner and CLI integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

# ---------------------------------------------------------------------------
# Fixture — full synthetic setup
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_synthetic_setup(tmp_path):
    """Build synthetic CERES, MERRA-2, AERONET, and a YAML config.

    CERES files span Sep 5-15 (11 days).  Background window is 3 days
    (uses days 5-7).  MERRA-2 files cover the same range.
    """
    rng = np.random.default_rng(42)
    domain = (-135, -105, 30, 52)

    # --- Synthetic CERES dir ---
    ceres_dir = tmp_path / "ceres"
    ceres_dir.mkdir()
    lats = np.linspace(89.5, -89.5, 180)  # descending
    lons = np.linspace(-179.5, 179.5, 360)
    ceres_vars = [
        "obs_all_toa_sw",
        "obs_clr_toa_sw",
        "obs_all_toa_net",
        "init_match_aod55",
        "obs_cld_amount",
        "toa_sw_insol",
    ]
    for day in range(5, 16):  # Sep 5 - Sep 15
        ds = xr.Dataset(
            {
                name: (["lat", "lon"], rng.uniform(10, 400, (180, 360)).astype("f4"))
                for name in ceres_vars
            },
            coords={"lat": lats, "lon": lons},
        )
        ds.to_netcdf(ceres_dir / f"CERES_SYN1deg-Day_202009{day:02d}.nc")

    # --- Synthetic MERRA-2 dir ---
    m2_dir = tmp_path / "merra2"
    m2_dir.mkdir()
    m2_lats = np.arange(-90, 90.5, 0.5)
    m2_lons = np.arange(-180, 180.625, 0.625)
    for day in range(5, 16):
        times = pd.date_range(f"2020-09-{day:02d}", periods=8, freq="3h")
        ds = xr.Dataset(
            {
                "TOTEXTTAU": (
                    ["time", "lat", "lon"],
                    rng.uniform(0, 1, (8, len(m2_lats), len(m2_lons))).astype("f4"),
                ),
                "OCEXTTAU": (
                    ["time", "lat", "lon"],
                    rng.uniform(0, 0.5, (8, len(m2_lats), len(m2_lons))).astype("f4"),
                ),
                "BCEXTTAU": (
                    ["time", "lat", "lon"],
                    rng.uniform(0, 0.1, (8, len(m2_lats), len(m2_lons))).astype("f4"),
                ),
            },
            coords={"time": times, "lat": m2_lats, "lon": m2_lons},
        )
        ds.to_netcdf(m2_dir / f"MERRA2_401.tavg1_2d_aer_Nx.202009{day:02d}.nc4")

    # --- Synthetic AERONET CSV ---
    aeronet_csv = tmp_path / "aeronet.csv"
    rows = []
    sites = [
        ("Fresno_2", 36.8, -119.8),
        ("UCSB", 34.4, -119.8),
    ]
    for day in range(5, 16):
        for name, lat, lon in sites:
            for hour in [8, 12, 16]:
                rows.append(
                    {
                        "time": f"2020-09-{day:02d} {hour:02d}:00:00",
                        "site": name,
                        "AOD_500nm": float(rng.uniform(0.1, 2.0)),
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
    pd.DataFrame(rows).to_csv(aeronet_csv, index=False)

    # --- YAML config ---
    output_dir = tmp_path / "output"
    config = {
        "radiative": {
            "event": {
                "name": "test-event",
                "start_time": "2020-09-05",
                "end_time": "2020-09-15",
                "domain": [-135, -105, 30, 52],
                "background_window": 3,
            },
            "ceres": {
                "product": "syn1deg",
                "source": "local",
                "files": str(ceres_dir / "CERES_SYN1deg-Day_*.nc"),
            },
            "merra2": {
                "files": str(m2_dir / "MERRA2_401.tavg1_2d_aer_Nx.*.nc4"),
            },
            "aeronet": {
                "files": str(aeronet_csv),
                "sites": ["Fresno_2", "UCSB"],
            },
            "plots": ["toa_event_fields", "anomaly_maps"],
            "output_dir": str(output_dir),
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path, output_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunRadiativeAnalysis:
    def test_run_produces_plots(self, full_synthetic_setup):
        """Runner returns success with at least 2 plots generated."""
        import matplotlib

        matplotlib.use("Agg")

        from davinci_monet.radiative.runner import run_radiative_analysis

        config_path, _ = full_synthetic_setup
        result = run_radiative_analysis(str(config_path))

        assert result["success"] is True
        assert len(result["plots_generated"]) >= 2
        assert len(result["errors"]) == 0

    def test_output_dir_created(self, full_synthetic_setup):
        """Output directory exists after run and contains PNG files."""
        import matplotlib

        matplotlib.use("Agg")

        from davinci_monet.radiative.runner import run_radiative_analysis

        config_path, output_dir = full_synthetic_setup
        run_radiative_analysis(str(config_path))

        assert output_dir.exists()
        pngs = list(output_dir.glob("*.png"))
        assert len(pngs) >= 2
