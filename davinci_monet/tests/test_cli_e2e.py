"""CLI end-to-end tests for DAVINCI.

These tests invoke the CLI app with actual YAML config files, exercising
the full path: CLI → YAML parsing → load_config() → PipelineRunner.
This complements test_integration.py (which tests pipeline core with dicts)
and test_cli.py (which tests CLI argument parsing and help).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.cli.app import app
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_model_dataset
from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_data(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create synthetic NetCDF files and a YAML config pointing to them.

    Returns (config_path, output_dir, log_dir, tmp_path).
    """
    domain = Domain(
        lon_min=-105.0, lon_max=-95.0,
        lat_min=35.0, lat_max=45.0,
        n_lon=8, n_lat=8,
    )
    time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-16 00:00", freq="3h")

    model_ds = create_model_dataset(
        variables=["O3"], domain=domain, time_config=time_cfg, seed=42,
    )

    scenario = PerfectMatchScenario(
        variables=["O3"], domain=domain, time_config=time_cfg,
        geometry=DataGeometry.POINT, n_obs=5, noise_level=0.0, seed=42,
    )
    obs_ds = scenario._generate_point_obs(model_ds)

    # Add small bias so stats are non-trivial
    rng = np.random.default_rng(42)
    model_ds["O3"] = model_ds["O3"] + 3.0 + rng.normal(0, 2.0, size=model_ds["O3"].shape)

    model_path = tmp_path / "model.nc"
    obs_path = tmp_path / "obs.nc"
    model_ds.to_netcdf(model_path)
    obs_ds.to_netcdf(obs_path)

    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"

    config_text = textwrap.dedent(f"""\
        analysis:
          start_time: "2024-01-15"
          end_time: "2024-01-16"
          output_dir: "{output_dir}"
          log_dir: "{log_dir}"

        model:
          synthetic:
            mod_type: generic
            files: "{model_path}"
            radius_of_influence: 50000
            mapping:
              surface:
                O3: O3
            variables:
              O3:
                units: ppb
                vmin_plot: 30
                vmax_plot: 70
                vdiff_plot: 10

        obs:
          surface:
            obs_type: pt_sfc
            filename: "{obs_path}"
            variables:
              O3:
                obs_min: 0
                obs_max: 200
                units: ppb

        pairs:
          synthetic_surface:
            model: synthetic
            obs: surface
            variable:
              model_var: O3
              obs_var: O3

        plots:
          scatter_o3:
            type: scatter
            pairs: [synthetic_surface]
            title: "O3 Scatter"

        stats:
          metrics: [N, MB, RMSE]
    """)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    return config_path, output_dir, log_dir, tmp_path


# =============================================================================
# CLI Run Tests
# =============================================================================


class TestCLIRunE2E:
    """End-to-end tests for `davinci-monet run <config.yaml>`."""

    def test_cli_run_happy_path(self, synthetic_data: tuple) -> None:
        """Full pipeline through CLI with YAML config file."""
        from typer.testing import CliRunner

        config_path, output_dir, log_dir, _ = synthetic_data
        runner = CliRunner()
        result = runner.invoke(app, ["run", str(config_path)])

        assert result.exit_code == 0, (
            f"CLI failed with exit code {result.exit_code}.\n"
            f"stdout: {result.stdout}\n"
        )
        assert "Analysis complete" in result.stdout

        # Verify outputs
        png_files = list(output_dir.rglob("*.png"))
        assert len(png_files) >= 1, f"No plots generated in {output_dir}"

        csv_files = list(output_dir.rglob("*.csv"))
        assert len(csv_files) >= 1, f"No stats CSV in {output_dir}"

        log_files = list(log_dir.glob("pipeline_*.md"))
        assert len(log_files) >= 1, f"No pipeline log in {log_dir}"

    def test_cli_run_file_not_found(self, tmp_path: Path) -> None:
        """CLI with nonexistent config path exits with code 2."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(tmp_path / "nonexistent.yaml")])

        assert result.exit_code == 2
        assert "does not exist" in result.stdout

    def test_cli_run_invalid_config(self, tmp_path: Path) -> None:
        """CLI with invalid config shows configuration error."""
        from typer.testing import CliRunner

        config_path = tmp_path / "bad.yaml"
        config_path.write_text(textwrap.dedent("""\
            analysis:
              start_time: not-a-date
              end_time: also-not-a-date
        """))

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(config_path)])

        assert result.exit_code != 0


# =============================================================================
# Error Config Tests
# =============================================================================


ERROR_CONFIG_DIR = Path(__file__).resolve().parents[2] / "tests" / "error_configs"
ERROR_CONFIGS = sorted(ERROR_CONFIG_DIR.glob("*.yaml")) if ERROR_CONFIG_DIR.exists() else []


@pytest.mark.parametrize(
    "config_path",
    ERROR_CONFIGS,
    ids=lambda p: p.stem,
)
class TestErrorConfigs:
    """Validate that all curated error configs are properly rejected."""

    def test_error_config_rejected_by_validate(self, config_path: Path) -> None:
        """Each error config should fail validation with non-zero exit."""
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(config_path)])

        assert result.exit_code != 0 or "error" in result.stdout.lower(), (
            f"{config_path.name} was not rejected.\n"
            f"exit_code={result.exit_code}\n"
            f"stdout: {result.stdout}"
        )
