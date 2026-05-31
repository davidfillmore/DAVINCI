"""Tests for source-label aliasing on paired output (renderer rewire R-1).

R-1 adds an additive *dual-naming bridge*: alongside the legacy ``model_<v>`` /
``obs_<v>`` paired variables, the paired dataset also exposes
``<comparand_label>_<v>`` / ``<reference_label>_<v>`` aliases pointing at the same
data, each tagged with ``role`` and ``source_label`` attrs. This lets the
renderers (R-2/R-3) resolve series by source label and color by role while the
legacy prefixes keep working. The step is purely additive.

Pairing maps ``obs`` -> reference and ``model`` -> comparand, so a ``model_<v>``
variable aliases to ``<comparand_label>_<v>`` and ``obs_<v>`` to
``<reference_label>_<v>``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from davinci_monet.pipeline.stages import tag_paired_roles


def _paired() -> xr.Dataset:
    rng = np.random.default_rng(0)
    n = 8
    return xr.Dataset(
        {
            "model_o3": ("time", rng.uniform(10, 60, n)),
            "obs_o3": ("time", rng.uniform(10, 60, n)),
        },
        coords={"time": np.arange(n)},
    )


class TestSourceLabelAliases:
    """Unit tests of ``tag_paired_roles`` additive source-label aliasing."""

    def test_adds_source_label_aliases(self) -> None:
        ds = _paired()
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        # Legacy prefixes are retained.
        assert "model_o3" in ds.data_vars
        assert "obs_o3" in ds.data_vars
        # Source-label aliases are added (model -> comparand, obs -> reference).
        assert "cam_o3" in ds.data_vars
        assert "airnow_o3" in ds.data_vars

    def test_aliases_have_identical_values(self) -> None:
        ds = _paired()
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        np.testing.assert_array_equal(ds["cam_o3"].values, ds["model_o3"].values)
        np.testing.assert_array_equal(ds["airnow_o3"].values, ds["obs_o3"].values)

    def test_role_and_source_label_attrs(self) -> None:
        ds = _paired()
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        # Legacy vars carry both role and source_label.
        assert ds["model_o3"].attrs["role"] == "model"
        assert ds["model_o3"].attrs["source_label"] == "cam"
        assert ds["obs_o3"].attrs["role"] == "obs"
        assert ds["obs_o3"].attrs["source_label"] == "airnow"
        # Alias vars carry the same role and source_label.
        assert ds["cam_o3"].attrs["role"] == "model"
        assert ds["cam_o3"].attrs["source_label"] == "cam"
        assert ds["airnow_o3"].attrs["role"] == "obs"
        assert ds["airnow_o3"].attrs["source_label"] == "airnow"

    def test_no_labels_is_backward_compatible(self) -> None:
        # The existing call signature (no labels) must still tag roles only:
        # no aliases, no source_label. Guards the Phase 6 behavior.
        ds = _paired()
        tag_paired_roles(ds)
        assert set(ds.data_vars) == {"model_o3", "obs_o3"}
        assert ds["model_o3"].attrs["role"] == "model"
        assert ds["obs_o3"].attrs["role"] == "obs"
        assert "source_label" not in ds["model_o3"].attrs
        assert "source_label" not in ds["obs_o3"].attrs

    def test_alias_equal_to_legacy_name_is_noop(self) -> None:
        # Labels that collide with the reserved prefixes must not duplicate vars.
        ds = _paired()
        tag_paired_roles(ds, reference_label="obs", comparand_label="model")
        assert set(ds.data_vars) == {"model_o3", "obs_o3"}
        # source_label is still recorded on the legacy vars.
        assert ds["model_o3"].attrs["source_label"] == "model"
        assert ds["obs_o3"].attrs["source_label"] == "obs"

    def test_reserved_prefix_labels_do_not_pollute_namespace(self) -> None:
        # A pathological source label that starts with a reserved prefix
        # (e.g. "model_foo") must NOT create an alias that re-enters the
        # model_/obs_ namespace; downstream prefix-based selection (statistics,
        # per-flight stats, var counts) keys off those prefixes and would
        # otherwise mistake the alias for a legacy variable.
        ds = _paired()
        tag_paired_roles(ds, reference_label="obs_x", comparand_label="model_foo")
        assert set(ds.data_vars) == {"model_o3", "obs_o3"}
        # source_label is still recorded on the legacy vars.
        assert ds["model_o3"].attrs["source_label"] == "model_foo"
        assert ds["obs_o3"].attrs["source_label"] == "obs_x"


class TestPairedSourceLabelPipeline:
    """Integration test: paired output from the pipeline carries source-label aliases.

    Runs the real user path (``PipelineRunner.run_from_config``) with a model
    labelled ``cam`` and obs labelled ``airnow`` and a ``stats`` block (so the
    statistics stage runs and proves the additive aliases do not break
    downstream prefix-based variable selection).
    """

    def test_pipeline_emits_source_label_aliases(self, tmp_path: Path) -> None:
        from davinci_monet.core.protocols import DataGeometry
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
        from davinci_monet.tests.synthetic.models import create_model_dataset
        from davinci_monet.tests.synthetic.scenarios import PerfectMatchScenario

        domain = Domain(
            lon_min=-105.0,
            lon_max=-95.0,
            lat_min=35.0,
            lat_max=45.0,
            n_lon=8,
            n_lat=8,
        )
        time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-15 06:00", freq="1h")

        model_ds = create_model_dataset(
            variables=["O3"], domain=domain, time_config=time_cfg, seed=42
        )
        scenario = PerfectMatchScenario(
            variables=["O3"],
            domain=domain,
            time_config=time_cfg,
            geometry=DataGeometry.POINT,
            n_obs=6,
            noise_level=0.0,
            seed=42,
        )
        obs_ds = scenario._generate_point_obs(model_ds)

        model_path = tmp_path / "model.nc"
        obs_path = tmp_path / "obs.nc"
        model_ds.to_netcdf(model_path)
        obs_ds.to_netcdf(obs_path)

        config = {
            "analysis": {
                "start_time": "2024-01-15 00:00",
                "end_time": "2024-01-15 06:00",
                "output_dir": str(tmp_path / "output"),
                "log_dir": str(tmp_path / "logs"),
            },
            "model": {
                "cam": {
                    "mod_type": "generic",
                    "files": str(model_path),
                    "radius_of_influence": 50000,
                    "mapping": {"airnow": {"O3": "O3"}},
                    "variables": {"O3": {"units": "ppb"}},
                },
            },
            "obs": {
                "airnow": {
                    "obs_type": "pt_sfc",
                    "filename": str(obs_path),
                    "variables": {"O3": {"obs_min": 0, "obs_max": 200, "units": "ppb"}},
                },
            },
            "pairs": {
                "cam_airnow": {
                    "model": "cam",
                    "obs": "airnow",
                    "variable": {"model_var": "O3", "obs_var": "O3"},
                },
            },
            "stats": {"metrics": ["N", "MB", "RMSE", "R"]},
        }

        runner = PipelineRunner(show_progress=False)
        result = runner.run_from_config(config)
        assert result.success, (
            "Pipeline failed. Failed stages: "
            f"{[s.stage_name + ': ' + str(s.error) for s in result.failed_stages]}"
        )

        assert result.context is not None
        paired = result.context.paired
        assert paired, "no paired output produced"

        paired_obj = next(iter(paired.values()))
        ds = paired_obj.data if hasattr(paired_obj, "data") else paired_obj

        # Both legacy prefixes and source-label aliases are present.
        assert "model_O3" in ds.data_vars
        assert "obs_O3" in ds.data_vars
        assert "cam_O3" in ds.data_vars
        assert "airnow_O3" in ds.data_vars

        # Aliases share the legacy data exactly.
        np.testing.assert_array_equal(ds["cam_O3"].values, ds["model_O3"].values)
        np.testing.assert_array_equal(ds["airnow_O3"].values, ds["obs_O3"].values)

        # Aliases self-describe role and source label.
        assert ds["cam_O3"].attrs["role"] == "model"
        assert ds["cam_O3"].attrs["source_label"] == "cam"
        assert ds["airnow_O3"].attrs["role"] == "obs"
        assert ds["airnow_O3"].attrs["source_label"] == "airnow"
