"""Tests for source-label paired naming (renderer rewire R-5, the clean break).

R-5 drops the legacy dual-naming bridge: when source labels are supplied,
``tag_paired_roles`` now *renames* the paired variables to ``<comparand_label>_<v>``
(model/comparand side) and ``<reference_label>_<v>`` (obs/reference side),
dropping the legacy ``model_<v>`` / ``obs_<v>`` names, and tags each with ``role``
and ``source_label`` attrs.

Without labels (the low-level engine API path, and untagged/legacy data), the
legacy names are kept and only the ``role`` attr is set — so direct
``engine.pair`` / ``strategy.pair`` callers are unaffected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from davinci_monet.core.base import PairedData, iter_paired_variable_pairs
from davinci_monet.core.protocols import DataGeometry
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


class TestSourceLabelRename:
    """Unit tests of ``tag_paired_roles`` source-label renaming (R-5)."""

    def test_renames_to_source_labels(self) -> None:
        ds = _paired()
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        # Renamed to source-label names (model -> comparand, obs -> reference).
        assert "cam_o3" in ds.data_vars
        assert "airnow_o3" in ds.data_vars
        # Legacy prefixes are dropped.
        assert "model_o3" not in ds.data_vars
        assert "obs_o3" not in ds.data_vars
        assert set(ds.data_vars) == {"cam_o3", "airnow_o3"}

    def test_renamed_vars_preserve_values(self) -> None:
        ds = _paired()
        model_before = ds["model_o3"].values.copy()
        obs_before = ds["obs_o3"].values.copy()
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        np.testing.assert_array_equal(ds["cam_o3"].values, model_before)
        np.testing.assert_array_equal(ds["airnow_o3"].values, obs_before)

    def test_role_and_source_label_attrs(self) -> None:
        ds = _paired()
        tag_paired_roles(ds, reference_label="airnow", comparand_label="cam")
        assert ds["cam_o3"].attrs["role"] == "model"
        assert ds["cam_o3"].attrs["source_label"] == "cam"
        assert ds["airnow_o3"].attrs["role"] == "obs"
        assert ds["airnow_o3"].attrs["source_label"] == "airnow"

    def test_no_labels_keeps_legacy_names(self) -> None:
        # The low-level path (no labels) keeps the legacy names and only tags
        # role, so direct engine.pair / strategy.pair callers are unaffected.
        ds = _paired()
        tag_paired_roles(ds)
        assert set(ds.data_vars) == {"model_o3", "obs_o3"}
        assert ds["model_o3"].attrs["role"] == "model"
        assert ds["obs_o3"].attrs["role"] == "obs"
        assert "source_label" not in ds["model_o3"].attrs
        assert "source_label" not in ds["obs_o3"].attrs

    def test_reserved_prefix_labels_keep_legacy_names(self) -> None:
        # A label whose rename would re-enter the reserved model_/obs_ namespace
        # (e.g. "model"/"obs" or "model_foo") must NOT rename — the legacy name is
        # kept (and source_label still recorded).
        ds = _paired()
        tag_paired_roles(ds, reference_label="obs", comparand_label="model")
        assert set(ds.data_vars) == {"model_o3", "obs_o3"}
        assert ds["model_o3"].attrs["source_label"] == "model"
        assert ds["obs_o3"].attrs["source_label"] == "obs"


class TestPairedSourceLabelPipeline:
    """Integration test: the pipeline emits source-label paired names only.

    Runs the real user path (``PipelineRunner.run_from_config``) with a model
    labelled ``cam`` and obs labelled ``airnow`` plus a ``stats`` block, proving
    the statistics stage works with role-based selection (no model_/obs_ prefix).
    """

    def test_pipeline_emits_source_label_names(self, tmp_path: Path) -> None:
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

        # Source-label names only — the legacy prefixes are gone.
        assert "cam_O3" in ds.data_vars
        assert "airnow_O3" in ds.data_vars
        assert "model_O3" not in ds.data_vars
        assert "obs_O3" not in ds.data_vars

        # Vars self-describe role and source label.
        assert ds["cam_O3"].attrs["role"] == "model"
        assert ds["cam_O3"].attrs["source_label"] == "cam"
        assert ds["airnow_O3"].attrs["role"] == "obs"
        assert ds["airnow_O3"].attrs["source_label"] == "airnow"

        # Statistics still computed for the canonical variable (role-based).
        stats_files = list((tmp_path / "output").rglob("*.csv"))
        assert stats_files, "no statistics CSV produced"


class TestPairedHelperRobustness:
    """Hardening from the R-5 review: role/canonical helpers must be internally
    consistent and respect roles."""

    def test_iter_pairs_handles_mixed_case_prefixes(self) -> None:
        # paired_variable_role matches prefixes case-insensitively; the canonical
        # helper must too, so mixed-case legacy names still pair.
        ds = xr.Dataset(
            {"Model_O3": ("time", np.zeros(3)), "obs_O3": ("time", np.ones(3))},
            coords={"time": np.arange(3)},
        )
        ds["Model_O3"].attrs["role"] = "model"
        ds["obs_O3"].attrs["role"] = "obs"
        assert iter_paired_variable_pairs(ds) == [("obs_O3", "Model_O3", "O3")]

    def test_get_obs_get_model_resolve_canonical(self) -> None:
        ds = xr.Dataset(
            {"airnow_o3": ("time", np.ones(3)), "cam_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        ds["airnow_o3"].attrs.update({"role": "obs", "source_label": "airnow"})
        ds["cam_o3"].attrs.update({"role": "model", "source_label": "cam"})
        pd = PairedData(data=ds, model_label="cam", obs_label="airnow", geometry=DataGeometry.POINT)
        np.testing.assert_array_equal(pd.get_obs("o3").values, np.ones(3))
        np.testing.assert_array_equal(pd.get_model("o3").values, np.zeros(3))

    def test_legacy_fallback_respects_role(self) -> None:
        # get_model must not return a legacy-named var whose role attr is 'obs'.
        ds = xr.Dataset(
            {"model_o3": ("time", np.zeros(3)), "obs_o3": ("time", np.ones(3))},
            coords={"time": np.arange(3)},
        )
        ds["model_o3"].attrs["role"] = "model"
        ds["obs_o3"].attrs["role"] = "obs"
        pd = PairedData(data=ds, model_label="cam", obs_label="airnow", geometry=DataGeometry.POINT)
        np.testing.assert_array_equal(pd.get_model("o3").values, np.zeros(3))
        np.testing.assert_array_equal(pd.get_obs("o3").values, np.ones(3))
