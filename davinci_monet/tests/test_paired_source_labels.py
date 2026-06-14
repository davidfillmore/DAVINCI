"""Tests for source-label paired naming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from davinci_monet.core.base import PairedData, iter_paired_variable_xy
from davinci_monet.core.protocols import DataGeometry


@pytest.mark.integration
class TestPairedSourceLabelPipeline:
    """Integration test: the pipeline emits source-label paired names only.

    Runs the real user path (``PipelineRunner.run_from_config``) with a dataset
    labelled ``cam`` and geometry labelled ``airnow`` plus a ``stats`` block, proving
    the statistics stage works with pair-axis selection (no dataset_/geometry_ prefix).
    """

    def test_pipeline_emits_dataset_label_names(self, tmp_path: Path) -> None:
        from davinci_monet.core.protocols import DataGeometry
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.tests.synthetic.datasets import create_dataset_dataset
        from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
        from davinci_monet.tests.synthetic.scenarios import (
            PerfectMatchScenario,
            sample_geometry_from,
        )

        domain = Domain(
            lon_min=-105.0,
            lon_max=-95.0,
            lat_min=35.0,
            lat_max=45.0,
            n_lon=8,
            n_lat=8,
        )
        time_cfg = TimeConfig(start="2024-01-15 00:00", end="2024-01-15 06:00", freq="1h")

        dataset_ds = create_dataset_dataset(
            variables=["O3"], domain=domain, time_config=time_cfg, seed=42
        )
        scenario = PerfectMatchScenario(
            variables=["O3"],
            domain=domain,
            time_config=time_cfg,
            geometry=DataGeometry.POINT,
            n_geometry=6,
            noise_level=0.0,
            seed=42,
        )
        geometry_ds = sample_geometry_from(dataset_ds, "point", scenario=scenario)

        dataset_path = tmp_path / "dataset.nc"
        geometry_path = tmp_path / "geometry.nc"
        dataset_ds.to_netcdf(dataset_path)
        geometry_ds.to_netcdf(geometry_path)

        config = {
            "analysis": {
                "start_time": "2024-01-15 00:00",
                "end_time": "2024-01-15 06:00",
                "output_dir": str(tmp_path / "output"),
                "log_dir": str(tmp_path / "logs"),
            },
            "sources": {
                "cam": {
                    "type": "generic",
                    "files": str(dataset_path),
                    "radius_of_influence": 50000,
                    "variables": {"O3": {"units": "ppb"}},
                },
                "airnow": {
                    "type": "pt_sfc",
                    "filename": str(geometry_path),
                    "variables": {"O3": {"valid_min": 0, "valid_max": 200, "units": "ppb"}},
                },
            },
            "pairs": {
                "cam_airnow": {
                    "sources": ["cam", "airnow"],
                    "geometry": "airnow",
                    "variables": {"cam": "O3", "airnow": "O3"},
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

        # Source-label names only.
        assert "cam_O3" in ds.data_vars
        assert "airnow_O3" in ds.data_vars
        assert "dataset_O3" not in ds.data_vars
        assert "geometry_O3" not in ds.data_vars

        # Vars self-describe axis and source label.
        assert ds["cam_O3"].attrs["axis"] == "y"
        assert ds["cam_O3"].attrs["source_label"] == "cam"
        assert ds["airnow_O3"].attrs["axis"] == "x"
        assert ds["airnow_O3"].attrs["source_label"] == "airnow"

        # Statistics still computed for the canonical variable (pair-axis).
        stats_files = list((tmp_path / "output").rglob("*.csv"))
        assert stats_files, "no statistics CSV produced"


class TestPairedHelperRobustness:
    """Pair-axis and canonical helpers stay internally consistent."""

    def test_iter_pairs_handles_mixed_case_prefixes(self) -> None:
        # Prefix matching and canonical-name handling are both case-insensitive.
        ds = xr.Dataset(
            {"Dataset_O3": ("time", np.zeros(3)), "geometry_O3": ("time", np.ones(3))},
            coords={"time": np.arange(3)},
        )
        ds["Dataset_O3"].attrs["axis"] = "y"
        ds["geometry_O3"].attrs["axis"] = "x"
        assert iter_paired_variable_xy(ds) == [("geometry_O3", "Dataset_O3", "O3")]

    def test_geometry_dataset_resolve_canonical(self) -> None:
        ds = xr.Dataset(
            {"airnow_o3": ("time", np.ones(3)), "cam_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        ds["airnow_o3"].attrs.update({"axis": "x", "source_label": "airnow"})
        ds["cam_o3"].attrs.update({"axis": "y", "source_label": "cam"})
        pd = PairedData(data=ds, y_source="cam", x_source="airnow", geometry=DataGeometry.POINT)
        np.testing.assert_array_equal(pd.get_geometry("o3").values, np.ones(3))
        np.testing.assert_array_equal(pd.get_dataset("o3").values, np.zeros(3))

    def test_geometry_dataset_accessors_are_canonical(self) -> None:
        ds = xr.Dataset(
            {"airnow_o3": ("time", np.ones(3)), "cam_o3": ("time", np.zeros(3))},
            coords={"time": np.arange(3)},
        )
        ds["airnow_o3"].attrs.update({"axis": "x", "source_label": "airnow"})
        ds["cam_o3"].attrs.update({"axis": "y", "source_label": "cam"})
        pd = PairedData(
            data=ds,
            y_source="cam",
            x_source="airnow",
            geometry=DataGeometry.POINT,
            pairing_info={"geometry_label": "airnow", "source_label": "cam"},
        )

        assert pd.x_source == "airnow"
        assert pd.y_source == "cam"
        assert pd.geometry_variables == ["airnow_o3"]
        assert pd.dataset_variables == ["cam_o3"]
        np.testing.assert_array_equal(pd.get_geometry("o3").values, np.ones(3))
        np.testing.assert_array_equal(pd.get_dataset("o3").values, np.zeros(3))

    def test_prefix_fallback_respects_axis(self) -> None:
        # get_dataset must not return a prefixed var whose axis attr is 'x'.
        ds = xr.Dataset(
            {"dataset_o3": ("time", np.zeros(3)), "geometry_o3": ("time", np.ones(3))},
            coords={"time": np.arange(3)},
        )
        ds["dataset_o3"].attrs["axis"] = "y"
        ds["geometry_o3"].attrs["axis"] = "x"
        pd = PairedData(data=ds, y_source="cam", x_source="airnow", geometry=DataGeometry.POINT)
        np.testing.assert_array_equal(pd.get_dataset("o3").values, np.zeros(3))
        np.testing.assert_array_equal(pd.get_geometry("o3").values, np.ones(3))
