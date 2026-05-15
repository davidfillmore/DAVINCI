"""Tests for PlumeSentinel add-on config schema."""

from __future__ import annotations

import pytest

from davinci_monet.config.schema import AnalysisConfig


class TestAnalysisConfigWorkflow:
    def test_workflow_defaults_to_none(self):
        cfg = AnalysisConfig()
        assert cfg.workflow is None

    def test_workflow_accepts_plume_sentinel(self):
        cfg = AnalysisConfig(workflow="plume_sentinel")
        assert cfg.workflow == "plume_sentinel"

    def test_workflow_accepts_none_explicitly(self):
        cfg = AnalysisConfig(workflow=None)
        assert cfg.workflow is None


class TestPlumeSentinelConfigGOESHMS:
    """Test GOES true-color + HMS smoke overlay config parsing."""

    def test_parse_goes_hms_config(self):
        from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig

        raw = {
            "inputs": {
                "goes_east": {
                    "type": "goes_truecolor",
                    "file": "/data/goes/OR_ABI-L2-MCMIPC-M6_G16_s20231801800.nc",
                    "gamma": 2.0,
                },
                "hms_smoke": {
                    "type": "hms_smoke",
                    "file": "/data/hms/hms_smoke_20230629.shp",
                },
            },
            "plots": {
                "smoke_overview": {
                    "type": "truecolor_overlay",
                    "background": "goes_east",
                    "overlays": ["hms_smoke"],
                    "extent": [-130, -60, 20, 55],
                    "title": "GOES-16 with HMS Smoke",
                },
            },
        }
        cfg = PlumeSentinelConfig(**raw)

        assert "goes_east" in cfg.inputs
        assert cfg.inputs["goes_east"].type == "goes_truecolor"
        assert cfg.inputs["goes_east"].gamma == 2.0
        assert cfg.inputs["hms_smoke"].type == "hms_smoke"

        assert "smoke_overview" in cfg.plots
        assert cfg.plots["smoke_overview"].type == "truecolor_overlay"
        assert cfg.plots["smoke_overview"].background == "goes_east"
        assert cfg.plots["smoke_overview"].overlays == ["hms_smoke"]
        assert cfg.plots["smoke_overview"].extent == [-130, -60, 20, 55]

    def test_input_spec_defaults(self):
        from davinci_monet.addons.plume_sentinel.schema import InputSpec

        spec = InputSpec(type="goes_truecolor")
        assert spec.file is None
        assert spec.files is None
        assert spec.gamma == 1.8
        assert spec.variable is None
        assert spec.valid_range is None
        assert spec.grid is None


class TestPlumeSentinelConfigMODIS:
    """Test MODIS AOD config parsing with gridding."""

    def test_parse_modis_aod_config(self):
        from davinci_monet.addons.plume_sentinel.schema import PlumeSentinelConfig

        raw = {
            "inputs": {
                "modis_aod": {
                    "type": "modis_l2_aod",
                    "files": [
                        "/data/modis/MOD04_L2.A2023180.1800.061.hdf",
                        "/data/modis/MOD04_L2.A2023180.1805.061.hdf",
                    ],
                    "variable": "Optical_Depth_Land_And_Ocean",
                    "valid_range": [0.0, 5.0],
                    "grid": {
                        "resolution": 0.25,
                        "lon_range": [-130.0, -60.0],
                        "lat_range": [20.0, 55.0],
                        "min_obs_count": 2,
                    },
                },
            },
            "plots": {
                "aod_map": {
                    "type": "gridded_field",
                    "field": "modis_aod",
                    "background": {
                        "type": "gibs_wmts",
                        "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
                        "date": "2023-06-29",
                    },
                    "extent": [-130, -60, 20, 55],
                    "projection": {
                        "type": "LambertConformal",
                        "central_longitude": -95.0,
                        "central_latitude": 37.5,
                    },
                    "cmap": "YlOrRd",
                    "alpha": 0.6,
                    "colorbar_label": "AOD 550nm",
                    "title": "MODIS AOD",
                },
            },
        }
        cfg = PlumeSentinelConfig(**raw)

        # Verify input
        aod = cfg.inputs["modis_aod"]
        assert aod.type == "modis_l2_aod"
        assert len(aod.files) == 2
        assert aod.variable == "Optical_Depth_Land_And_Ocean"
        assert aod.valid_range == [0.0, 5.0]
        assert aod.grid is not None
        assert aod.grid.resolution == 0.25
        assert aod.grid.min_obs_count == 2
        assert aod.grid.lon_range == [-130.0, -60.0]
        assert aod.grid.lat_range == [20.0, 55.0]

        # Verify plot
        plot = cfg.plots["aod_map"]
        assert plot.type == "gridded_field"
        assert plot.field == "modis_aod"
        assert plot.alpha == 0.6
        assert plot.cmap == "YlOrRd"
        assert plot.colorbar_label == "AOD 550nm"

        # Background parsed as GibsBackgroundConfig
        assert plot.background.type == "gibs_wmts"
        assert plot.background.layer == "MODIS_Terra_CorrectedReflectance_TrueColor"
        assert plot.background.date == "2023-06-29"

        # Projection
        assert plot.projection.type == "LambertConformal"
        assert plot.projection.central_longitude == -95.0
        assert plot.projection.central_latitude == 37.5

    def test_grid_config_defaults(self):
        from davinci_monet.addons.plume_sentinel.schema import GridConfig

        grid = GridConfig(resolution=0.5, lon_range=[-180, 180], lat_range=[-90, 90])
        assert grid.min_obs_count == 1

    def test_plot_spec_defaults(self):
        from davinci_monet.addons.plume_sentinel.schema import PlotSpec

        plot = PlotSpec(type="truecolor_overlay")
        assert plot.background is None
        assert plot.overlays is None
        assert plot.field is None
        assert plot.extent is None
        assert plot.projection is None
        assert plot.title is None
        assert plot.cmap is None
        assert plot.alpha == 0.7
        assert plot.colorbar_label is None

    def test_projection_config_defaults(self):
        from davinci_monet.addons.plume_sentinel.schema import ProjectionConfig

        proj = ProjectionConfig(type="PlateCarree")
        assert proj.central_longitude == 0.0
        assert proj.central_latitude == 0.0

    def test_background_string_passthrough(self):
        from davinci_monet.addons.plume_sentinel.schema import PlotSpec

        plot = PlotSpec(type="truecolor_overlay", background="goes_east")
        assert plot.background == "goes_east"

    def test_background_dict_parsed(self):
        from davinci_monet.addons.plume_sentinel.schema import PlotSpec

        plot = PlotSpec(
            type="gridded_field",
            background={
                "type": "gibs_wmts",
                "layer": "BlueMarble",
                "date": "2023-01-01",
            },
        )
        assert plot.background.type == "gibs_wmts"
        assert plot.background.layer == "BlueMarble"


class TestWorkflowFactory:
    def test_creates_three_stages(self):
        from davinci_monet.addons.plume_sentinel.workflow import (
            create_plume_sentinel_pipeline,
        )

        stages = create_plume_sentinel_pipeline()
        assert len(stages) == 3

    def test_stage_names(self):
        from davinci_monet.addons.plume_sentinel.workflow import (
            create_plume_sentinel_pipeline,
        )

        stages = create_plume_sentinel_pipeline()
        names = [s.name for s in stages]
        assert names == ["load_inputs", "prepare_geospatial", "plotting"]


import pytest
from pydantic import ValidationError

from davinci_monet.addons.plume_sentinel.schema import (
    BulletinConfig,
    MqttConfig,
    PlumeSentinelConfig,
)


def test_mqtt_config_defaults():
    cfg = MqttConfig(topic="plume-sentinel-ai/reports/test")
    assert cfg.broker == "broker.hivemq.com"
    assert cfg.port == 1883
    assert cfg.qos == 0
    assert cfg.topic == "plume-sentinel-ai/reports/test"


def test_bulletin_config_defaults():
    cfg = BulletinConfig()
    assert cfg.template is None
    assert cfg.output_filename == "bulletin.txt"
    assert cfg.model == "claude-sonnet-4-6"
    assert cfg.include_images is False
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.mqtt is None
    assert cfg.on_error == "warn"


def test_bulletin_config_on_error_only_accepts_warn():
    with pytest.raises(ValidationError):
        BulletinConfig(on_error="fail")


def test_bulletin_config_parses_mqtt_subblock():
    cfg = BulletinConfig(
        mqtt={"topic": "plume-sentinel-ai/reports/west-coast", "qos": 1}
    )
    assert isinstance(cfg.mqtt, MqttConfig)
    assert cfg.mqtt.topic == "plume-sentinel-ai/reports/west-coast"
    assert cfg.mqtt.qos == 1


def test_plume_sentinel_config_bulletin_optional():
    cfg = PlumeSentinelConfig(inputs={}, plots={})
    assert cfg.bulletin is None


def test_plume_sentinel_config_bulletin_present():
    cfg = PlumeSentinelConfig(
        inputs={},
        plots={},
        bulletin={"model": "claude-opus-4-7", "include_images": True},
    )
    assert cfg.bulletin is not None
    assert cfg.bulletin.model == "claude-opus-4-7"
    assert cfg.bulletin.include_images is True
