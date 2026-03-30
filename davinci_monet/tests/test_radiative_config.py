"""Tests for radiative analysis configuration schema."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from davinci_monet.radiative.config import (
    AeronetConfig,
    CeresConfig,
    EventConfig,
    Merra2Config,
    RadiativeConfig,
    SurfaceImpactConfig,
)


class TestEventConfig:
    def test_valid_event(self) -> None:
        cfg = EventConfig(
            name="Camp Fire",
            start_time=datetime(2018, 11, 8),
            end_time=datetime(2018, 11, 25),
            domain=(-125.0, -119.0, 37.0, 42.0),
        )
        assert cfg.name == "Camp Fire"
        assert cfg.background_window == 3  # default

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValidationError, match="end_time must be after start_time"):
            EventConfig(
                name="bad",
                start_time=datetime(2018, 11, 25),
                end_time=datetime(2018, 11, 8),
                domain=(-125.0, -119.0, 37.0, 42.0),
            )

    def test_custom_background_window(self) -> None:
        cfg = EventConfig(
            name="Test",
            start_time=datetime(2020, 1, 1),
            end_time=datetime(2020, 1, 10),
            domain=(-100.0, -90.0, 30.0, 40.0),
            background_window=7,
        )
        assert cfg.background_window == 7


class TestCeresConfig:
    def test_local_source(self) -> None:
        cfg = CeresConfig(
            product="ebaf",
            source="local",
            files="/data/ceres/*.nc",
        )
        assert cfg.product == "ebaf"
        assert cfg.source == "local"
        assert cfg.files == "/data/ceres/*.nc"

    def test_opendap_source(self) -> None:
        cfg = CeresConfig(
            product="syn1deg",
            source="opendap",
        )
        assert cfg.files is None

    def test_local_requires_files(self) -> None:
        with pytest.raises(ValidationError, match="files is required when source is local"):
            CeresConfig(
                product="ebaf",
                source="local",
            )


class TestRadiativeConfig:
    @pytest.fixture()
    def event_data(self) -> dict:
        return {
            "name": "Camp Fire",
            "start_time": "2018-11-08",
            "end_time": "2018-11-25",
            "domain": (-125.0, -119.0, 37.0, 42.0),
        }

    @pytest.fixture()
    def ceres_data(self) -> dict:
        return {
            "product": "ebaf",
            "source": "opendap",
        }

    def test_minimal_config(self, event_data: dict, ceres_data: dict) -> None:
        cfg = RadiativeConfig(
            event=event_data,
            ceres=ceres_data,
            plots=["toa_event_fields", "anomaly_maps"],
            output_dir="/tmp/output",
        )
        assert cfg.merra2 is None
        assert cfg.aeronet is None
        assert cfg.surface_impact is None

    def test_full_config(self, event_data: dict, ceres_data: dict) -> None:
        cfg = RadiativeConfig(
            event=event_data,
            ceres=ceres_data,
            merra2={"files": "/data/merra2/*.nc", "smoke_species": ["OCEXTTAU"]},
            aeronet={"files": "/data/aeronet/*.lev20", "sites": ["Fresno", "Modesto"]},
            surface_impact={"method": ["semi_empirical"], "ssa": 0.90},
            plots=["toa_event_fields", "sw_vs_aod_scatter", "surface_impact"],
            output_dir="/tmp/output",
        )
        assert cfg.merra2 is not None
        assert cfg.aeronet is not None
        assert cfg.surface_impact is not None
        assert cfg.surface_impact.ssa == 0.90

    def test_surface_impact_requires_merra2(self, event_data: dict, ceres_data: dict) -> None:
        with pytest.raises(ValidationError, match="surface_impact requires merra2"):
            RadiativeConfig(
                event=event_data,
                ceres=ceres_data,
                surface_impact={"method": ["semi_empirical"]},
                plots=["surface_impact"],
                output_dir="/tmp/output",
            )

    def test_invalid_plot_type(self, event_data: dict, ceres_data: dict) -> None:
        with pytest.raises(ValidationError, match="Invalid plot types"):
            RadiativeConfig(
                event=event_data,
                ceres=ceres_data,
                plots=["toa_event_fields", "bogus_plot"],
                output_dir="/tmp/output",
            )
