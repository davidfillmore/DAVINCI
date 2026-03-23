"""Tests for configuration schema (Pydantic models)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from davinci_monet.config.schema import (
    AnalysisConfig,
    DataProcConfig,
    ModelConfig,
    MonetConfig,
    ObservationConfig,
    PlotGroupConfig,
    PlotStyleConfig,
    StatsConfig,
    VariableConfig,
)


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = AnalysisConfig()
        assert config.start_time is None
        assert config.end_time is None
        assert config.debug is False

    def test_datetime_parsing_standard(self) -> None:
        """Test standard datetime format parsing."""
        config = AnalysisConfig(start_time="2024-01-01", end_time="2024-01-02")
        assert config.start_time == datetime(2024, 1, 1)
        assert config.end_time == datetime(2024, 1, 2)

    def test_datetime_parsing_melodies_format(self) -> None:
        """Test MELODIES-MONET datetime format."""
        config = AnalysisConfig(start_time="2019-08-02-12:00:00")
        assert config.start_time == datetime(2019, 8, 2, 12, 0, 0)

    def test_datetime_parsing_iso_format(self) -> None:
        """Test ISO datetime format."""
        config = AnalysisConfig(start_time="2024-01-01T14:30:00")
        assert config.start_time == datetime(2024, 1, 1, 14, 30, 0)

    def test_output_dir_path_conversion(self) -> None:
        """Test output_dir is converted to Path."""
        config = AnalysisConfig(output_dir="/path/to/output")
        assert config.output_dir == Path("/path/to/output")

    def test_invalid_datetime_raises(self) -> None:
        """Test invalid datetime raises ValueError."""
        with pytest.raises(ValueError):
            AnalysisConfig(start_time="not-a-date")

    def test_style_config_none(self) -> None:
        """Test style config defaults to None."""
        config = AnalysisConfig()
        assert config.style is None

    def test_style_config_from_dict(self) -> None:
        """Test style config parsed from dict."""
        config = AnalysisConfig(style={"theme": "ncar", "context": "presentation"})
        assert config.style is not None
        assert config.style.theme == "ncar"
        assert config.style.context == "presentation"

    def test_style_config_object(self) -> None:
        """Test style config as PlotStyleConfig object."""
        style = PlotStyleConfig(theme="ncar", context="publication")
        config = AnalysisConfig(style=style)
        assert config.style.theme == "ncar"
        assert config.style.context == "publication"


class TestPlotStyleConfig:
    """Tests for PlotStyleConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = PlotStyleConfig()
        assert config.theme is None
        assert config.context == "default"
        assert config.use_seaborn is True
        assert config.seaborn_style == "whitegrid"

    def test_ncar_theme(self) -> None:
        """Test NCAR theme configuration."""
        config = PlotStyleConfig(theme="ncar", context="presentation")
        assert config.theme == "ncar"
        assert config.context == "presentation"

    def test_default_theme(self) -> None:
        """Test explicit default theme."""
        config = PlotStyleConfig(theme="default")
        assert config.theme == "default"

    def test_seaborn_options(self) -> None:
        """Test seaborn configuration options."""
        config = PlotStyleConfig(
            use_seaborn=False,
            seaborn_style="darkgrid",
        )
        assert config.use_seaborn is False
        assert config.seaborn_style == "darkgrid"


class TestVariableConfig:
    """Tests for VariableConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = VariableConfig()
        assert config.unit_scale == 1.0
        assert config.unit_scale_method == "*"

    def test_all_fields(self) -> None:
        """Test all fields can be set."""
        config = VariableConfig(
            unit_scale=1000.0,
            unit_scale_method="+",
            obs_min=0.0,
            obs_max=100.0,
            nan_value=-1.0,
            rename="new_name",
            ylabel_plot="Label",
            vmin_plot=0.0,
            vmax_plot=50.0,
            vdiff_plot=10.0,
            nlevels_plot=20,
        )
        assert config.unit_scale == 1000.0
        assert config.unit_scale_method == "+"
        assert config.nan_value == -1.0


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ModelConfig()
        assert config.radius_of_influence == 12000.0
        assert config.mapping == {}
        assert config.variables == {}

    def test_mod_type(self) -> None:
        """Test mod_type field."""
        config = ModelConfig(mod_type="cmaq")
        assert config.mod_type == "cmaq"

    def test_files_kept_as_string(self) -> None:
        """Test files are kept as strings for glob patterns."""
        config = ModelConfig(files="/path/to/*.nc")
        assert config.files == "/path/to/*.nc"

    def test_mapping(self) -> None:
        """Test variable mapping."""
        config = ModelConfig(mapping={"airnow": {"O3": "OZONE", "PM25": "PM2.5"}})
        assert config.mapping["airnow"]["O3"] == "OZONE"

    def test_variables_parsing(self) -> None:
        """Test variables are parsed as VariableConfig."""
        config = ModelConfig.model_validate(
            {"variables": {"co": {"unit_scale": 1000.0, "rename": "CO"}}}
        )
        assert config.variables["co"].unit_scale == 1000.0
        assert config.variables["co"].rename == "CO"


class TestObservationConfig:
    """Tests for ObservationConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ObservationConfig()
        assert config.filename is None
        assert config.obs_type is None
        assert config.variables == {}

    def test_obs_type(self) -> None:
        """Test obs_type values."""
        config = ObservationConfig(obs_type="pt_sfc")
        assert config.obs_type == "pt_sfc"

        config = ObservationConfig(obs_type="aircraft")
        assert config.obs_type == "aircraft"

    def test_variables_parsing(self) -> None:
        """Test variables are parsed correctly."""
        config = ObservationConfig.model_validate(
            {
                "variables": {
                    "O3": {"unit_scale": 1.0, "nan_value": -1.0},
                    "PM25": {"ylabel_plot": "PM2.5 (ug/m3)"},
                }
            }
        )
        assert config.variables["O3"].nan_value == -1.0
        assert config.variables["PM25"].ylabel_plot == "PM2.5 (ug/m3)"


class TestDataProcConfig:
    """Tests for DataProcConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = DataProcConfig()
        assert config.rem_obs_nan is True
        assert config.ts_select_time == "time"
        assert config.set_axis is False

    def test_all_fields(self) -> None:
        """Test all fields."""
        config = DataProcConfig(
            rem_obs_nan=False,
            ts_select_time="time_local",
            ts_avg_window="h",
            set_axis=True,
        )
        assert config.rem_obs_nan is False
        assert config.ts_select_time == "time_local"


class TestPlotGroupConfig:
    """Tests for PlotGroupConfig."""

    def test_required_type(self) -> None:
        """Test type is required."""
        config = PlotGroupConfig(type="timeseries")
        assert config.type == "timeseries"

    def test_default_domain(self) -> None:
        """Test default domain settings."""
        config = PlotGroupConfig(type="taylor")
        assert config.domain_type == ["all"]
        assert config.domain_name == ["CONUS"]

    def test_data_list(self) -> None:
        """Test data list."""
        config = PlotGroupConfig(
            type="spatial_bias",
            data=["airnow_cmaq", "airnow_wrfchem"],
        )
        assert len(config.data) == 2

    def test_data_proc_parsing(self) -> None:
        """Test data_proc is parsed correctly."""
        config = PlotGroupConfig.model_validate(
            {
                "type": "boxplot",
                "data_proc": {"rem_obs_nan": False, "set_axis": True},
            }
        )
        assert isinstance(config.data_proc, DataProcConfig)
        assert config.data_proc.rem_obs_nan is False


class TestStatsConfig:
    """Tests for StatsConfig."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = StatsConfig()
        assert "MB" in config.stat_list
        assert config.round_output == 3
        assert config.output_table is False

    def test_stat_list(self) -> None:
        """Test custom stat list."""
        config = StatsConfig(stat_list=["MB", "RMSE", "R2"])
        assert config.stat_list == ["MB", "RMSE", "R2"]


class TestMonetConfig:
    """Tests for root MonetConfig."""

    def test_empty_config(self) -> None:
        """Test empty config is valid."""
        config = MonetConfig()
        assert config.model == {}
        assert config.obs == {}
        assert config.plots == {}

    def test_analysis_section(self) -> None:
        """Test analysis section parsing."""
        config = MonetConfig.model_validate(
            {
                "analysis": {
                    "start_time": "2024-01-01",
                    "end_time": "2024-01-02",
                    "debug": True,
                }
            }
        )
        assert config.analysis.debug is True
        assert config.analysis.start_time == datetime(2024, 1, 1)

    def test_model_section(self) -> None:
        """Test model section parsing."""
        config = MonetConfig.model_validate(
            {
                "model": {
                    "cmaq_test": {"mod_type": "cmaq", "files": "/data/*.nc"},
                    "wrf_test": {"mod_type": "wrfchem"},
                }
            }
        )
        assert "cmaq_test" in config.model
        assert config.model["cmaq_test"].mod_type == "cmaq"
        assert config.model["wrf_test"].mod_type == "wrfchem"

    def test_obs_section(self) -> None:
        """Test observation section parsing."""
        config = MonetConfig.model_validate(
            {
                "obs": {
                    "airnow": {"obs_type": "pt_sfc", "filename": "/data/airnow.nc"},
                }
            }
        )
        assert "airnow" in config.obs
        assert config.obs["airnow"].obs_type == "pt_sfc"

    def test_plots_section(self) -> None:
        """Test plots section parsing."""
        config = MonetConfig.model_validate(
            {
                "plots": {
                    "plot_grp1": {
                        "type": "timeseries",
                        "data": ["airnow_cmaq"],
                    },
                }
            }
        )
        assert "plot_grp1" in config.plots
        assert config.plots["plot_grp1"].type == "timeseries"

    def test_stats_section(self) -> None:
        """Test stats section parsing."""
        config = MonetConfig.model_validate(
            {"stats": {"stat_list": ["MB", "R2"], "round_output": 2}}
        )
        assert config.stats is not None
        assert config.stats.round_output == 2

    def test_get_model_obs_pairs(self) -> None:
        """Test extracting model-observation pairs."""
        config = MonetConfig.model_validate(
            {
                "model": {
                    "cmaq": {"mapping": {"airnow": {"O3": "OZONE"}}},
                    "wrf": {"mapping": {"airnow": {"O3": "O3"}}},
                },
                "obs": {"airnow": {}},
            }
        )
        pairs = config.get_model_obs_pairs()
        assert ("airnow", "cmaq") in pairs
        assert ("airnow", "wrf") in pairs

    def test_get_model_obs_pairs_from_plot_refs(self) -> None:
        """Test extracting pairs from plot references (model_obs and obs_model)."""
        config = MonetConfig.model_validate(
            {
                "model": {"cmaq": {}},
                "obs": {"airnow": {}},
                "plots": {
                    "p1": {"type": "timeseries", "data": ["cmaq_airnow", "airnow_cmaq"]},
                },
            }
        )
        pairs = config.get_model_obs_pairs()
        assert ("airnow", "cmaq") in pairs

    def test_full_config(self) -> None:
        """Test full configuration."""
        config = MonetConfig.model_validate(
            {
                "analysis": {
                    "start_time": "2024-01-01",
                    "end_time": "2024-01-02",
                    "output_dir": "/output",
                    "debug": True,
                },
                "model": {
                    "cmaq": {
                        "files": "/data/cmaq/*.nc",
                        "mod_type": "cmaq",
                        "radius_of_influence": 15000,
                        "mapping": {"airnow": {"O3": "OZONE"}},
                    }
                },
                "obs": {
                    "airnow": {
                        "filename": "/data/airnow.nc",
                        "obs_type": "pt_sfc",
                        "variables": {"OZONE": {"nan_value": -1.0}},
                    }
                },
                "plots": {
                    "timeseries": {
                        "type": "timeseries",
                        "data": ["airnow_cmaq"],
                        "domain_type": ["all"],
                    }
                },
                "stats": {"stat_list": ["MB", "RMSE"]},
            }
        )

        assert config.analysis.debug is True
        assert config.model["cmaq"].radius_of_influence == 15000
        assert config.obs["airnow"].variables["OZONE"].nan_value == -1.0


class TestExtraFieldsHandling:
    """Tests for handling extra/unknown fields."""

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed (backward compatibility)."""
        config = MonetConfig.model_validate(
            {
                "analysis": {"unknown_field": "value"},
                "model": {"cmaq": {"custom_option": True}},
            }
        )
        # Should not raise - extra fields allowed
        extra = config.analysis.model_extra
        assert extra is not None
        assert extra.get("unknown_field") == "value"
