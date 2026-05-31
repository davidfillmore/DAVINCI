"""Configuration module for DAVINCI.

This module provides Pydantic-based configuration validation,
YAML parsing, and migration utilities for MELODIES-MONET configs.
"""

from davinci_monet.config.migration import (
    CURRENT_VERSION,
    ConfigMigration,
    detect_config_version,
    migrate_config,
)
from davinci_monet.config.parser import (
    ConfigBuilder,
    config_to_yaml,
    dump_config,
    load_config,
    load_yaml,
    merge_configs,
    validate_config,
)
from davinci_monet.config.schema import (
    AnalysisConfig,
    Config,
    DataProcConfig,
    ModelConfig,
    MonetConfig,
    ObservationConfig,
    PlotGroupConfig,
    PlotStyleConfig,
    SourceConfig,
    SourcePairConfig,
    StatsConfig,
    VariableConfig,
)

__all__ = [
    # Schema classes
    "MonetConfig",
    "Config",
    "AnalysisConfig",
    "ModelConfig",
    "ObservationConfig",
    "SourceConfig",
    "SourcePairConfig",
    "PlotGroupConfig",
    "PlotStyleConfig",
    "StatsConfig",
    "VariableConfig",
    "DataProcConfig",
    # Parser functions
    "load_config",
    "load_yaml",
    "validate_config",
    "dump_config",
    "config_to_yaml",
    "merge_configs",
    "ConfigBuilder",
    # Migration
    "migrate_config",
    "detect_config_version",
    "ConfigMigration",
    "CURRENT_VERSION",
]
