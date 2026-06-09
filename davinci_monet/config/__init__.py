"""Configuration module for DAVINCI.

This module provides Pydantic-based configuration validation,
YAML parsing, and migration utilities for MELODIES-MONET configs.
"""

from davinci_monet.config.migration import (
    LegacyConfigWarning,
    migrate_to_sources,
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
    MonetConfig,
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
    # Legacy -> unified conversion
    "migrate_to_sources",
    "LegacyConfigWarning",
]
