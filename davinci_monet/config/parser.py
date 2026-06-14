"""YAML configuration parser for the ``sources:``/``pairs:`` schema.

This module loads and parses DAVINCI YAML configuration files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TextIO

import yaml

from davinci_monet.config.schema import MonetConfig
from davinci_monet.core.exceptions import ConfigurationError
from davinci_monet.core.schema_utils import dump_schema, validate_schema


def load_yaml(source: str | Path | TextIO) -> dict[str, Any]:
    """Load raw YAML from file or string.

    Parameters
    ----------
    source
        File path, file object, or YAML string.

    Returns
    -------
    dict[str, Any]
        Raw YAML data as dictionary.

    Raises
    ------
    ConfigurationError
        If YAML cannot be parsed or file not found.

    Examples
    --------
    >>> data = load_yaml("config.yaml")
    >>> data = load_yaml(Path("config.yaml"))
    >>> data = load_yaml("analysis:\\n  debug: true")
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists():
                with open(path) as f:
                    data = yaml.safe_load(f)
            else:
                # Try parsing as YAML string
                data = yaml.safe_load(str(source))
        else:
            # File-like object
            data = yaml.safe_load(source)

        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"YAML root must be a mapping, got {type(data)}")
        return data

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML: {e}") from e
    except FileNotFoundError as e:
        raise ConfigurationError(f"Configuration file not found: {source}") from e
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}") from e


def expand_env_vars(data: dict[str, Any]) -> dict[str, Any]:
    """Expand environment variables in string values.

    Supports ${VAR} and $VAR syntax.

    Parameters
    ----------
    data
        Dictionary to process.

    Returns
    -------
    dict[str, Any]
        Dictionary with environment variables expanded.

    Examples
    --------
    >>> os.environ["MY_PATH"] = "/data"
    >>> expand_env_vars({"output_dir": "${MY_PATH}/output"})
    {'output_dir': '/data/output'}
    """

    def _expand(value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_expand(item) for item in value]
        return value

    result: dict[str, Any] = _expand(data)
    return result


def preprocess_config(data: dict[str, Any]) -> dict[str, Any]:
    """Preprocess raw YAML data.

    Parameters
    ----------
    data
        Raw YAML dictionary.

    Returns
    -------
    dict[str, Any]
        Preprocessed dictionary ready for validation.
    """
    # Expand environment variables
    data = expand_env_vars(data)

    # Normalize analysis time fields
    if "analysis" in data and data["analysis"]:
        analysis = data["analysis"]
        # Some configs have times as comments or nulled out
        for key in ["start_time", "end_time"]:
            if key in analysis and analysis[key] is None:
                del analysis[key]
        # Preserve whether end_time had an explicit time component for inclusive filtering
        end_time = analysis.get("end_time")
        if isinstance(end_time, str) and "_end_time_has_time" not in analysis:
            import re

            analysis["_end_time_has_time"] = bool(re.search(r"\d{2}:\d{2}", end_time))

    # Ensure all sections exist as dicts (not None)
    for section in ["sources", "pairs", "plots"]:
        if section not in data or data[section] is None:
            data[section] = {}

    return data


def load_config(source: str | Path | TextIO) -> MonetConfig:
    """Load and validate a MELODIES-MONET configuration.

    This is the main entry point for loading configuration files.
    It handles YAML parsing, preprocessing, and Pydantic validation.

    Parameters
    ----------
    source
        File path, file object, or YAML string.

    Returns
    -------
    MonetConfig
        Validated configuration object.

    Raises
    ------
    ConfigurationError
        If configuration is invalid.

    Examples
    --------
    >>> config = load_config("control.yaml")
    >>> print(config.analysis.output_dir)
    /path/to/output

    >>> config = load_config(Path("control.yaml"))

    >>> yaml_str = '''
    ... analysis:
    ...   start_time: '2024-01-01'
    ...   end_time: '2024-01-02'
    ... '''
    >>> config = load_config(yaml_str)
    """
    # Load raw YAML
    data = load_yaml(source)

    # Preprocess before schema validation.
    data = preprocess_config(data)

    # Validate with Pydantic
    try:
        return validate_schema(MonetConfig, data)
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


def validate_config(data: dict[str, Any]) -> MonetConfig:
    """Validate a dictionary as a MELODIES-MONET configuration.

    Parameters
    ----------
    data
        Dictionary to validate.

    Returns
    -------
    MonetConfig
        Validated configuration object.

    Raises
    ------
    ConfigurationError
        If validation fails.
    """
    data = preprocess_config(data)
    try:
        return validate_schema(MonetConfig, data)
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e


def dump_config(config: MonetConfig, path: str | Path) -> None:
    """Write configuration to YAML file.

    Parameters
    ----------
    config
        Configuration to write.
    path
        Output file path.
    """
    data = dump_schema(config, exclude_none=True, exclude_unset=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def config_to_yaml(config: MonetConfig) -> str:
    """Convert configuration to YAML string.

    Parameters
    ----------
    config
        Configuration object.

    Returns
    -------
    str
        YAML string representation.
    """
    data = dump_schema(config, exclude_none=True, exclude_unset=True)
    result: str = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return result


def merge_configs(*configs: MonetConfig | dict[str, Any]) -> MonetConfig:
    """Merge multiple configurations, later ones override earlier.

    Parameters
    ----------
    *configs
        Configuration objects or dictionaries to merge.

    Returns
    -------
    MonetConfig
        Merged configuration.

    Examples
    --------
    >>> base = load_config("base.yaml")
    >>> override = {"analysis": {"debug": True}}
    >>> merged = merge_configs(base, override)
    """
    result: dict[str, Any] = {}

    for config in configs:
        if isinstance(config, MonetConfig):
            data = dump_schema(config, exclude_none=True)
        else:
            data = config

        result = _deep_merge(result, data)

    return validate_config(result)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Parameters
    ----------
    base
        Base dictionary.
    override
        Dictionary to merge in (takes precedence).

    Returns
    -------
    dict[str, Any]
        Merged dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


class ConfigBuilder:
    """Builder pattern for constructing configurations programmatically.

    Examples
    --------
    >>> config = (ConfigBuilder()
    ...     .set_analysis(start_time="2024-01-01", end_time="2024-01-02")
    ...     .add_source("cam", type="cesm_fv", files="/data/*.nc")
    ...     .add_source("airnow", type="pt_sfc")
    ...     .build())
    """

    def __init__(self) -> None:
        """Initialize empty configuration."""
        self._data: dict[str, Any] = {
            "analysis": {},
            "sources": {},
            "pairs": {},
            "plots": {},
        }

    def set_analysis(self, **kwargs: Any) -> "ConfigBuilder":
        """Set analysis configuration.

        Parameters
        ----------
        **kwargs
            Analysis parameters (start_time, end_time, output_dir, debug).

        Returns
        -------
        ConfigBuilder
            Self for chaining.
        """
        self._data["analysis"].update(kwargs)
        return self

    def add_source(self, name: str, **kwargs: Any) -> "ConfigBuilder":
        """Add a unified source configuration."""
        self._data["sources"][name] = kwargs
        return self

    def add_pair(
        self,
        name: str,
        x: dict[str, str],
        y: dict[str, str],
        **kwargs: Any,
    ) -> "ConfigBuilder":
        """Add a binary pair as an ordered ``(x, y)``.

        Each axis is ``{"source": <label>, "variable": <var>}``. ``x`` is the
        horizontal/reference axis; ``y`` is vertical (diffs are ``y - x``).
        """
        pair: dict[str, Any] = {"x": dict(x), "y": dict(y)}
        pair.update(kwargs)
        self._data["pairs"][name] = pair
        return self

    def add_plot(self, name: str, plot_type: str, **kwargs: Any) -> "ConfigBuilder":
        """Add a plot configuration.

        Parameters
        ----------
        name
            Plot group name.
        plot_type
            Type of plot (timeseries, taylor, etc.).
        **kwargs
            Plot parameters.

        Returns
        -------
        ConfigBuilder
            Self for chaining.
        """
        self._data["plots"][name] = {"type": plot_type, **kwargs}
        return self

    def set_stats(self, **kwargs: Any) -> "ConfigBuilder":
        """Set statistics configuration.

        Parameters
        ----------
        **kwargs
            Statistics parameters.

        Returns
        -------
        ConfigBuilder
            Self for chaining.
        """
        self._data["stats"] = kwargs
        return self

    def build(self) -> MonetConfig:
        """Build and validate the configuration.

        Returns
        -------
        MonetConfig
            Validated configuration.
        """
        return validate_config(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Get raw dictionary representation.

        Returns
        -------
        dict[str, Any]
            Configuration as dictionary.
        """
        return self._data.copy()
