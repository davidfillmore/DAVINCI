"""Configuration migration utilities.

This module provides tools for migrating MELODIES-MONET configurations
between versions and handling legacy format differences.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Callable

from davinci_monet.core.exceptions import ConfigurationError

# Version constants
CURRENT_VERSION = "2.0.0"
MINIMUM_VERSION = "1.0.0"


# Type alias for migration functions
MigrationFunc = Callable[[dict[str, Any]], dict[str, Any]]


class ConfigMigration:
    """Registry and executor for configuration migrations.

    Each migration function transforms a config from one version to the next.

    Examples
    --------
    >>> migration = ConfigMigration()
    >>> config = migration.migrate(old_config, from_version="1.0.0")
    """

    def __init__(self) -> None:
        """Initialize migration registry."""
        self._migrations: dict[str, list[tuple[str, MigrationFunc]]] = {}
        self._register_builtin_migrations()

    def register(
        self,
        from_version: str,
        to_version: str,
        func: MigrationFunc,
    ) -> None:
        """Register a migration function.

        Parameters
        ----------
        from_version
            Source version.
        to_version
            Target version.
        func
            Migration function.
        """
        if from_version not in self._migrations:
            self._migrations[from_version] = []
        self._migrations[from_version].append((to_version, func))

    def migrate(
        self,
        config: dict[str, Any],
        from_version: str | None = None,
        to_version: str = CURRENT_VERSION,
    ) -> dict[str, Any]:
        """Migrate configuration to target version.

        Parameters
        ----------
        config
            Configuration dictionary to migrate.
        from_version
            Source version. If None, auto-detected.
        to_version
            Target version.

        Returns
        -------
        dict[str, Any]
            Migrated configuration.
        """
        config = copy.deepcopy(config)

        if from_version is None:
            from_version = detect_config_version(config)

        current = from_version
        while current != to_version:
            if current not in self._migrations:
                break

            # Find migration to next version
            next_migration = None
            for target, func in self._migrations[current]:
                if self._version_leq(target, to_version):
                    if next_migration is None or self._version_gt(target, next_migration[0]):
                        next_migration = (target, func)

            if next_migration is None:
                break

            target, func = next_migration
            config = func(config)
            current = target

        # Set version in config
        if "version" not in config:
            config["version"] = to_version

        return config

    def _version_leq(self, v1: str, v2: str) -> bool:
        """Check if v1 <= v2."""
        return self._parse_version(v1) <= self._parse_version(v2)

    def _version_gt(self, v1: str, v2: str) -> bool:
        """Check if v1 > v2."""
        return self._parse_version(v1) > self._parse_version(v2)

    def _parse_version(self, version: str) -> tuple[int, ...]:
        """Parse version string to tuple."""
        try:
            return tuple(int(x) for x in version.split("."))
        except ValueError:
            return (0, 0, 0)

    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # Migration from legacy (no version) to 1.0.0
        self.register("0.0.0", "1.0.0", _migrate_legacy_to_v1)

        # Migration from 1.0.0 to 2.0.0
        self.register("1.0.0", "2.0.0", _migrate_v1_to_v2)


def detect_config_version(config: dict[str, Any]) -> str:
    """Detect configuration version from content.

    Parameters
    ----------
    config
        Configuration dictionary.

    Returns
    -------
    str
        Detected version string.
    """
    # Check for explicit version field
    if "version" in config:
        return str(config["version"])

    # Heuristics for detecting legacy configs
    # Legacy configs don't have explicit version fields

    # Check for v2 features
    if _has_v2_features(config):
        return "2.0.0"

    # Check for v1 features (MELODIES-MONET original format)
    if _is_v1_format(config):
        return "1.0.0"

    # Default to legacy (no version)
    return "0.0.0"


def _has_v2_features(config: dict[str, Any]) -> bool:
    """Check if config has v2-specific features."""
    # V2 specific markers could include new config keys
    # For now, assume explicit version field is the marker
    return "version" in config and str(config["version"]).startswith("2")


def _is_v1_format(config: dict[str, Any]) -> bool:
    """Check if config follows v1 (MELODIES-MONET) format."""
    # V1 format has analysis, model, obs sections
    has_sections = any(key in config for key in ["analysis", "model", "obs", "plots", "stats"])
    return has_sections


# =============================================================================
# Migration Functions
# =============================================================================


def _migrate_legacy_to_v1(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy config (pre-versioning) to v1.0.0.

    This handles very old config formats that may have different structure.
    """
    # Ensure all required sections exist
    for section in ["analysis", "model", "obs"]:
        if section not in config:
            config[section] = {}

    # Normalize observation type names
    if "obs" in config:
        for obs_name, obs_config in config["obs"].items():
            if isinstance(obs_config, dict):
                obs_type = obs_config.get("obs_type", "")
                # Normalize legacy obs_type values
                type_mapping = {
                    "surface": "pt_sfc",
                    "point": "pt_sfc",
                    "airnow": "pt_sfc",
                    "aircraft": "aircraft",
                    "flight": "aircraft",
                    "satellite": "sat_swath_clm",
                }
                if obs_type.lower() in type_mapping:
                    obs_config["obs_type"] = type_mapping[obs_type.lower()]

    config["version"] = "1.0.0"
    return config


def _migrate_v1_to_v2(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate v1.0.0 config to v2.0.0.

    V2 changes:
    - Standardized field names
    - Improved validation
    - New optional fields
    """
    # Model section updates
    if "model" in config and isinstance(config["model"], dict):
        for model_name, model_config in config["model"].items():
            if isinstance(model_config, dict):
                # Ensure mod_type is set
                if "mod_type" not in model_config:
                    # Try to infer from model name
                    name_lower = model_name.lower()
                    if "cmaq" in name_lower:
                        model_config["mod_type"] = "cmaq"
                    elif "wrf" in name_lower:
                        model_config["mod_type"] = "wrfchem"
                    elif "ufs" in name_lower or "rrfs" in name_lower:
                        model_config["mod_type"] = "ufs"

    # Observation section updates
    if "obs" in config and isinstance(config["obs"], dict):
        for obs_name, obs_config in config["obs"].items():
            if isinstance(obs_config, dict):
                # Convert use_airnow flag to explicit obs_type
                if obs_config.get("use_airnow") and "obs_type" not in obs_config:
                    obs_config["obs_type"] = "pt_sfc"

    # Stats section updates
    if "stats" in config and isinstance(config["stats"], dict):
        stats = config["stats"]
        # Ensure stat_list is a list
        if "stat_list" in stats and isinstance(stats["stat_list"], str):
            stats["stat_list"] = [stats["stat_list"]]

    config["version"] = "2.0.0"
    return config


# =============================================================================
# Validation and Checking
# =============================================================================


def check_deprecated_fields(config: dict[str, Any]) -> list[str]:
    """Check for deprecated fields in configuration.

    Parameters
    ----------
    config
        Configuration dictionary.

    Returns
    -------
    list[str]
        List of deprecation warnings.
    """
    warnings_list: list[str] = []

    # Check for deprecated observation flags
    if "obs" in config and isinstance(config["obs"], dict):
        for obs_name, obs_config in config["obs"].items():
            if isinstance(obs_config, dict):
                if "use_airnow" in obs_config:
                    warnings_list.append(
                        f"obs.{obs_name}.use_airnow is deprecated. "
                        "Use obs_type='pt_sfc' instead."
                    )

    # Check for deprecated model fields
    if "model" in config and isinstance(config["model"], dict):
        for model_name, model_config in config["model"].items():
            if isinstance(model_config, dict):
                if "projection" in model_config and model_config["projection"] is None:
                    warnings_list.append(
                        f"model.{model_name}.projection=null is deprecated. "
                        "Omit the field instead."
                    )

    return warnings_list


def emit_deprecation_warnings(config: dict[str, Any]) -> None:
    """Emit deprecation warnings for config.

    Parameters
    ----------
    config
        Configuration dictionary.
    """
    for warning in check_deprecated_fields(config):
        warnings.warn(warning, DeprecationWarning, stacklevel=2)


def validate_version_compatibility(
    config: dict[str, Any],
    min_version: str = MINIMUM_VERSION,
    max_version: str = CURRENT_VERSION,
) -> None:
    """Validate configuration version is within supported range.

    Parameters
    ----------
    config
        Configuration dictionary.
    min_version
        Minimum supported version.
    max_version
        Maximum supported version.

    Raises
    ------
    ConfigurationError
        If version is outside supported range.
    """
    version = detect_config_version(config)

    def parse_version(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in v.split("."))
        except ValueError:
            return (0, 0, 0)

    v = parse_version(version)
    min_v = parse_version(min_version)
    max_v = parse_version(max_version)

    if v < min_v:
        raise ConfigurationError(
            f"Configuration version {version} is too old. "
            f"Minimum supported version is {min_version}."
        )

    if v > max_v:
        raise ConfigurationError(
            f"Configuration version {version} is newer than supported. "
            f"Maximum supported version is {max_version}."
        )


# Singleton instance for convenient access
_default_migration = ConfigMigration()


def migrate_config(
    config: dict[str, Any],
    from_version: str | None = None,
    to_version: str = CURRENT_VERSION,
) -> dict[str, Any]:
    """Migrate configuration using default migration registry.

    Parameters
    ----------
    config
        Configuration dictionary.
    from_version
        Source version. If None, auto-detected.
    to_version
        Target version.

    Returns
    -------
    dict[str, Any]
        Migrated configuration.
    """
    return _default_migration.migrate(config, from_version, to_version)
