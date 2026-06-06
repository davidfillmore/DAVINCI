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


LEGACY_SATELLITE_SOURCE_TYPES = {
    "satellite": "satellite_l2",
    "sat_swath_clm": "satellite_l2",
    "sat_grid_clm": "satellite_l3",
}


def _migrated_obs_source_type(label: str, entry: dict[str, Any]) -> Any:
    """Return the unified source type for a legacy observation entry."""
    obs_type = entry.pop("obs_type")
    sat_type = entry.get("sat_type")
    if sat_type is not None and str(sat_type).lower() == "modis_l2":
        raise ConfigurationError(
            f"Cannot migrate legacy MODIS L2 gridding source {label!r}. "
            "MODIS L2 gridding migration is unsupported and requires manual conversion."
        )
    if isinstance(obs_type, str):
        return LEGACY_SATELLITE_SOURCE_TYPES.get(obs_type.lower(), obs_type)
    return obs_type


def migrate_to_sources(config: dict[str, Any]) -> dict[str, Any]:
    """Migrate a legacy ``model:``/``obs:``/``pairs:`` config to the unified
    ``sources:``/``pairs:`` form (Phase 6).

    - Each ``model`` entry becomes a source with ``role: model`` and its
      ``mod_type`` renamed to ``type``.
    - Each ``obs`` entry becomes a source with ``role: obs`` and its
      ``obs_type`` renamed to ``type``.
    - Each legacy pair (``model``/``obs``/``variable: {model_var, obs_var}``)
      becomes a binary pair: ``sources: [model, obs]``, ``reference: <obs>``
      (preserving model→obs sampling), and a per-source ``variables`` map.
      Other pair keys (e.g. ``title``) are preserved.

    Operates on raw dicts so it round-trips cleanly through YAML. Idempotent:
    a config already in unified form is returned essentially unchanged.

    Parameters
    ----------
    config
        Legacy (or already-unified) configuration dictionary.

    Returns
    -------
    dict
        Configuration in the unified ``sources:`` form.
    """
    cfg = copy.deepcopy(config)
    sources: dict[str, Any] = dict(cfg.pop("sources", None) or {})

    for label, raw in (cfg.pop("model", None) or {}).items():
        entry = dict(raw) if isinstance(raw, dict) else {}
        new_entry: dict[str, Any] = {"role": "model"}
        if "mod_type" in entry:
            new_entry["type"] = entry.pop("mod_type")
        new_entry.update(entry)
        sources[label] = new_entry

    for label, raw in (cfg.pop("obs", None) or {}).items():
        entry = dict(raw) if isinstance(raw, dict) else {}
        new_entry = {"role": "obs"}
        if "obs_type" in entry:
            new_entry["type"] = _migrated_obs_source_type(label, entry)
        new_entry.update(entry)
        sources[label] = new_entry

    if sources:
        cfg["sources"] = sources

    pairs = cfg.get("pairs")
    if isinstance(pairs, dict):
        migrated_pairs: dict[str, Any] = {}
        for pname, p in pairs.items():
            if not isinstance(p, dict) or "model" not in p or "obs" not in p:
                migrated_pairs[pname] = p
                continue
            model_label = p["model"]
            obs_label = p["obs"]
            var = p.get("variable") or {}
            new_pair: dict[str, Any] = {
                "sources": [model_label, obs_label],
                "reference": obs_label,
            }
            vmap: dict[str, str] = {}
            if var.get("model_var"):
                vmap[model_label] = var["model_var"]
            if var.get("obs_var"):
                vmap[obs_label] = var["obs_var"]
            if vmap:
                new_pair["variables"] = vmap
            for k, v in p.items():
                if k not in ("model", "obs", "variable"):
                    new_pair[k] = v
            migrated_pairs[pname] = new_pair
        cfg["pairs"] = migrated_pairs

    return cfg


class LegacyConfigWarning(DeprecationWarning):
    """Emitted when a legacy ``model:``/``obs:`` config is used.

    The unified ``sources:`` schema is the going-forward format; ``model:``/
    ``obs:`` are deprecated and auto-converted internally. Subclasses
    ``DeprecationWarning`` (not ``UserWarning``) so it is informational and not
    escalated by strict warning filters. Run ``davinci-monet migrate-config`` to
    convert a control file.
    """


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
