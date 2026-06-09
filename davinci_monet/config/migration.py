"""Legacy configuration conversion utilities.

Converts legacy MELODIES-MONET ``model:``/``obs:``/``pairs:`` control files to
the unified ``sources:``/``pairs:`` form. This is the supported legacy off-ramp
(``davinci-monet migrate-config``).

There is intentionally no runtime version-migration engine: the standard
pipeline rejects legacy ``model:``/``obs:`` configs at parse time and points the
user at ``migrate-config``, so configs are converted explicitly, not silently
upgraded between schema versions.
"""

from __future__ import annotations

import copy
from typing import Any

from davinci_monet.core.exceptions import ConfigurationError

LEGACY_SATELLITE_SOURCE_TYPES = {
    "satellite": "satellite_l2",
    "sat_swath_clm": "satellite_l2",
    "sat_grid_clm": "satellite_l3",
}

# Legacy geometry-only obs_type aliases -> their registered generic reader type.
# These mirror the legacy obs_type->geometry mapping: the old loader stage opened
# a plain NetCDF and set geometry from the obs_type string. The unified path loads
# through a registered reader, so map each alias to the generic reader of the same
# geometry (registered in observations/base.py). Names that are themselves
# registered reader types (e.g. ``aircraft``, ``profile``, ``gridded``,
# ``pt_sfc``, ``ozonesonde``) pass through unchanged.
LEGACY_GEOMETRY_SOURCE_TYPES = {
    "surface": "pt_sfc",
    "ground": "pt_sfc",
    "point": "pt_sfc",
    "mobile": "aircraft",
    "ship": "aircraft",
    "track": "aircraft",
    "sonde": "profile",
    "grid": "gridded",
    "reanalysis": "gridded",
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
        key = obs_type.lower()
        if key in LEGACY_SATELLITE_SOURCE_TYPES:
            return LEGACY_SATELLITE_SOURCE_TYPES[key]
        return LEGACY_GEOMETRY_SOURCE_TYPES.get(key, obs_type)
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
        # Keys this function derives from the legacy model/obs/variable fields.
        # They must NOT be overwritten by schema-default values when copying
        # through "other" pair keys (a model_dump()'d legacy pair carries empty
        # ``sources: []`` / ``reference: None`` / ``variables: {}`` defaults
        # alongside the legacy keys).
        _derived = ("sources", "reference", "variables")
        for pname, p in pairs.items():
            if not isinstance(p, dict):
                migrated_pairs[pname] = p
                continue
            # Already unified: a truthy ``sources`` means this pair is in the new
            # form; pass it through (dropping any empty legacy-key defaults).
            if p.get("sources"):
                migrated_pairs[pname] = {
                    k: v for k, v in p.items() if k not in ("model", "obs", "variable")
                }
                continue
            if not p.get("model") or not p.get("obs"):
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
                if k not in ("model", "obs", "variable") and k not in _derived:
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
