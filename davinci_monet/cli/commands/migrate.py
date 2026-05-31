"""CLI command: migrate a legacy model/obs config to the unified sources form."""

from __future__ import annotations

from pathlib import Path

import yaml

from davinci_monet.config.migration import migrate_to_sources
from davinci_monet.config.parser import load_yaml


def migrate_config_command(control: str, output: str) -> None:
    """Read a legacy control file, migrate it to the unified ``sources:`` form,
    and write the result to ``output`` as YAML.

    Parameters
    ----------
    control
        Path to the legacy (model:/obs:) control file.
    output
        Path to write the migrated (sources:) control file.
    """
    data = load_yaml(control)
    migrated = migrate_to_sources(data)
    out_path = Path(output)
    with open(out_path, "w") as f:
        yaml.dump(migrated, f, default_flow_style=False, sort_keys=False)
    n_sources = len(migrated.get("sources", {}) or {})
    print(f"Migrated config written to {out_path} ({n_sources} sources)")
