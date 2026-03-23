"""Tests for configuration migration."""

from __future__ import annotations

from typing import Any

import pytest

from davinci_monet.config.migration import (
    CURRENT_VERSION,
    ConfigMigration,
    check_deprecated_fields,
    detect_config_version,
    migrate_config,
    validate_version_compatibility,
)
from davinci_monet.core.exceptions import ConfigurationError


class TestDetectConfigVersion:
    """Tests for detect_config_version function."""

    def test_explicit_version(self) -> None:
        """Test detecting explicit version."""
        config: dict[str, Any] = {"version": "2.0.0"}
        assert detect_config_version(config) == "2.0.0"

    def test_v1_format_detection(self) -> None:
        """Test detecting v1 format from structure."""
        config: dict[str, Any] = {
            "analysis": {"start_time": "2024-01-01"},
            "model": {},
            "obs": {},
        }
        assert detect_config_version(config) == "1.0.0"

    def test_legacy_format_detection(self) -> None:
        """Test detecting legacy format."""
        config: dict[str, Any] = {"some_other_key": "value"}
        assert detect_config_version(config) == "0.0.0"


class TestConfigMigration:
    """Tests for ConfigMigration class."""

    def test_migration_registry(self) -> None:
        """Test migration registry."""
        migration = ConfigMigration()
        # Should have built-in migrations registered
        assert "0.0.0" in migration._migrations
        assert "1.0.0" in migration._migrations

    def test_migrate_legacy_to_v1(self) -> None:
        """Test migrating legacy to v1."""
        config: dict[str, Any] = {"some_field": "value"}
        result = migrate_config(config, from_version="0.0.0", to_version="1.0.0")
        assert result["version"] == "1.0.0"
        assert "analysis" in result
        assert "model" in result
        assert "obs" in result

    def test_migrate_v1_to_v2(self) -> None:
        """Test migrating v1 to v2."""
        config: dict[str, Any] = {
            "analysis": {},
            "model": {"cmaq_test": {}},
            "obs": {"airnow": {"use_airnow": True}},
        }
        result = migrate_config(config, from_version="1.0.0", to_version="2.0.0")
        assert result["version"] == "2.0.0"

    def test_migrate_auto_detect(self) -> None:
        """Test migration with auto-detected version."""
        config: dict[str, Any] = {
            "analysis": {"start_time": "2024-01-01"},
            "model": {},
            "obs": {},
        }
        result = migrate_config(config, to_version="2.0.0")
        assert result["version"] == "2.0.0"

    def test_migrate_preserves_data(self) -> None:
        """Test migration preserves existing data."""
        config: dict[str, Any] = {
            "analysis": {"debug": True, "output_dir": "/output"},
            "model": {"cmaq": {"files": "/data/*.nc"}},
            "obs": {},
        }
        result = migrate_config(config, from_version="1.0.0")
        assert result["analysis"]["debug"] is True
        assert result["analysis"]["output_dir"] == "/output"
        assert result["model"]["cmaq"]["files"] == "/data/*.nc"


class TestMigrateConfig:
    """Tests for migrate_config convenience function."""

    def test_full_migration(self) -> None:
        """Test full migration from legacy to current."""
        config: dict[str, Any] = {}
        result = migrate_config(config)
        assert result["version"] == CURRENT_VERSION

    def test_no_change_when_current(self) -> None:
        """Test no change when already at current version."""
        config: dict[str, Any] = {
            "version": CURRENT_VERSION,
            "analysis": {"debug": True},
        }
        result = migrate_config(config)
        assert result["analysis"]["debug"] is True


class TestCheckDeprecatedFields:
    """Tests for check_deprecated_fields function."""

    def test_use_airnow_deprecated(self) -> None:
        """Test use_airnow deprecation warning."""
        config: dict[str, Any] = {"obs": {"airnow": {"use_airnow": True}}}
        warnings = check_deprecated_fields(config)
        assert len(warnings) == 1
        assert "use_airnow" in warnings[0]

    def test_null_projection_deprecated(self) -> None:
        """Test null projection deprecation warning."""
        config: dict[str, Any] = {"model": {"cmaq": {"projection": None}}}
        warnings = check_deprecated_fields(config)
        assert len(warnings) == 1
        assert "projection" in warnings[0]

    def test_no_deprecations(self) -> None:
        """Test no warnings for clean config."""
        config: dict[str, Any] = {
            "analysis": {},
            "model": {"cmaq": {"mod_type": "cmaq"}},
            "obs": {"airnow": {"obs_type": "pt_sfc"}},
        }
        warnings = check_deprecated_fields(config)
        assert len(warnings) == 0


class TestValidateVersionCompatibility:
    """Tests for validate_version_compatibility function."""

    def test_current_version_valid(self) -> None:
        """Test current version is valid."""
        config: dict[str, Any] = {"version": CURRENT_VERSION}
        # Should not raise
        validate_version_compatibility(config)

    def test_old_version_invalid(self) -> None:
        """Test too-old version raises error."""
        config: dict[str, Any] = {"version": "0.0.1"}
        with pytest.raises(ConfigurationError, match="too old"):
            validate_version_compatibility(config, min_version="1.0.0")

    def test_future_version_invalid(self) -> None:
        """Test too-new version raises error."""
        config: dict[str, Any] = {"version": "99.0.0"}
        with pytest.raises(ConfigurationError, match="newer than supported"):
            validate_version_compatibility(config, max_version="3.0.0")


class TestObsTypeMigration:
    """Tests for observation type normalization in migration."""

    def test_surface_to_pt_sfc(self) -> None:
        """Test 'surface' is normalized to 'pt_sfc'."""
        config: dict[str, Any] = {"obs": {"test": {"obs_type": "surface"}}}
        result = migrate_config(config, from_version="0.0.0", to_version="1.0.0")
        assert result["obs"]["test"]["obs_type"] == "pt_sfc"

    def test_aircraft_preserved(self) -> None:
        """Test 'aircraft' type is preserved."""
        config: dict[str, Any] = {"obs": {"flight": {"obs_type": "aircraft"}}}
        result = migrate_config(config, from_version="0.0.0", to_version="1.0.0")
        assert result["obs"]["flight"]["obs_type"] == "aircraft"


class TestModelTypeMigration:
    """Tests for model type inference in migration."""

    def test_infer_cmaq_from_name(self) -> None:
        """Test inferring cmaq type from model name."""
        config: dict[str, Any] = {
            "analysis": {},
            "model": {"my_cmaq_run": {}},
            "obs": {},
        }
        result = migrate_config(config, from_version="1.0.0", to_version="2.0.0")
        assert result["model"]["my_cmaq_run"].get("mod_type") == "cmaq"

    def test_infer_wrfchem_from_name(self) -> None:
        """Test inferring wrfchem type from model name."""
        config: dict[str, Any] = {
            "analysis": {},
            "model": {"wrfchem_test": {}},
            "obs": {},
        }
        result = migrate_config(config, from_version="1.0.0", to_version="2.0.0")
        assert result["model"]["wrfchem_test"].get("mod_type") == "wrfchem"

    def test_preserve_existing_mod_type(self) -> None:
        """Test existing mod_type is preserved."""
        config: dict[str, Any] = {
            "analysis": {},
            "model": {"my_cmaq": {"mod_type": "ufs"}},
            "obs": {},
        }
        result = migrate_config(config, from_version="1.0.0", to_version="2.0.0")
        # Should not override existing mod_type
        assert result["model"]["my_cmaq"]["mod_type"] == "ufs"
