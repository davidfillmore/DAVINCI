"""Tests for the unified data-source abstraction (Phase 1).

These verify the new SourceReader / SourceProcessor protocols and the
source_registry exist and behave correctly. They are additive: the legacy
ModelReader / ObservationReader protocols and registries are untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.protocols import DataGeometry, SourceProcessor, SourceReader


class _FullSourceReader:
    """A reader with name, geometry, open, and get_variable_mapping."""

    @property
    def name(self) -> str:
        return "mock_source"

    @property
    def geometry(self) -> DataGeometry:
        return DataGeometry.GRID

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        time_range: tuple[Any, Any] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        return xr.Dataset(attrs={"geometry": self.geometry.name})

    def get_variable_mapping(self) -> Mapping[str, str]:
        return {"ozone": "O3"}


class _NoGeometryReader:
    """A reader missing the required geometry property (former ModelReader shape)."""

    @property
    def name(self) -> str:
        return "no_geometry"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        return xr.Dataset()

    def get_variable_mapping(self) -> Mapping[str, str]:
        return {}


class TestSourceReaderProtocol:
    def test_runtime_checkable_accepts_full_reader(self) -> None:
        assert isinstance(_FullSourceReader(), SourceReader)

    def test_runtime_checkable_rejects_reader_without_geometry(self) -> None:
        # The key contract change: every source reader MUST declare geometry.
        assert not isinstance(_NoGeometryReader(), SourceReader)

    def test_geometry_is_data_geometry(self) -> None:
        assert _FullSourceReader().geometry is DataGeometry.GRID


class _MockProcessor:
    def process(self, dataset: xr.Dataset, **kwargs: Any) -> xr.Dataset:
        return dataset


class TestSourceProcessorProtocol:
    def test_runtime_checkable_accepts_processor(self) -> None:
        assert isinstance(_MockProcessor(), SourceProcessor)

    def test_processor_returns_dataset(self) -> None:
        ds = xr.Dataset()
        assert _MockProcessor().process(ds) is ds


from davinci_monet.core.registry import Registry, source_registry


class TestSourceRegistry:
    def test_source_registry_exists(self) -> None:
        assert source_registry.name == "source"
        assert isinstance(source_registry, Registry)

    def test_register_and_get(self) -> None:
        local: Registry[type] = Registry("source")

        @local.register("cesm_fv")
        class _Reader:
            pass

        assert local.get("cesm_fv") is _Reader


class TestCorePackageExports:
    def test_protocols_exported_from_core(self) -> None:
        from davinci_monet.core import SourceProcessor, SourceReader

        assert SourceReader is not None
        assert SourceProcessor is not None

    def test_registry_exported_from_core(self) -> None:
        from davinci_monet.core import source_registry

        assert source_registry.name == "source"
