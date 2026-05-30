"""Plugin registry system for DAVINCI components.

This module provides a generic, type-safe registry for pluggable components.
Components are registered via decorators and can be looked up by name.

Example usage:
    # Create a registry for model readers
    model_registry: Registry[ModelReader] = Registry("model")

    # Register a component using decorator
    @model_registry.register("cmaq")
    class CMAQReader:
        ...

    # Or register programmatically
    model_registry.register("wrfchem")(WRFChemReader)

    # Look up a component
    reader_cls = model_registry.get("cmaq")
    reader = reader_cls()

    # List all registered components
    for name in model_registry:
        print(name)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar, overload

T = TypeVar("T")


class RegistryError(Exception):
    """Base exception for registry errors."""


class ComponentNotFoundError(RegistryError):
    """Raised when a requested component is not found in the registry."""

    def __init__(self, name: str, registry_name: str) -> None:
        self.name = name
        self.registry_name = registry_name
        super().__init__(
            f"Component '{name}' not found in {registry_name} registry. "
            f"Available components: use list() to see registered components."
        )


class ComponentAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a component with an existing name."""

    def __init__(self, name: str, registry_name: str) -> None:
        self.name = name
        self.registry_name = registry_name
        super().__init__(
            f"Component '{name}' is already registered in {registry_name} registry. "
            f"Use replace=True to override."
        )


class Registry(Generic[T]):
    """A generic registry for pluggable components.

    The registry stores component classes (not instances) indexed by name.
    Components can be registered via decorators or programmatically.

    Type Parameters
    ---------------
    T
        The protocol/base type that registered components must satisfy.

    Parameters
    ----------
    name
        A descriptive name for this registry (e.g., "model", "observation").
        Used in error messages.

    Examples
    --------
    >>> from davinci_monet.core import ModelReader
    >>> model_registry: Registry[type[ModelReader]] = Registry("model")
    >>> @model_registry.register("cmaq")
    ... class CMAQReader:
    ...     pass
    >>> model_registry.get("cmaq")
    <class 'CMAQReader'>
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._components: dict[str, T] = {}

    @property
    def name(self) -> str:
        """The registry name."""
        return self._name

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._components)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered component names."""
        return iter(self._components)

    def __contains__(self, name: str) -> bool:
        """Check if a component name is registered."""
        return name in self._components

    def __repr__(self) -> str:
        components = ", ".join(sorted(self._components.keys()))
        return f"Registry(name={self._name!r}, components=[{components}])"

    @overload
    def register(self, name: str, *, replace: bool = False) -> Callable[[T], T]: ...

    @overload
    def register(self, name: str, component: T, *, replace: bool = False) -> T: ...

    def register(
        self, name: str, component: T | None = None, *, replace: bool = False
    ) -> T | Callable[[T], T]:
        """Register a component with the given name.

        Can be used as a decorator or called directly.

        Parameters
        ----------
        name
            Unique identifier for the component.
        component
            The component to register (if not using as decorator).
        replace
            If True, allow replacing an existing registration.

        Returns
        -------
        T | Callable[[T], T]
            The component (if provided) or a decorator function.

        Raises
        ------
        ComponentAlreadyRegisteredError
            If name is already registered and replace=False.

        Examples
        --------
        As a decorator:

        >>> @registry.register("my_component")
        ... class MyComponent:
        ...     pass

        Direct registration:

        >>> registry.register("my_component", MyComponent)
        """

        def decorator(comp: T) -> T:
            if name in self._components and not replace:
                raise ComponentAlreadyRegisteredError(name, self._name)
            self._components[name] = comp
            return comp

        if component is not None:
            return decorator(component)
        return decorator

    def unregister(self, name: str) -> T:
        """Remove a component from the registry.

        Parameters
        ----------
        name
            The component name to remove.

        Returns
        -------
        T
            The removed component.

        Raises
        ------
        ComponentNotFoundError
            If the component is not registered.
        """
        if name not in self._components:
            raise ComponentNotFoundError(name, self._name)
        return self._components.pop(name)

    def get(self, name: str) -> T:
        """Get a registered component by name.

        Parameters
        ----------
        name
            The component name to look up.

        Returns
        -------
        T
            The registered component.

        Raises
        ------
        ComponentNotFoundError
            If the component is not registered.
        """
        if name not in self._components:
            raise ComponentNotFoundError(name, self._name)
        return self._components[name]

    def get_or_none(self, name: str) -> T | None:
        """Get a registered component by name, or None if not found.

        Parameters
        ----------
        name
            The component name to look up.

        Returns
        -------
        T | None
            The registered component, or None if not found.
        """
        return self._components.get(name)

    def list(self) -> list[str]:
        """Return a sorted list of all registered component names.

        Returns
        -------
        list[str]
            Sorted list of component names.
        """
        return sorted(self._components.keys())

    def items(self) -> Iterator[tuple[str, T]]:
        """Iterate over (name, component) pairs.

        Yields
        ------
        tuple[str, T]
            Pairs of (name, component).
        """
        yield from self._components.items()

    def clear(self) -> None:
        """Remove all registered components."""
        self._components.clear()


# =============================================================================
# Pre-configured registries for each component type
# =============================================================================

# Note: These registries store component *classes* (types), not instances.
# The type hint uses `type[Protocol]` to indicate we're storing classes
# that implement the protocol.

# Using Any here to avoid circular imports - actual type checking
# happens at registration time via runtime_checkable protocols

model_registry: Registry[type] = Registry("model")
"""Registry for model reader classes (CMAQ, WRF-Chem, etc.)."""

observation_registry: Registry[type] = Registry("observation")
"""Registry for observation reader classes (surface, aircraft, satellite)."""

source_registry: Registry[type] = Registry("source")
"""Unified registry for data source reader classes.

Models and observations both register here, keyed by a single ``type`` id and
distinguished only by the geometry their reader declares. Replaces the separate
model_registry and observation_registry in later phases of the unification."""

pairing_registry: Registry[type] = Registry("pairing")
"""Registry for pairing strategy classes (point, track, profile, swath, grid)."""

plotter_registry: Registry[type] = Registry("plotter")
"""Registry for plotter classes (timeseries, scatter, spatial, etc.)."""

statistic_registry: Registry[type] = Registry("statistic")
"""Registry for statistic metric classes (MB, RMSE, R2, etc.)."""

reader_registry: Registry[type] = Registry("reader")
"""Registry for file reader classes (NetCDF, CSV, ICARTT, etc.)."""

writer_registry: Registry[type] = Registry("writer")
"""Registry for file writer classes (NetCDF, pickle, etc.)."""
