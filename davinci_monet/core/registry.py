"""Plugin registry system for DAVINCI components.

This module provides a generic, type-safe registry for pluggable components.
Components are registered via decorators and can be looked up by name.

Example usage:
    # Create a registry for source readers
    source_registry: Registry[SourceReader] = Registry("source")

    # Register a component using decorator
    @source_registry.register("cmaq")
    class CMAQReader:
        ...

    # Or register programmatically
    source_registry.register("wrfchem")(WRFChemReader)

    # Look up a component
    reader_cls = source_registry.get("cmaq")
    reader = reader_cls()

    # List all registered components
    for name in source_registry:
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
        A descriptive name for this registry (e.g., "dataset", "dataset").
        Used in error messages.

    Examples
    --------
    >>> from davinci_monet.core import SourceReader
    >>> source_registry: Registry[type[SourceReader]] = Registry("source")
    >>> @source_registry.register("cmaq")
    ... class CMAQReader:
    ...     pass
    >>> source_registry.get("cmaq")
    <class 'CMAQReader'>
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._components: dict[str, T] = {}
        # Alternate-name redirects: alias -> canonical target name.
        self._aliases: dict[str, str] = {}

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
        """Check if a component name is registered (resolving aliases)."""
        return self.resolve(name) in self._components

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

        def decorator(registered: T) -> T:
            if name in self._components and not replace:
                raise ComponentAlreadyRegisteredError(name, self._name)
            self._components[name] = registered
            return registered

        if component is not None:
            return decorator(component)
        return decorator

    def register_alias(self, alias: str, target: str) -> None:
        """Register an ``alias`` name that redirects to ``target``.

        The target need not exist yet (it may register later). Lookups via
        :meth:`get`/:meth:`get_or_none`/``in`` resolve the alias to the target.

        Parameters
        ----------
        alias
            The alternate name.
        target
            The canonical name it should resolve to.
        """
        self._aliases[alias] = target

    def is_alias(self, name: str) -> bool:
        """Return True if ``name`` is a registered alias (not a real component)."""
        return name in self._aliases

    def resolve(self, name: str) -> str:
        """Resolve ``name`` through any alias chain to its canonical target.

        Non-alias names are returned unchanged. Alias chains are followed with a
        visited-guard so a cyclic alias cannot loop forever.
        """
        seen: set[str] = set()
        while name in self._aliases and name not in seen:
            seen.add(name)
            name = self._aliases[name]
        return name

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
        resolved = self.resolve(name)
        if resolved not in self._components:
            raise ComponentNotFoundError(name, self._name)
        return self._components[resolved]

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
        return self._components.get(self.resolve(name))

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

# Note: These registries store component *classes* (types), not instances,
# keyed by a unique name. Registration validates only that the name is unique
# (see Registry.register) — it does NOT type-check the class against any
# protocol. The `type` hint is intentionally broad to avoid circular imports.

source_registry: Registry[type] = Registry("source")
"""Unified registry for data source reader classes.

All data source readers register here, keyed by a single ``type`` id and
distinguished by reader module, declared geometry, and source metadata.
This is the single source of truth for reader registration."""

plotter_registry: Registry[type] = Registry("plotter")
"""Registry for plotter classes (timeseries, scatter, spatial, etc.)."""

statistic_registry: Registry[type] = Registry("statistic")
"""Registry for statistic metric classes (MB, RMSE, R2, etc.)."""

analysis_registry: Registry[type] = Registry("analysis")
"""Registry for derived-analysis classes (eof, wavelet, ...).

Like the other registries it stores component *classes* keyed by a unique
``type`` id. An analysis consumes one source dataset and emits a derived
dataset (see ``davinci_monet.analysis.base.DerivedAnalysis``)."""
