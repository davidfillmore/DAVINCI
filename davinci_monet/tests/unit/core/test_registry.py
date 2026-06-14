"""Tests for the plugin registry system."""

from __future__ import annotations

import pytest

from davinci_monet.core.registry import (
    ComponentAlreadyRegisteredError,
    ComponentNotFoundError,
    Registry,
    RegistryError,
    plotter_registry,
    source_registry,
    statistic_registry,
)


class TestRegistry:
    """Tests for the Registry class."""

    def test_create_registry(self) -> None:
        """Test creating a new registry."""
        registry: Registry[type] = Registry("test")
        assert registry.name == "test"
        assert len(registry) == 0

    def test_register_with_decorator(self) -> None:
        """Test registering a component using decorator syntax."""
        registry: Registry[type] = Registry("test")

        @registry.register("my_component")
        class MyComponent:
            pass

        assert "my_component" in registry
        assert registry.get("my_component") is MyComponent

    def test_register_directly(self) -> None:
        """Test registering a component directly."""
        registry: Registry[type] = Registry("test")

        class MyComponent:
            pass

        result = registry.register("my_component", MyComponent)
        assert result is MyComponent
        assert registry.get("my_component") is MyComponent

    def test_register_duplicate_raises_error(self) -> None:
        """Test that registering a duplicate name raises an error."""
        registry: Registry[type] = Registry("test")

        @registry.register("component")
        class FirstComponent:
            pass

        with pytest.raises(ComponentAlreadyRegisteredError) as exc_info:

            @registry.register("component")
            class SecondComponent:
                pass

        assert exc_info.value.name == "component"
        assert exc_info.value.registry_name == "test"

    def test_register_with_replace(self) -> None:
        """Test that replace=True allows overwriting."""
        registry: Registry[type] = Registry("test")

        @registry.register("component")
        class FirstComponent:
            pass

        @registry.register("component", replace=True)
        class SecondComponent:
            pass

        assert registry.get("component") is SecondComponent

    def test_get_not_found_raises_error(self) -> None:
        """Test that getting a non-existent component raises an error."""
        registry: Registry[type] = Registry("test")

        with pytest.raises(ComponentNotFoundError) as exc_info:
            registry.get("nonexistent")

        assert exc_info.value.name == "nonexistent"
        assert exc_info.value.registry_name == "test"

    def test_get_or_none_returns_none(self) -> None:
        """Test that get_or_none returns None for missing component."""
        registry: Registry[type] = Registry("test")
        assert registry.get_or_none("nonexistent") is None

    def test_get_or_none_returns_component(self) -> None:
        """Test that get_or_none returns component when found."""
        registry: Registry[type] = Registry("test")

        @registry.register("component")
        class MyComponent:
            pass

        assert registry.get_or_none("component") is MyComponent

    def test_unregister(self) -> None:
        """Test unregistering a component."""
        registry: Registry[type] = Registry("test")

        @registry.register("component")
        class MyComponent:
            pass

        removed = registry.unregister("component")
        assert removed is MyComponent
        assert "component" not in registry

    def test_unregister_not_found_raises_error(self) -> None:
        """Test that unregistering non-existent component raises error."""
        registry: Registry[type] = Registry("test")

        with pytest.raises(ComponentNotFoundError):
            registry.unregister("nonexistent")

    def test_list_components(self) -> None:
        """Test listing registered components."""
        registry: Registry[type] = Registry("test")

        @registry.register("zebra")
        class Zebra:
            pass

        @registry.register("alpha")
        class Alpha:
            pass

        @registry.register("middle")
        class Middle:
            pass

        # list() should return sorted names
        assert registry.list() == ["alpha", "middle", "zebra"]

    def test_iterate_over_names(self) -> None:
        """Test iterating over registry yields component names."""
        registry: Registry[type] = Registry("test")

        @registry.register("a")
        class A:
            pass

        @registry.register("b")
        class B:
            pass

        names = list(registry)
        assert set(names) == {"a", "b"}

    def test_items_iterator(self) -> None:
        """Test items() yields (name, component) pairs."""
        registry: Registry[type] = Registry("test")

        @registry.register("component")
        class Component:
            pass

        items = list(registry.items())
        assert len(items) == 1
        assert items[0] == ("component", Component)

    def test_contains(self) -> None:
        """Test the 'in' operator."""
        registry: Registry[type] = Registry("test")

        @registry.register("exists")
        class Exists:
            pass

        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self) -> None:
        """Test len() on registry."""
        registry: Registry[type] = Registry("test")
        assert len(registry) == 0

        @registry.register("a")
        class A:
            pass

        assert len(registry) == 1

        @registry.register("b")
        class B:
            pass

        assert len(registry) == 2

    def test_clear(self) -> None:
        """Test clearing all components."""
        registry: Registry[type] = Registry("test")

        @registry.register("a")
        class A:
            pass

        @registry.register("b")
        class B:
            pass

        assert len(registry) == 2
        registry.clear()
        assert len(registry) == 0

    def test_repr(self) -> None:
        """Test string representation."""
        registry: Registry[type] = Registry("test")

        @registry.register("beta")
        class Beta:
            pass

        @registry.register("alpha")
        class Alpha:
            pass

        repr_str = repr(registry)
        assert "test" in repr_str
        assert "alpha" in repr_str
        assert "beta" in repr_str


class TestRegistryExceptions:
    """Tests for registry exception classes."""

    def test_registry_error_is_exception(self) -> None:
        """Test RegistryError is an Exception."""
        assert issubclass(RegistryError, Exception)

    def test_component_not_found_error(self) -> None:
        """Test ComponentNotFoundError attributes and message."""
        error = ComponentNotFoundError("my_comp", "my_registry")
        assert error.name == "my_comp"
        assert error.registry_name == "my_registry"
        assert "my_comp" in str(error)
        assert "my_registry" in str(error)

    def test_component_already_registered_error(self) -> None:
        """Test ComponentAlreadyRegisteredError attributes and message."""
        error = ComponentAlreadyRegisteredError("my_comp", "my_registry")
        assert error.name == "my_comp"
        assert error.registry_name == "my_registry"
        assert "my_comp" in str(error)
        assert "my_registry" in str(error)
        assert "replace=True" in str(error)


class TestPreConfiguredRegistries:
    """Tests for pre-configured component registries."""

    def test_source_registry_exists(self) -> None:
        """Test source_registry is properly configured."""
        assert source_registry.name == "source"
        assert isinstance(source_registry, Registry)

    def test_plotter_registry_exists(self) -> None:
        """Test plotter_registry is properly configured."""
        assert plotter_registry.name == "plotter"
        assert isinstance(plotter_registry, Registry)

    def test_statistic_registry_exists(self) -> None:
        """Test statistic_registry is properly configured."""
        assert statistic_registry.name == "statistic"
        assert isinstance(statistic_registry, Registry)


class TestRegistryWithCallables:
    """Test registry with different callable types."""

    def test_register_function(self) -> None:
        """Test registering a function."""
        registry: Registry[type] = Registry("funcs")

        def my_func() -> str:
            return "hello"

        registry.register("greet", my_func)  # type: ignore[call-overload]
        assert registry.get("greet") is my_func

    def test_register_lambda(self) -> None:
        """Test registering a lambda."""
        registry: Registry[type] = Registry("funcs")

        my_lambda = lambda x: x * 2  # noqa: E731
        registry.register("double", my_lambda)  # type: ignore[call-overload]
        assert registry.get("double") is my_lambda
