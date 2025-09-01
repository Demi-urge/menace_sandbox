from __future__ import annotations

"""Registry for language model backend factories.

This module allows dynamic registration of :class:`LLMClient` factories so new
backends can be plugged in without modifying core routing logic.
"""

from importlib import import_module
from typing import Callable, Dict

from llm_interface import LLMClient

ClientFactory = Callable[[], LLMClient]

# Internal mapping of backend name to factory callable
_REGISTRY: Dict[str, ClientFactory] = {}


def register_backend(name: str, factory: ClientFactory) -> None:
    """Register *factory* under *name*.

    Parameters
    ----------
    name:
        Identifier used in ``SandboxSettings`` to select the backend.
    factory:
        Callable returning an :class:`LLMClient` instance.
    """

    _REGISTRY[name.lower()] = factory


def register_backend_from_path(name: str, path: str) -> None:
    """Register a backend by dotted import *path*.

    The path must reference a callable returning an :class:`LLMClient` instance.
    The import is deferred until the backend is instantiated.
    """

    def _lazy_factory() -> LLMClient:
        module_name, attr_name = path.rsplit(".", 1)
        module = import_module(module_name)
        factory = getattr(module, attr_name)
        if not callable(factory):  # pragma: no cover - defensive
            raise TypeError(f"Backend factory at {path!r} is not callable")
        return factory()

    register_backend(name, _lazy_factory)


def get_backend(name: str) -> ClientFactory:
    """Return the factory registered for *name*.

    Raises
    ------
    ValueError
        If *name* is not present in the registry.
    """

    try:
        return _REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown LLM backend: {name}") from exc


def create_backend(name: str) -> LLMClient:
    """Instantiate the backend registered under *name*."""

    factory = get_backend(name)
    return factory()


def backend(name: str):
    """Decorator registering ``name`` for the decorated factory."""

    def _decorator(func: ClientFactory) -> ClientFactory:
        register_backend(name, func)
        return func

    return _decorator


# Public alias exposing the registry for introspection in tests or debugging
REGISTRY = _REGISTRY

__all__ = [
    "ClientFactory",
    "REGISTRY",
    "backend",
    "create_backend",
    "get_backend",
    "register_backend",
    "register_backend_from_path",
]
