from __future__ import annotations

"""Compatibility shim for :mod:`data_bot` legacy imports."""

from importlib import import_module

_SOURCE_MODULE = import_module("data_bot")


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [name for name in dir(module) if not name.startswith("_")]
    globals().update({name: getattr(module, name) for name in exported})
    return list(exported)


__all__ = _reexport(_SOURCE_MODULE)

if "MetricsDB" not in __all__ and hasattr(_SOURCE_MODULE, "MetricsDB"):
    MetricsDB = getattr(_SOURCE_MODULE, "MetricsDB")
    __all__.append("MetricsDB")


def __getattr__(name: str):
    try:
        return getattr(_SOURCE_MODULE, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
