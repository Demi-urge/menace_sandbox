from __future__ import annotations

"""Compatibility package for :mod:`sandbox_runner`."""

from importlib import import_module

_IMPL_MODULE_NAME = (
    f"{__package__}.runner_impl" if __package__ else "menace.sandbox_runner.runner_impl"
)


def _resolve_source_module():
    try:
        return import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        return import_module("sandbox_runner")


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [symbol for symbol in dir(module) if not symbol.startswith("_")]
    globals().update({symbol: getattr(module, symbol) for symbol in exported})
    return list(exported)


__all__ = _reexport(_resolve_source_module())
