from __future__ import annotations

"""Compatibility shim for :mod:`patch_score_backend`."""

from importlib import import_module

_IMPL_MODULE_NAME = f"{__package__}.patch_score_backend_impl" if __package__ else "menace.patch_score_backend_impl"


def _resolve_source_module():
    try:
        return import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        return import_module("patch_score_backend")


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [symbol for symbol in dir(module) if not symbol.startswith("_")]
    globals().update({symbol: getattr(module, symbol) for symbol in exported})
    return list(exported)


__all__ = _reexport(_resolve_source_module())
