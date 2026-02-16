from __future__ import annotations

"""Compatibility shim for :mod:`code_database`."""

from importlib import import_module

_IMPL_MODULE_NAME = f"{__package__}.code_database_impl" if __package__ else "menace.code_database_impl"
_COMPAT_PRIVATE_EXPORTS = {"_hash_code"}


def _resolve_source_module():
    try:
        return import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        return import_module("code_database")


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [symbol for symbol in dir(module) if not symbol.startswith("_")]

    # Required for packaged mode: self_debugger_sandbox_impl imports _hash_code directly.
    for symbol in _COMPAT_PRIVATE_EXPORTS:
        if hasattr(module, symbol) and symbol not in exported:
            exported = [*exported, symbol]

    globals().update({symbol: getattr(module, symbol) for symbol in exported})
    return list(exported)


__all__ = _reexport(_resolve_source_module())
