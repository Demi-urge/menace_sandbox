from __future__ import annotations

"""Compatibility shim for :mod:`code_database`."""

from importlib import import_module

_IMPL_MODULE_NAME = f"{__package__}.code_database_impl" if __package__ else "menace.code_database_impl"


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
    else:
        exported = list(exported)

    globals().update({symbol: getattr(module, symbol) for symbol in exported})

    # Legacy internal imports still rely on _hash_code from this shim.
    # This keeps backward compatibility rather than hiding/removing import errors.
    if hasattr(module, "_hash_code"):
        globals()["_hash_code"] = getattr(module, "_hash_code")
        if "_hash_code" not in exported:
            exported.append("_hash_code")

    return list(exported)


__all__ = _reexport(_resolve_source_module())
