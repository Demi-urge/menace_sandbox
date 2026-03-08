from __future__ import annotations

"""Compatibility shim for :mod:`self_coding_manager`."""

from importlib import import_module
from pathlib import Path

_IMPL_MODULE_NAME = f"{__package__}.self_coding_manager_impl" if __package__ else "menace.self_coding_manager_impl"


def _resolve_source_module():
    try:
        return import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise

    package_root = Path(__file__).resolve().parents[1].name
    fallback_candidates = [f"{package_root}.self_coding_manager", "self_coding_manager"]

    for module_name in fallback_candidates:
        try:
            return import_module(module_name)
        except ModuleNotFoundError as exc:
            if (exc.name or "") != module_name:
                raise

    raise ModuleNotFoundError(
        "Could not resolve self_coding_manager source module via compatibility shim"
    )


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [symbol for symbol in dir(module) if not symbol.startswith("_")]
    globals().update({symbol: getattr(module, symbol) for symbol in exported})
    return list(exported)


__all__ = _reexport(_resolve_source_module())
