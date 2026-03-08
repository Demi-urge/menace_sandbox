from __future__ import annotations

"""Compatibility shim for :mod:`roi_tracker`."""

from importlib import import_module

_IMPL_MODULE_NAME = f"{__package__}.roi_tracker_impl" if __package__ else "menace.roi_tracker_impl"
_SOURCE_MODULE = None


def _resolve_source_module():
    global _SOURCE_MODULE
    if _SOURCE_MODULE is not None:
        return _SOURCE_MODULE
    try:
        _SOURCE_MODULE = import_module(_IMPL_MODULE_NAME)
        return _SOURCE_MODULE
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        _SOURCE_MODULE = import_module("roi_tracker")
        return _SOURCE_MODULE


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [symbol for symbol in dir(module) if not symbol.startswith("_")]
    globals().update({symbol: getattr(module, symbol) for symbol in exported})
    return list(exported)


__all__ = _reexport(_resolve_source_module())


def __getattr__(name: str):
    """Resolve late-bound attributes from the backing ROI tracker module.

    This keeps imports robust when callers request symbols (for example
    ``ROITracker``) before the source module has fully initialized due to
    circular imports.
    """

    return getattr(_resolve_source_module(), name)
