from __future__ import annotations

"""Compatibility shim for :mod:`roi_tracker`."""

from importlib import import_module
from types import ModuleType

_IMPL_MODULE_NAME = f"{__package__}.roi_tracker_impl" if __package__ else "menace.roi_tracker_impl"
_SOURCE_MODULE: ModuleType | None = None


def _resolve_source_module() -> ModuleType:
    global _SOURCE_MODULE
    if _SOURCE_MODULE is not None:
        return _SOURCE_MODULE

    try:
        _SOURCE_MODULE = import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        # A missing package-local implementation module is expected in setups
        # that only ship the legacy top-level ``roi_tracker`` module.
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        _SOURCE_MODULE = import_module("roi_tracker")

    return _SOURCE_MODULE


# Keep a conservative static export surface that is safe before the backing
# module has fully initialized.
__all__ = ["ROITracker"]


def __getattr__(name: str):
    """Resolve late-bound attributes from the backing ROI tracker module.

    This keeps imports robust when callers request symbols (for example
    ``ROITracker``) before the source module has fully initialized due to
    circular imports.
    """

    return getattr(_resolve_source_module(), name)


def __dir__() -> list[str]:
    """Include attributes from the lazily resolved backing module."""

    names = set(globals())
    try:
        names.update(dir(_resolve_source_module()))
    except Exception:
        pass
    return sorted(names)
