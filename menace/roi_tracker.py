from __future__ import annotations

"""Compatibility shim for :mod:`roi_tracker`."""

from importlib import import_module
from types import ModuleType

_SOURCE_MODULE: ModuleType = import_module("roi_tracker")

if not hasattr(_SOURCE_MODULE, "ROITracker"):
    raise ImportError(
        "menace.roi_tracker expected 'ROITracker' in module 'roi_tracker', "
        "but it was not found."
    )

ROITracker = _SOURCE_MODULE.ROITracker

_PUBLIC_SYMBOLS = list(getattr(_SOURCE_MODULE, "__all__", ()))
if not _PUBLIC_SYMBOLS:
    _PUBLIC_SYMBOLS = [name for name in dir(_SOURCE_MODULE) if not name.startswith("_")]

for _name in _PUBLIC_SYMBOLS:
    if _name in {"ROITracker", "__all__"}:
        continue
    try:
        globals()[_name] = getattr(_SOURCE_MODULE, _name)
    except AttributeError:
        continue


__all__ = list(dict.fromkeys(["ROITracker", *_PUBLIC_SYMBOLS]))


def __getattr__(name: str):
    """Resolve attributes from the backing ROI tracker module."""

    try:
        return getattr(_SOURCE_MODULE, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    """Include attributes from the backing module."""

    names = set(globals())
    try:
        names.update(dir(_SOURCE_MODULE))
    except Exception:
        pass
    return sorted(names)
