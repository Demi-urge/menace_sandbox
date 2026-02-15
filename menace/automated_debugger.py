from __future__ import annotations

"""Compatibility shim for :mod:`automated_debugger`."""

from importlib import import_module
_IS_PACKAGED_CONTEXT = bool(__package__) or __name__.startswith("menace.")


def _is_missing_module_at_import_path(exc: BaseException, *module_names: str) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
        return False
    return (exc.name or "") in set(module_names)


_packaged_module_name = "menace.automated_debugger_impl"
_package_relative_name = f"{__package__}.automated_debugger_impl" if __package__ else _packaged_module_name

try:
    from .automated_debugger_impl import AutomatedDebugger
except ModuleNotFoundError as exc:
    if not _is_missing_module_at_import_path(exc, _packaged_module_name, _package_relative_name):
        raise
    _flat_module = import_module("automated_debugger")
    AutomatedDebugger = _flat_module.AutomatedDebugger
except (ImportError, AttributeError):
    if _IS_PACKAGED_CONTEXT:
        raise
    _flat_module = import_module("automated_debugger")
    AutomatedDebugger = _flat_module.AutomatedDebugger

__all__ = ["AutomatedDebugger"]
