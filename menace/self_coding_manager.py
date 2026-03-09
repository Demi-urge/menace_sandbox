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


_SOURCE_MODULE = _resolve_source_module()
__all__ = _reexport(_SOURCE_MODULE)

# Legacy compatibility: expose ``DataBot`` from this import path even when the
# source module keeps a narrow ``__all__``.
try:
    from .data_bot import DataBot as _LegacyDataBot
except Exception:  # pragma: no cover - preserve import-time behaviour
    _LegacyDataBot = None
else:
    DataBot = _LegacyDataBot
    if "DataBot" not in __all__:
        __all__.append("DataBot")


def __getattr__(name: str):
    """Defer unresolved attributes to the source module.

    This preserves compatibility for private helpers imported explicitly from this
    shim (for example ``from menace.self_coding_manager import
    _manager_generate_helper_with_builder``).
    """

    try:
        return getattr(_SOURCE_MODULE, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
