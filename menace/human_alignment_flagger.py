from __future__ import annotations

"""Compatibility shim for :mod:`human_alignment_flagger`."""

from importlib import import_module

_IMPL_MODULE_NAME = (
    f"{__package__}.human_alignment_flagger_impl" if __package__ else "menace.human_alignment_flagger_impl"
)


def _resolve_source_module():
    try:
        return import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        return import_module("human_alignment_flagger")


def _reexport(module) -> list[str]:
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [symbol for symbol in dir(module) if not symbol.startswith("_")]
    globals().update({symbol: getattr(module, symbol) for symbol in exported})
    return list(exported)


_source_module = _resolve_source_module()
_public_all = _reexport(_source_module)
_collect_diff_data = getattr(_source_module, "_collect_diff_data")

# Keep compatibility with the upstream public API while exporting this
# semiprivate helper explicitly for internal callers.
__all__ = list(dict.fromkeys([*_public_all, "_collect_diff_data"]))
