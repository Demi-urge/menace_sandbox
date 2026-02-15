from __future__ import annotations

"""Compatibility shim for :mod:`logging_utils`."""

from importlib import import_module

_IMPL_MODULE_NAME = f"{__package__}.logging_utils_impl" if __package__ else "menace.logging_utils_impl"


def _resolve_source_module():
    try:
        return import_module(_IMPL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if (exc.name or "") != _IMPL_MODULE_NAME:
            raise
        return import_module("logging_utils")


_base = _resolve_source_module()
setup_logging = _base.setup_logging
get_logger = _base.get_logger
log_record = _base.log_record
set_correlation_id = getattr(_base, "set_correlation_id", lambda *_a, **_k: None)
__all__ = ["setup_logging", "get_logger", "log_record", "set_correlation_id"]
