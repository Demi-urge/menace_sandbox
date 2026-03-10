"""Shim bootstrap placeholder utilities for self-improvement entrypoints."""

from __future__ import annotations

from importlib import import_module, util
from typing import Any

_MODULE_CANDIDATES: tuple[str, ...]
if __package__ and "." in __package__:
    _parent = __package__.rsplit(".", 1)[0]
    _MODULE_CANDIDATES = (
        f"{_parent}.bootstrap_placeholder",
        "bootstrap_placeholder",
    )
else:
    _MODULE_CANDIDATES = ("bootstrap_placeholder",)

class _BootstrapModuleShim:
    """Fallback bootstrap-placeholder module shim."""

    def advertise_broker_placeholder(self, *_args: Any, **_kwargs: Any) -> tuple[None, None]:
        return None, None

    def bootstrap_broker(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}


_bootstrap_module: Any = _BootstrapModuleShim()
for _name in _MODULE_CANDIDATES:
    if util.find_spec(_name) is None:
        continue
    try:
        _bootstrap_module = import_module(_name)
        break
    except Exception:
        continue
if isinstance(_bootstrap_module, _BootstrapModuleShim):
    try:
        _bootstrap_module = import_module("bootstrap_placeholder")
    except Exception:
        pass

advertise_broker_placeholder = _bootstrap_module.advertise_broker_placeholder
bootstrap_broker = _bootstrap_module.bootstrap_broker

__all__ = ["advertise_broker_placeholder", "bootstrap_broker"]
