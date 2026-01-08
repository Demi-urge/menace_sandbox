from __future__ import annotations

from importlib import import_module

_mod = import_module("bootstrap_gate")

__all__ = getattr(_mod, "__all__", [
    "resolve_bootstrap_placeholders",
    "wait_for_bootstrap_gate",
])

globals().update({name: getattr(_mod, name) for name in __all__})
