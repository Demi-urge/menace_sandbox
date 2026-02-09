"""Shim bootstrap placeholder utilities for self-improvement entrypoints."""

from __future__ import annotations

from importlib import import_module, util

_MODULE_CANDIDATES: tuple[str, ...]
if __package__ and "." in __package__:
    _parent = __package__.rsplit(".", 1)[0]
    _MODULE_CANDIDATES = (
        f"{_parent}.bootstrap_placeholder",
        "bootstrap_placeholder",
    )
else:
    _MODULE_CANDIDATES = ("bootstrap_placeholder",)

_bootstrap_module = None
for _name in _MODULE_CANDIDATES:
    if util.find_spec(_name) is not None:
        _bootstrap_module = import_module(_name)
        break
if _bootstrap_module is None:
    _bootstrap_module = import_module("bootstrap_placeholder")

advertise_broker_placeholder = _bootstrap_module.advertise_broker_placeholder
bootstrap_broker = _bootstrap_module.bootstrap_broker

__all__ = ["advertise_broker_placeholder", "bootstrap_broker"]
