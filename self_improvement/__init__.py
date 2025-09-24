from __future__ import annotations

"""Self-improvement engine public API."""

import sys as _sys
from types import ModuleType as _ModuleType
from pathlib import Path as _Path


if __name__ == "self_improvement":  # pragma: no cover - runtime import aliasing
    # Allow the package to operate both as ``menace_sandbox.self_improvement`` and
    # as the legacy top-level ``self_improvement`` module.  Older entrypoints
    # import the package directly, which breaks the ``from ..`` relative imports
    # defined throughout the module tree.  Registering this alias ensures Python
    # recognises ``menace_sandbox`` as the parent package so those imports
    # resolve correctly when running in a flat layout.
    __package__ = "menace_sandbox.self_improvement"
    parent = _sys.modules.get("menace_sandbox")
    if parent is None:
        parent = _ModuleType("menace_sandbox")
        parent.__path__ = [str(_Path(__file__).resolve().parent.parent)]
        _sys.modules["menace_sandbox"] = parent
    _sys.modules["menace_sandbox.self_improvement"] = _sys.modules[__name__]


from .api import *  # noqa: F401,F403
from .api import __all__  # noqa: F401

# Backwards compatibility for the deprecated `state_snapshot` module
from . import snapshot_tracker as state_snapshot  # noqa: F401
_sys.modules[__name__ + ".state_snapshot"] = state_snapshot
