"""Compatibility shim exporting :class:`SelfDebuggerSandbox`."""

from __future__ import annotations

import importlib.util
from pathlib import Path


try:
    from self_debugger_sandbox import SelfDebuggerSandbox
except ImportError:
    _MODULE_PATH = Path(__file__).resolve().parents[1] / "self_debugger_sandbox.py"
    _SPEC = importlib.util.spec_from_file_location("self_debugger_sandbox", _MODULE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Could not load module specification from {_MODULE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    SelfDebuggerSandbox = _MODULE.SelfDebuggerSandbox


__all__ = ["SelfDebuggerSandbox"]
