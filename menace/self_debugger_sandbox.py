"""Compatibility shim for importing :class:`SelfDebuggerSandbox`."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_root_self_debugger_sandbox():
    module_path = Path(__file__).resolve().parents[1] / "self_debugger_sandbox.py"
    spec = importlib.util.spec_from_file_location(
        "root_self_debugger_sandbox", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "SelfDebuggerSandbox"):
        raise AttributeError(
            f"Module loaded from {module_path} does not expose SelfDebuggerSandbox"
        )

    return module


try:
    from menace_sandbox.self_debugger_sandbox import SelfDebuggerSandbox
except ModuleNotFoundError as exc:
    if exc.name not in {"menace_sandbox", "menace_sandbox.self_debugger_sandbox"}:
        raise
    try:
        from self_debugger_sandbox import SelfDebuggerSandbox
    except ImportError:
        _root_module = _load_root_self_debugger_sandbox()
        SelfDebuggerSandbox = _root_module.SelfDebuggerSandbox

__all__ = ["SelfDebuggerSandbox"]
