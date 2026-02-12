"""Compatibility shim for importing :class:`SelfDebuggerSandbox`."""

from __future__ import annotations

try:
    from menace_sandbox.self_debugger_sandbox import SelfDebuggerSandbox
except ModuleNotFoundError as exc:
    if exc.name != "menace_sandbox":
        raise
    from self_debugger_sandbox import SelfDebuggerSandbox
except ImportError:
    raise

__all__ = ["SelfDebuggerSandbox"]
