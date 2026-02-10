"""Compatibility shim for importing :class:`SelfDebuggerSandbox`."""

from __future__ import annotations

try:
    from menace_sandbox.self_debugger_sandbox import SelfDebuggerSandbox
except ImportError:
    from self_debugger_sandbox import SelfDebuggerSandbox

__all__ = ["SelfDebuggerSandbox"]
