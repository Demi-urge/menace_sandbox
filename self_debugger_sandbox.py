"""Backward-compatible wrapper for repo-local imports."""

from __future__ import annotations

try:  # pragma: no cover - packaged layout
    from menace.self_debugger_sandbox_impl import SelfDebuggerSandbox
except Exception:  # pragma: no cover - flat-layout fallback or partial installs
    try:
        from menace.self_debugger_sandbox import SelfDebuggerSandbox  # type: ignore
    except Exception:
        class SelfDebuggerSandbox:  # type: ignore
            """Compatibility fallback used when debugger implementation is unavailable."""

            pass

__all__ = ["SelfDebuggerSandbox"]
