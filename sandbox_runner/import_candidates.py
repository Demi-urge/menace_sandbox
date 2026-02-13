"""Shared module import candidate lists for resolver helpers."""

from __future__ import annotations

# Runtime imports must resolve from the packaged ``menace`` namespace.
# A repo-root compatibility shim still exists for legacy direct imports,
# but runtime-critical callers should only depend on the packaged module.
SELF_DEBUGGER_SANDBOX_MODULE_CANDIDATES: tuple[str, ...] = (
    "menace.self_debugger_sandbox",
)
