"""Shared module import candidate lists for resolver helpers."""

from __future__ import annotations

SELF_DEBUGGER_SANDBOX_MODULE_CANDIDATES: tuple[str, ...] = (
    "menace.self_debugger_sandbox",
    "menace_sandbox.self_debugger_sandbox",
    "self_debugger_sandbox",
)
