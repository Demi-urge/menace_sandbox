"""Focused import compatibility checks for automated_debugger."""

from __future__ import annotations

import importlib


def test_automated_debugger_import_paths_expose_class() -> None:
    pkg_module = importlib.import_module("menace.automated_debugger")
    flat_module = importlib.import_module("automated_debugger")

    assert hasattr(pkg_module, "AutomatedDebugger")
    assert hasattr(flat_module, "AutomatedDebugger")
