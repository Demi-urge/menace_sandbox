"""Smoke tests for menace compatibility shims."""

from __future__ import annotations

import importlib


def test_critical_menace_shims_import() -> None:
    critical_modules = [
        "menace.logging_utils",
        "menace.retry_utils",
        "menace.automated_debugger",
        "menace.human_alignment_flagger",
    ]

    imported = {name: importlib.import_module(name) for name in critical_modules}

    assert hasattr(imported["menace.logging_utils"], "log_record")
    assert hasattr(imported["menace.retry_utils"], "with_retry")
    assert hasattr(imported["menace.automated_debugger"], "AutomatedDebugger")
    assert hasattr(imported["menace.human_alignment_flagger"], "_collect_diff_data")
