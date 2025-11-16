from __future__ import annotations

import builtins
import importlib
import logging
import sys
import types
from pathlib import Path

import pytest


def test_orphan_analyzer_missing(monkeypatch, caplog):
    pkg_path = Path(__file__).resolve().parents[2] / "sandbox_runner"
    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = [str(pkg_path)]
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.delitem(sys.modules, "sandbox_runner.orphan_discovery", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "orphan_analyzer":
            raise ModuleNotFoundError("No module named 'orphan_analyzer'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="orphan_analyzer is required"):
            importlib.import_module("sandbox_runner.orphan_discovery")

    assert any(
        "Failed to import 'orphan_analyzer'" in rec.message for rec in caplog.records
    )
