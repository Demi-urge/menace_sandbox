"""Integration tests for self-improvement dependency checks.

Dependencies are expected to be installed separately via
``make install-self-improvement-deps``; the verifier reports problems by
default but can optionally install missing packages when enabled.
"""
import importlib
import importlib.util
import types
import sys
from pathlib import Path

import pytest


def _load_verify_dependencies():
    root = Path(__file__).resolve().parent.parent
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = [str(root)]
    sys.modules.setdefault("menace_sandbox", pkg)
    sub_pkg = types.ModuleType("menace_sandbox.self_improvement")
    sub_pkg.__path__ = [str(root / "self_improvement")]
    sys.modules.setdefault("menace_sandbox.self_improvement", sub_pkg)

    spec = importlib.util.spec_from_file_location(
        "menace_sandbox.self_improvement.init", root / "self_improvement" / "init.py"  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "menace_sandbox.self_improvement"
    sys.modules["menace_sandbox.self_improvement.init"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.verify_dependencies


def test_verify_dependencies_reports_all_missing(monkeypatch):
    """Missing modules are aggregated into a single error."""

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda *a, **k: None
    sandbox_runner = types.ModuleType("sandbox_runner")
    sandbox_runner.bootstrap = bootstrap
    sys.modules.update({"sandbox_runner": sandbox_runner, "sandbox_runner.bootstrap": bootstrap})

    verify_dependencies = _load_verify_dependencies()

    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name in {"relevancy_radar", "error_logger"}:
            raise ModuleNotFoundError(name)
        if name in {
            "quick_fix_engine",
            "sandbox_runner.orphan_integration",
            "torch",
            "pytorch",
        }:
            return types.ModuleType(name)
        return original_import(name, package)

    def fake_version(name):
        return "2.0.0"

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    with pytest.raises(RuntimeError) as excinfo:
        verify_dependencies()

    msg = str(excinfo.value)
    assert "relevancy_radar" in msg and "error_logger" in msg
    assert "quick_fix_engine" not in msg


def test_verify_dependencies_requires_torch(monkeypatch):
    """A clear error is raised when PyTorch is unavailable."""

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda *a, **k: None
    sandbox_runner = types.ModuleType("sandbox_runner")
    sandbox_runner.bootstrap = bootstrap
    sys.modules.update({"sandbox_runner": sandbox_runner, "sandbox_runner.bootstrap": bootstrap})

    verify_dependencies = _load_verify_dependencies()

    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name in {"torch", "pytorch"}:
            raise ModuleNotFoundError(name)
        if name in {
            "quick_fix_engine",
            "sandbox_runner.orphan_integration",
            "relevancy_radar",
            "error_logger",
        }:
            return types.ModuleType(name)
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        verify_dependencies()

    assert "torch" in str(excinfo.value)
