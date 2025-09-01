"""Dependency verification tests for the self-improvement module.

The module no longer performs automatic installations; dependencies must be
installed ahead of time using ``make install-self-improvement-deps``.
"""
import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _prepare_modules(*, missing: tuple[str, ...] = ()):  # pragma: no cover - setup helper
    deps = [
        "quick_fix_engine",
        "sandbox_runner",
        "sandbox_runner.bootstrap",
        "sandbox_runner.orphan_integration",
        "relevancy_radar",
        "error_logger",
        "telemetry_feedback",
        "telemetry_backend",
        "torch",
    ]
    for mod in deps:
        if mod in missing:
            continue
        parts = mod.split(".")
        for i in range(1, len(parts)):
            pkg = ".".join(parts[:i])
            pkg_mod = sys.modules.setdefault(pkg, types.ModuleType(pkg))
            pkg_mod.__path__ = []
        module = types.ModuleType(mod)
        if mod == "sandbox_runner.bootstrap":
            module.initialize_autonomous_sandbox = lambda *a, **k: None
        sys.modules[mod] = module
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg


def test_verify_dependencies_does_not_attempt_install(monkeypatch):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", Path("self_improvement/init.py")
    )

    def fail_pip(*args, **kwargs):
        raise AssertionError("pip install attempted")

    monkeypatch.setattr(subprocess, "check_call", fail_pip)

    calls = {"import": 0}

    def fake_import(name, package=None):
        if name == "quick_fix_engine":
            calls["import"] += 1
            raise ModuleNotFoundError(name)
        return sys.modules.setdefault(name, types.ModuleType(name))

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError) as err:
        init_mod.verify_dependencies()

    assert "quick_fix_engine" in str(err.value)
    assert calls["import"] == 1


def test_verify_dependencies_does_not_install_even_when_enabled(monkeypatch):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", Path("self_improvement/init.py")
    )

    def fail_pip(*args, **kwargs):
        raise AssertionError("pip install attempted")

    monkeypatch.setattr(subprocess, "check_call", fail_pip)
    init_mod.settings.auto_install_dependencies = True
    init_mod.settings.menace_offline_install = False

    with pytest.raises(RuntimeError) as err:
        init_mod.verify_dependencies()

    assert "quick_fix_engine" in str(err.value)


def test_verify_dependencies_reports_version_mismatch(monkeypatch):
    _prepare_modules()
    init_mod = _load_module(
        "menace.self_improvement.init", Path("self_improvement/init.py")
    )

    def fake_version(name):
        if name == "torch":
            return "1.0.0"
        return "0"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    with pytest.raises(RuntimeError) as err:
        init_mod.verify_dependencies()

    assert "torch" in str(err.value)
