"""Unit tests for :func:`self_improvement.init.verify_dependencies`.

The verifier can optionally install missing dependencies when the
``auto_install`` flag is enabled.  These tests ensure the default behaviour is
non-installing and that opt-in installation behaves as expected.
"""

import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest
from dynamic_path_router import resolve_path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _prepare_modules(*, missing: tuple[str, ...] = ()):  # pragma: no cover - helper
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
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg


def test_verify_dependencies_does_not_attempt_install(monkeypatch):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("pip install attempted")

    monkeypatch.setattr(subprocess, "run", fail_run)

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


def test_verify_dependencies_attempts_install_when_enabled(monkeypatch):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    cmds: list[list[str]] = []

    def fake_run(cmd, check):
        cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    def fake_version(name: str) -> str:
        versions = {
            "quick_fix_engine": "1.1",
            "sandbox_runner": "1.0",
            "torch": "2.0",
        }
        return versions.get(name, "0")

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    calls = {"import": 0}

    def fake_import(name, package=None):
        if name == "quick_fix_engine" and calls["import"] == 0:
            calls["import"] += 1
            raise ModuleNotFoundError(name)
        return sys.modules.setdefault(name, types.ModuleType(name))

    monkeypatch.setattr(importlib, "import_module", fake_import)

    init_mod.verify_dependencies(auto_install=True)

    assert cmds and "pip" in cmds[0][0]
    assert calls["import"] == 1  # initial failure only; second import succeeds


def test_verify_dependencies_install_failure_raises(monkeypatch):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    def failing_run(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", failing_run)

    def fake_import(name, package=None):
        if name == "quick_fix_engine":
            raise ModuleNotFoundError(name)
        return sys.modules.setdefault(name, types.ModuleType(name))

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError):
        init_mod.verify_dependencies(auto_install=True)


def test_verify_dependencies_reports_version_mismatch(monkeypatch):
    _prepare_modules()
    init_mod = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    def fake_version(name):
        if name == "torch":
            return "1.0.0"
        return "0"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    with pytest.raises(RuntimeError) as err:
        init_mod.verify_dependencies()

    assert "torch" in str(err.value)


def test_logs_when_neurosales_metadata_missing(monkeypatch, caplog):
    _prepare_modules()
    sys.modules["neurosales"] = types.ModuleType("neurosales")
    init_mod = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    def fake_version(name: str) -> str:
        if name == "neurosales":
            raise importlib.metadata.PackageNotFoundError(name)
        versions = {
            "quick_fix_engine": "1.0",
            "sandbox_runner": "1.0",
            "torch": "2.0",
        }
        return versions.get(name, "0")

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    with caplog.at_level("DEBUG"):
        init_mod.verify_dependencies()

    assert any(
        "failed to read neurosales metadata" in rec.message for rec in caplog.records
    )
