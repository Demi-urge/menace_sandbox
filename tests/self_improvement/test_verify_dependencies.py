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


def test_verify_dependencies_autoinstall_success(monkeypatch):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", Path("self_improvement/init.py")
    )

    calls = {"import": 0, "pip": 0}

    def fake_import(name):
        if name == "quick_fix_engine":
            calls["import"] += 1
            if calls["import"] == 1:
                raise ImportError("missing")
        return sys.modules.setdefault(name, types.ModuleType(name))

    monkeypatch.setattr(importlib, "import_module", fake_import)

    def fake_check_call(cmd, **kwargs):
        calls["pip"] += 1
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    init_mod.verify_dependencies()

    assert calls["import"] == 2
    assert calls["pip"] == 1


def test_verify_dependencies_autoinstall_failure(monkeypatch, caplog):
    _prepare_modules(missing=("quick_fix_engine",))
    init_mod = _load_module(
        "menace.self_improvement.init", Path("self_improvement/init.py")
    )

    def always_fail(name):
        if name == "quick_fix_engine":
            raise ImportError("boom")
        return sys.modules.setdefault(name, types.ModuleType(name))

    monkeypatch.setattr(importlib, "import_module", always_fail)

    called = {}

    def fake_check_call(cmd, **kwargs):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr(subprocess, "check_call", fake_check_call)

    caplog.set_level("DEBUG")
    with pytest.raises(RuntimeError) as err:
        init_mod.verify_dependencies()

    assert "quick_fix_engine" in str(err.value)
    assert "boom" in str(err.value)
    assert called["cmd"][-1] == "quick_fix_engine"
    record = next(
        r
        for r in caplog.records
        if r.message == "auto-install for quick_fix_engine failed"
    )
    assert record.error == "boom"
