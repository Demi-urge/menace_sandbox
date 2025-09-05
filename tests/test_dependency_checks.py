"""Dependency verification tests for self-improvement helpers.

These tests assume required packages are installed separately using
``make install-self-improvement-deps``.  Automatic installation is available
but disabled for these tests.
"""
import importlib
import importlib.util
import sys
import types

import pytest
from dynamic_path_router import resolve_path


def _stub_modules(exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    needed = {
        "quick_fix_engine": types.ModuleType("quick_fix_engine"),
        "sandbox_runner": types.ModuleType("sandbox_runner"),
        "sandbox_runner.orphan_integration": types.ModuleType("sandbox_runner.orphan_integration"),
        "sandbox_runner.environment": types.ModuleType("sandbox_runner.environment"),
        "sandbox_runner.bootstrap": types.ModuleType("sandbox_runner.bootstrap"),
        "sandbox_runner.cli": types.ModuleType("sandbox_runner.cli"),
        "sandbox_runner.cycle": types.ModuleType("sandbox_runner.cycle"),
        "relevancy_radar": types.ModuleType("relevancy_radar"),
        "error_logger": types.ModuleType("error_logger"),
        "telemetry_feedback": types.ModuleType("telemetry_feedback"),
        "telemetry_backend": types.ModuleType("telemetry_backend"),
        "torch": types.ModuleType("torch"),
        "neurosales": types.ModuleType("neurosales"),
    }
    needed["sandbox_runner.bootstrap"].initialize_autonomous_sandbox = lambda *a, **k: None
    needed["sandbox_runner.cli"].main = lambda *a, **k: None
    needed["sandbox_runner.cycle"].ensure_vector_service = lambda: None
    pkg_spec = importlib.util.spec_from_loader("sandbox_runner", loader=None, is_package=True)
    needed["sandbox_runner"].__path__ = []
    needed["sandbox_runner"].__spec__ = pkg_spec
    err = needed["error_logger"]
    err.ErrorLogger = object
    rr = needed["relevancy_radar"]
    rr.tracked_import = __import__
    for name, mod in needed.items():
        if name not in exclude:
            sys.modules[name] = mod
        elif name in sys.modules:
            del sys.modules[name]
    pkg = sys.modules.get("sandbox_runner")
    if pkg:
        pkg.bootstrap = sys.modules.get("sandbox_runner.bootstrap")
        pkg.cli = sys.modules.get("sandbox_runner.cli")
        pkg.cycle = sys.modules.get("sandbox_runner.cycle")
        pkg.environment = sys.modules.get("sandbox_runner.environment")


def _load_init():
    spec = importlib.util.spec_from_file_location(
        "self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_dependency(monkeypatch):
    _stub_modules({"neurosales"})
    import importlib
    import importlib.metadata as metadata

    monkeypatch.setattr(
        metadata,
        "version",
        lambda name: "2.0.0" if name == "torch" else "1.0.0",
    )
    orig_import = importlib.import_module

    def fake_import(name, *a, **k):
        if name == "neurosales":
            raise ImportError
        return orig_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    init_module = _load_init()

    with pytest.raises(RuntimeError) as exc:
        init_module.verify_dependencies()
    msg = str(exc.value)
    assert "neurosales" in msg
    assert "pip install neurosales" in msg


def test_version_mismatch(monkeypatch):
    _stub_modules()
    import importlib.metadata as metadata

    def fake_version(name: str) -> str:
        versions = {
            "quick_fix_engine": "0.9",
            "sandbox_runner": "0.9",
            "torch": "2.0",
        }
        return versions.get(name, "1.0")

    monkeypatch.setattr(metadata, "version", fake_version)
    init_module = _load_init()

    with pytest.raises(RuntimeError) as exc:
        init_module.verify_dependencies()
    msg = str(exc.value)
    assert "quick_fix_engine (installed 0.9" in msg
    assert "pip install quick_fix_engine --upgrade" in msg
    assert "sandbox_runner (installed 0.9" in msg
    assert "pip install sandbox_runner --upgrade" in msg
