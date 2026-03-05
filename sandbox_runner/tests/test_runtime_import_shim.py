from __future__ import annotations

import importlib
import logging
import sys
import types


def _import_run_autonomous_with_pydantic_stub(monkeypatch):
    pydantic_stub = types.ModuleType("pydantic")

    class _BaseModel:
        model_fields = {}

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            if not hasattr(cls, "model_fields"):
                cls.model_fields = {}

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    class _RootModel(_BaseModel):
        pass

    pydantic_stub.BaseModel = _BaseModel
    pydantic_stub.ConfigDict = dict
    pydantic_stub.RootModel = _RootModel
    pydantic_stub.ValidationError = ValueError
    pydantic_stub.field_validator = lambda *_a, **_k: (lambda fn: fn)
    pydantic_stub.model_validator = lambda *_a, **_k: (lambda fn: fn)

    monkeypatch.setitem(sys.modules, "pydantic", pydantic_stub)
    if "run_autonomous" in sys.modules:
        return sys.modules["run_autonomous"]
    return importlib.import_module("run_autonomous")


def test_runtime_import_initialization_preserves_package_submodule_imports(monkeypatch):
    run_autonomous = _import_run_autonomous_with_pydantic_stub(monkeypatch)
    sandbox_runner_pkg = importlib.import_module("sandbox_runner")
    monkeypatch.setattr(
        run_autonomous,
        "_load_legacy_sandbox_runner",
        lambda: types.SimpleNamespace(_sandbox_main=lambda *_a, **_k: None),
    )
    run_autonomous._initialize_sandbox_runner_runtime_imports(sandbox_runner_pkg)

    bootstrap = importlib.import_module("sandbox_runner.bootstrap")
    import_candidates = importlib.import_module("sandbox_runner.import_candidates")

    assert hasattr(bootstrap, "bootstrap_environment")
    assert hasattr(import_candidates, "SELF_DEBUGGER_SANDBOX_MODULE_CANDIDATES")


def test_runtime_import_logs_remediation_for_file_module(monkeypatch, caplog):
    run_autonomous = _import_run_autonomous_with_pydantic_stub(monkeypatch)
    package_module = importlib.import_module("sandbox_runner")
    legacy = types.SimpleNamespace(_sandbox_main=lambda *_a, **_k: None)

    file_module = types.ModuleType("sandbox_runner")
    file_module.__file__ = "/tmp/sandbox_runner.py"

    monkeypatch.setattr(run_autonomous, "_load_legacy_sandbox_runner", lambda: legacy)

    with caplog.at_level(logging.WARNING):
        resolved = run_autonomous._initialize_sandbox_runner_runtime_imports(file_module)

    assert resolved is file_module
    assert "without package path" in caplog.text
    assert importlib.import_module("sandbox_runner") is package_module
    assert importlib.import_module("sandbox_runner.bootstrap")
    assert importlib.import_module("sandbox_runner.import_candidates")


def test_sandbox_main_host_module_uses_legacy_fallback(monkeypatch):
    run_autonomous = _import_run_autonomous_with_pydantic_stub(monkeypatch)
    legacy = types.SimpleNamespace(_sandbox_main=lambda *_a, **_k: None)
    monkeypatch.setattr(run_autonomous, "sandbox_runner", types.SimpleNamespace())
    monkeypatch.setattr(run_autonomous, "_load_legacy_sandbox_runner", lambda: legacy)

    assert run_autonomous._sandbox_main_host_module() is legacy
