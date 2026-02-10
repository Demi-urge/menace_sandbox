import importlib.util
import types

import pytest
from dynamic_path_router import resolve_path


spec = importlib.util.spec_from_file_location(
    "sandbox_runner.environment",
    resolve_path("sandbox_runner/environment.py"),
)
env = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(env)


def test_resolver_prefers_package_prefixed_import(monkeypatch):
    package_cls = type("PackageSelfDebuggerSandbox", (), {})

    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            return types.SimpleNamespace(SelfDebuggerSandbox=package_cls)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    assert env._resolve_self_debugger_sandbox_class() is package_cls


def test_resolver_falls_back_to_flat_module(monkeypatch):
    flat_cls = type("FlatSelfDebuggerSandbox", (), {})

    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            raise ModuleNotFoundError("missing package module", name=name)
        if name == "self_debugger_sandbox":
            return types.SimpleNamespace(SelfDebuggerSandbox=flat_cls)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    assert env._resolve_self_debugger_sandbox_class() is flat_cls


def test_resolver_raises_enriched_error_when_all_imports_fail(monkeypatch):
    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            raise ModuleNotFoundError("package missing", name=name)
        if name == "self_debugger_sandbox":
            raise ImportError("flat import broke")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        env._resolve_self_debugger_sandbox_class()

    message = str(exc_info.value)
    assert "menace.self_debugger_sandbox" in message
    assert "self_debugger_sandbox" in message
    assert "ModuleNotFoundError('package missing')" in message
    assert "ImportError('flat import broke')" in message
