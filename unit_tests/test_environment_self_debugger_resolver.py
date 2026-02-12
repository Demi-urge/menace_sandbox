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


def test_resolver_prefers_menace_package_import(monkeypatch):
    package_cls = type("PackageSelfDebuggerSandbox", (), {})

    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            return types.SimpleNamespace(SelfDebuggerSandbox=package_cls)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    assert env._resolve_self_debugger_sandbox_class() is package_cls


def test_resolver_falls_back_to_menace_sandbox_package(monkeypatch):
    package_cls = type("MenaceSandboxSelfDebuggerSandbox", (), {})

    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            raise ModuleNotFoundError("missing package module", name=name)
        if name == "menace_sandbox.self_debugger_sandbox":
            return types.SimpleNamespace(SelfDebuggerSandbox=package_cls)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    assert env._resolve_self_debugger_sandbox_class() is package_cls


def test_resolver_falls_back_to_flat_module(monkeypatch):
    flat_cls = type("FlatSelfDebuggerSandbox", (), {})

    def fake_import_module(name):
        if name in {
            "menace.self_debugger_sandbox",
            "menace_sandbox.self_debugger_sandbox",
        }:
            raise ModuleNotFoundError("missing package module", name=name)
        if name == "self_debugger_sandbox":
            return types.SimpleNamespace(SelfDebuggerSandbox=flat_cls)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    assert env._resolve_self_debugger_sandbox_class() is flat_cls


def test_resolver_raises_module_not_found_when_all_candidates_absent(monkeypatch):
    def fake_import_module(name):
        if name in {
            "menace.self_debugger_sandbox",
            "menace_sandbox.self_debugger_sandbox",
            "self_debugger_sandbox",
        }:
            raise ModuleNotFoundError("candidate missing", name=name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        env._resolve_self_debugger_sandbox_class()

    message = str(exc_info.value)
    assert "menace.self_debugger_sandbox" in message
    assert "menace_sandbox.self_debugger_sandbox" in message
    assert "self_debugger_sandbox" in message
    assert "candidate missing ('menace.self_debugger_sandbox')" in message
    assert "candidate missing ('menace_sandbox.self_debugger_sandbox')" in message
    assert "candidate missing ('self_debugger_sandbox')" in message


def test_resolver_raises_import_error_when_nested_dependency_missing(monkeypatch):
    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            raise ModuleNotFoundError(
                "No module named 'missing_dependency'", name="missing_dependency"
            )
        if name in {
            "menace_sandbox.self_debugger_sandbox",
            "self_debugger_sandbox",
        }:
            raise ModuleNotFoundError("candidate missing", name=name)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        env._resolve_self_debugger_sandbox_class()

    message = str(exc_info.value)
    assert "dependency import failure" in message
    assert "missing_dependency" in message
    assert "self_debugger_sandbox" in message
