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


def test_resolver_uses_packaged_menace_module(monkeypatch):
    package_cls = type("PackageSelfDebuggerSandbox", (), {})

    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            return types.SimpleNamespace(SelfDebuggerSandbox=package_cls)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    assert env._resolve_self_debugger_sandbox_class() is package_cls


def test_self_debugger_candidates_are_packaged_only():
    assert env.SELF_DEBUGGER_SANDBOX_MODULE_CANDIDATES == (
        "menace.self_debugger_sandbox",
    )


def test_resolver_raises_module_not_found_when_packaged_module_missing(monkeypatch):
    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            raise ModuleNotFoundError(
                "No module named 'menace.self_debugger_sandbox'",
                name="menace.self_debugger_sandbox",
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        env._resolve_self_debugger_sandbox_class()

    message = str(exc_info.value)
    assert "menace.self_debugger_sandbox" in message
    assert "target module import; candidate missing ('menace.self_debugger_sandbox')" in message


def test_resolver_raises_import_error_when_nested_dependency_missing(monkeypatch):
    def fake_import_module(name):
        if name == "menace.self_debugger_sandbox":
            raise ModuleNotFoundError(
                "No module named 'missing_dependency'", name="missing_dependency"
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(env.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        env._resolve_self_debugger_sandbox_class()

    message = str(exc_info.value)
    assert "dependency import failure" in message
    assert "origin=nested dependency import" in message
    assert "missing_dependency" in message


def test_is_self_debugger_sandbox_import_failure_supports_packaged_candidate():
    for module_name in env.SELF_DEBUGGER_SANDBOX_MODULE_CANDIDATES:
        exc = ModuleNotFoundError(f"missing {module_name}", name=module_name)
        assert env.is_self_debugger_sandbox_import_failure(exc)
