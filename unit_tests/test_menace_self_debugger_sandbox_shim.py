import builtins
import importlib.util
import types
from pathlib import Path

import pytest


SHIM_PATH = Path(__file__).resolve().parents[1] / "menace" / "self_debugger_sandbox.py"


def _load_shim_with_import_overrides(monkeypatch, overrides):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in overrides:
            result = overrides[name]
            if isinstance(result, Exception):
                raise result
            return result
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location("menace_self_debugger_sandbox_test", SHIM_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shim_falls_back_when_primary_module_is_missing(monkeypatch):
    fallback_cls = type("FallbackSelfDebuggerSandbox", (), {})

    module = _load_shim_with_import_overrides(
        monkeypatch,
        {
            "menace_sandbox.self_debugger_sandbox": ModuleNotFoundError(
                "No module named 'menace_sandbox.self_debugger_sandbox'",
                name="menace_sandbox.self_debugger_sandbox",
            ),
            "self_debugger_sandbox": types.SimpleNamespace(
                SelfDebuggerSandbox=fallback_cls
            ),
        },
    )

    assert module.SelfDebuggerSandbox is fallback_cls


def test_shim_propagates_internal_dependency_failure(monkeypatch):
    original_exc = ModuleNotFoundError(
        "No module named 'missing_dependency'", name="missing_dependency"
    )

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _load_shim_with_import_overrides(
            monkeypatch,
            {"menace_sandbox.self_debugger_sandbox": original_exc},
        )

    assert exc_info.value is original_exc
    assert exc_info.value.name == "missing_dependency"
