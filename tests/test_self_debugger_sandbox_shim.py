from __future__ import annotations

import builtins
from pathlib import Path
from types import ModuleType

import pytest


SHIM_PATH = Path(__file__).resolve().parent.parent / "menace" / "self_debugger_sandbox.py"


def _exec_shim(custom_import):
    source = SHIM_PATH.read_text(encoding="utf-8")
    shim_builtins = dict(vars(builtins))
    shim_builtins["__import__"] = custom_import
    globals_dict = {
        "__name__": "test_shim_module",
        "__file__": str(SHIM_PATH),
        "__builtins__": shim_builtins,
    }
    exec(compile(source, str(SHIM_PATH), "exec"), globals_dict)
    return globals_dict["SelfDebuggerSandbox"]


def test_package_import_succeeds_without_flat_fallback() -> None:
    pkg_cls = type("PkgSelfDebuggerSandbox", (), {})
    flat_cls = type("FlatSelfDebuggerSandbox", (), {})

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "menace_sandbox.self_debugger_sandbox":
            mod = ModuleType(name)
            mod.SelfDebuggerSandbox = pkg_cls
            return mod
        if name == "self_debugger_sandbox":
            mod = ModuleType(name)
            mod.SelfDebuggerSandbox = flat_cls
            return mod
        return builtins.__import__(name, globals, locals, fromlist, level)

    resolved = _exec_shim(_import)
    assert resolved is pkg_cls


def test_package_missing_uses_flat_fallback() -> None:
    flat_cls = type("FlatSelfDebuggerSandbox", (), {})

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "menace_sandbox.self_debugger_sandbox":
            raise ModuleNotFoundError("No module named 'menace_sandbox'", name="menace_sandbox")
        if name == "self_debugger_sandbox":
            mod = ModuleType(name)
            mod.SelfDebuggerSandbox = flat_cls
            return mod
        return builtins.__import__(name, globals, locals, fromlist, level)

    resolved = _exec_shim(_import)
    assert resolved is flat_cls


def test_nested_dependency_missing_is_not_masked_by_flat_fallback() -> None:
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "menace_sandbox.self_debugger_sandbox":
            raise ModuleNotFoundError("No module named 'missing_dep'", name="missing_dep")
        if name == "self_debugger_sandbox":
            raise AssertionError("flat fallback import should not be attempted")
        return builtins.__import__(name, globals, locals, fromlist, level)

    with pytest.raises(ModuleNotFoundError, match="missing_dep") as exc_info:
        _exec_shim(_import)

    assert exc_info.value.name == "missing_dep"
