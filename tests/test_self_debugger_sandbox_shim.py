from __future__ import annotations

import builtins
import importlib
import sys
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


def test_human_alignment_flagger_shim_exports_collect_diff_data() -> None:
    module = importlib.import_module("menace.human_alignment_flagger")

    assert hasattr(
        module,
        "_collect_diff_data",
    ), "menace.self_debugger_sandbox_impl depends on menace.human_alignment_flagger._collect_diff_data remaining available."
    assert callable(
        module._collect_diff_data
    ), "menace.self_debugger_sandbox_impl expects menace.human_alignment_flagger._collect_diff_data to be callable."


def test_code_database_shim_reexports_hash_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "menace.code_database", raising=False)
    monkeypatch.delitem(sys.modules, "code_database", raising=False)

    module = importlib.import_module("menace.code_database")

    assert hasattr(
        module,
        "_hash_code",
    ), "menace.code_database should re-export _hash_code for packaged imports."
    if (module.__file__ or "").endswith("menace/code_database.py"):
        assert "_hash_code" in module.__all__

    from menace.code_database import _hash_code

    assert callable(_hash_code)
    assert _hash_code("shim-check") == module._hash_code("shim-check")
    assert _hash_code(b"shim-check") == module._hash_code(b"shim-check")
