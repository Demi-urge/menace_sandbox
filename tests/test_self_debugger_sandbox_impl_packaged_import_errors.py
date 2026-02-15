from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "menace" / "self_debugger_sandbox_impl.py"


def _load_import_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    keep_names = {
        "_import_internal_module",
        "_raise_packaged_import_error",
        "_is_missing_module_at_import_path",
    }
    selected_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep_names
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])

    namespace = {
        "importlib": __import__("importlib"),
        "__package__": "menace",
        "__name__": "menace.self_debugger_sandbox_impl",
        "_IS_PACKAGED_CONTEXT": True,
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_import_internal_module_wraps_relative_import_failure_details(monkeypatch):
    helpers = _load_import_helpers()

    def _fake_import_module(name, package=None):
        if name == ".automated_debugger" and package == "menace":
            raise ImportError("attempted relative import with no known parent package")
        raise AssertionError(f"unexpected import attempt: name={name!r}, package={package!r}")

    monkeypatch.setattr(helpers["importlib"], "import_module", _fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        helpers["_import_internal_module"]("automated_debugger")

    message = str(exc_info.value)
    assert "Failed to import internal module 'menace.automated_debugger'" in message
    assert "Original exception from 'menace.automated_debugger': ImportError:" in message
    assert "attempted relative import with no known parent package" in message
    assert "Missing nested dependency/import target: 'unknown'" not in message


def test_packaged_import_error_uses_fallback_diagnostics_when_exc_name_missing():
    helpers = _load_import_helpers()

    with pytest.raises(ImportError) as exc_info:
        helpers["_raise_packaged_import_error"](
            "menace.sample_module",
            ImportError("cannot import dependency_x"),
            import_path="packaged import path",
        )

    message = str(exc_info.value)
    assert "fallback diagnostics: type=ImportError" in message
    assert "message=cannot import dependency_x" in message
    assert "importing=menace.sample_module" in message
    assert "Root cause: ImportError: cannot import dependency_x" in message


def test_packaged_import_error_keeps_missing_symbol_context():
    helpers = _load_import_helpers()

    with pytest.raises(ImportError) as exc_info:
        helpers["_raise_packaged_import_error"](
            "menace.sample_module",
            ImportError("symbol not exported"),
            missing_symbol="required_symbol",
        )

    assert "Missing required symbol export: 'required_symbol'." in str(exc_info.value)
