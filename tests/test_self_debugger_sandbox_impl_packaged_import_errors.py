from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "menace" / "self_debugger_sandbox_impl.py"


def _load_import_helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    keep_names = {
        "_import_internal_module",
        "_raise_packaged_import_error",
        "_resolve_required_internal_import",
        "_is_missing_module_at_import_path",
    }
    selected_nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in keep_names
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])

    namespace = {
        "importlib": __import__("importlib"),
        "re": re,
        "__package__": "menace",
        "__name__": "menace.self_debugger_sandbox_impl",
        "_IS_PACKAGED_CONTEXT": True,
    }
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace


def _load_fallback_hash_code_impl():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))

    hash_code_node = None
    for node in tree.body:
        if not isinstance(node, ast.Try):
            continue
        has_code_database_import = any(
            isinstance(stmt, ast.ImportFrom)
            and stmt.module == "code_database"
            and any(alias.name == "_hash_code" for alias in stmt.names)
            for stmt in node.body
        )
        if not has_code_database_import:
            continue
        for handler in node.handlers:
            for stmt in handler.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "_hash_code":
                    hash_code_node = stmt
                    break
            if hash_code_node is not None:
                break
        if hash_code_node is not None:
            break

    assert hash_code_node is not None, "Unable to locate fallback _hash_code implementation"
    module = ast.Module(body=[hash_code_node], type_ignores=[])
    namespace = {"hashlib": __import__("hashlib")}
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_hash_code"]


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


def test_import_internal_module_prefers_deepest_nested_missing_module(monkeypatch):
    helpers = _load_import_helpers()

    def _fake_import_module(name, package=None):
        if name == ".automated_debugger" and package == "menace":
            top_level = ImportError("wrapper import failed")
            top_level.name = "menace.automated_debugger"  # type: ignore[attr-defined]
            nested = ModuleNotFoundError("No module named 'vendor.deep_dep'")
            nested.name = "vendor.deep_dep"  # type: ignore[attr-defined]
            top_level.__cause__ = nested
            raise top_level
        raise AssertionError(f"unexpected import attempt: name={name!r}, package={package!r}")

    monkeypatch.setattr(helpers["importlib"], "import_module", _fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        helpers["_import_internal_module"]("automated_debugger")

    assert "Missing nested dependency/import target: 'vendor.deep_dep'." in str(exc_info.value)


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


def test_packaged_import_error_includes_symbol_level_details_from_context_chain():
    helpers = _load_import_helpers()

    root_exc = ImportError("unable to resolve symbol")
    root_exc.name = "menace.sample_module"  # type: ignore[attr-defined]
    root_exc.__cause__ = AttributeError("module 'menace.sample_module' has no attribute 'required_symbol'")

    with pytest.raises(ImportError) as exc_info:
        helpers["_raise_packaged_import_error"](
            "menace.sample_module",
            root_exc,
            missing_symbol="required_symbol",
            import_path="packaged symbol resolution path",
        )

    message = str(exc_info.value)
    assert "Missing nested dependency/import target: 'menace.sample_module.required_symbol'." in message
    assert "Missing required symbol export: 'required_symbol'." in message


def test_resolve_required_internal_import_symbol_failure_enriches_missing_target(monkeypatch):
    helpers = _load_import_helpers()
    helpers["_import_internal_module"] = lambda _module_name: SimpleNamespace()

    with pytest.raises(ImportError) as exc_info:
        helpers["_resolve_required_internal_import"]("sample_module", "required_symbol")

    message = str(exc_info.value)
    assert "Missing nested dependency/import target: 'menace.sample_module.required_symbol'." in message
    assert "Missing required symbol export: 'required_symbol'." in message


def test_hash_code_fallback_is_deterministic_and_distinguishes_inputs():
    fallback_hash_code = _load_fallback_hash_code_impl()

    first = fallback_hash_code(b"alpha")
    second = fallback_hash_code(b"alpha")
    different = fallback_hash_code(b"beta")

    assert first == second
    assert first != different
    assert fallback_hash_code("alpha") == first
