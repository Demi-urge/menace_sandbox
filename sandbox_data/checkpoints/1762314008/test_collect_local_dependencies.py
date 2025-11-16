import ast
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pytest
from dynamic_path_router import resolve_path


def _load_resolver(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    src = resolve_path("self_test_service.py")
    tree = ast.parse(src.read_text(), filename=str(src))
    func_node = None
    for node in tree.body:
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                for sub in handler.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == "collect_local_dependencies":
                        func_node = sub
                        break
                if func_node:
                    break
        if func_node:
            break
    assert func_node is not None
    module = ast.Module([func_node], type_ignores=[])
    code = compile(module, str(src), "exec")
    env: dict[str, object] = {}
    exec(
        code,
        {
            "Path": Path,
            "Iterable": Iterable,
            "Mapping": Mapping,
            "Callable": Callable,
            "os": os,
            "ast": ast,
        },
        env,
    )
    return env["collect_local_dependencies"]


def test_cyclic_imports(tmp_path, monkeypatch):
    collect = _load_resolver(tmp_path, monkeypatch)
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("import a\n")  # path-ignore
    deps = collect([str(tmp_path / "a.py")])  # path-ignore
    assert deps == {"a.py", "b.py"}  # path-ignore


def test_nested_namespace_package(tmp_path, monkeypatch):
    collect = _load_resolver(tmp_path, monkeypatch)
    pkg = tmp_path / "pkg"
    subpkg = pkg / "subpkg"
    nested = subpkg / "nested"
    nested.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .subpkg import *\n")
    (subpkg / "mod1.py").write_text("\n")  # path-ignore
    (nested / "mod2.py").write_text("\n")  # path-ignore
    deps = collect([str(pkg / "__init__.py")])
    assert {"pkg/__init__.py", "pkg/subpkg/mod1.py", "pkg/subpkg/nested/mod2.py"} <= deps  # path-ignore


def test_missing_module_error(tmp_path, monkeypatch):
    collect = _load_resolver(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        collect([str(tmp_path / "missing.py")])  # path-ignore

