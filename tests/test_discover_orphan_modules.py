import ast
from pathlib import Path

def _load_func():
    path = Path(__file__).resolve().parents[1] / "sandbox_runner" / "orphan_discovery.py"
    src = path.read_text()
    tree = ast.parse(src)
    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "discover_orphan_modules":
            func = node
            break
    if func is None:
        raise AssertionError("function not found")
    from typing import List
    import os
    mod = {"ast": ast, "os": os, "List": List}
    ast.fix_missing_locations(func)
    code = ast.Module(body=[func], type_ignores=[])
    exec(compile(code, str(path), "exec"), mod)
    return mod["discover_orphan_modules"]

discover_orphan_modules = _load_func()


def test_orphan_detection(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "main.py").write_text("from . import util\n")
    (pkg / "util.py").write_text("def util(): pass\n")
    (pkg / "helper.py").write_text("def helper(): pass\n")
    (tmp_path / "cli.py").write_text("if __name__ == '__main__':\n    pass\n")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("import pkg.main\n")

    orphans = discover_orphan_modules(str(tmp_path), recursive=False)
    assert "pkg.helper" in orphans
    assert "pkg.util" not in orphans
    assert "cli" not in orphans


def test_recursive_chain_detection(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("x = 1\n")

    non_rec = discover_orphan_modules(str(tmp_path), recursive=False)
    rec = discover_orphan_modules(str(tmp_path))

    assert non_rec == ["a"]
    assert rec == ["a", "b", "c"]


def test_orphan_import_includes_dependency(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")

    res = discover_orphan_modules(str(tmp_path))
    assert sorted(res) == ["a", "b"]


def test_shared_dependency_not_included(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "d.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("import c\n")
    (tmp_path / "c.py").write_text("x = 1\n")
    (tmp_path / "e.py").write_text("import b\nimport f\n")
    (tmp_path / "f.py").write_text("import e\n")

    res = discover_orphan_modules(str(tmp_path), recursive=True)
    assert sorted(res) == ["a", "d"]
