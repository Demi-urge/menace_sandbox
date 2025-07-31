import ast
from pathlib import Path

def _load_func():
    path = Path(__file__).resolve().parents[1] / "sandbox_runner.py"
    src = path.read_text()
    tree = ast.parse(src)
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "discover_orphan_modules",
            "_discover_orphans_once",
        }:
            funcs.append(node)
    if not funcs:
        raise AssertionError("function not found")
    from typing import List, Iterable
    import os
    mod = {"ast": ast, "os": os, "List": List, "Iterable": Iterable}
    ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    code = ast.Module(body=funcs, type_ignores=[])
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
