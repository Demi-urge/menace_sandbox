import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_func():
    path = ROOT / "sandbox_runner.py"
    src = path.read_text()
    tree = ast.parse(src)
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "discover_recursive_orphans",
            "discover_orphan_modules",
            "_discover_orphans_once",
        }:
            funcs.append(node)
    from typing import List, Iterable
    import os, json as js
    mod = {"ast": ast, "os": os, "json": js, "List": List, "Iterable": Iterable, "Path": Path}
    ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    code = ast.Module(body=funcs, type_ignores=[])
    exec(compile(code, str(path), "exec"), mod)
    return mod["discover_recursive_orphans"]


discover_recursive_orphans = _load_func()


def test_recursive_import_includes_dependencies(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("x = 1\n")
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"a.py": 0}))

    res = discover_recursive_orphans(str(tmp_path), module_map=data_dir / "module_map.json")
    assert sorted(res) == ["a", "b"]
