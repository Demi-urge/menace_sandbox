import json
from pathlib import Path

from scripts.generate_module_map import generate_module_map
from module_graph_analyzer import build_import_graph, cluster_modules


def test_generate_module_map(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text("from . import b\n\ndef call_b():\n    b.func_b()\n")
    (pkg / "b.py").write_text("from .c import func_c\n\ndef func_b():\n    func_c()\n")
    (pkg / "c.py").write_text("def func_c():\n    pass\n")
    (tmp_path / "d.py").write_text("def d():\n    pass\n")

    out = tmp_path / "map.json"
    mapping = generate_module_map(out, root=tmp_path)
    data = json.loads(out.read_text())
    assert mapping == data
    group_abc = {mapping['pkg/a'], mapping['pkg/b'], mapping['pkg/c']}
    assert len(group_abc) == 1
    assert mapping['d'] not in group_abc


def test_cluster_modules_direct(tmp_path):
    (tmp_path / "a.py").write_text("def a(): pass\n")
    (tmp_path / "b.py").write_text("import a\n\na.a()\n")
    graph = build_import_graph(tmp_path)
    mapping = cluster_modules(graph)
    assert mapping['a'] == mapping['b']
