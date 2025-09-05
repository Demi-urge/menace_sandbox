import pytest

pytest.importorskip("networkx")

from module_mapper import build_module_graph, cluster_modules


def test_cluster_strongly_connected(tmp_path):
    (tmp_path / "a.py").write_text("import b\nb.g()\n")  # path-ignore
    (tmp_path / "b.py").write_text("import a\n\ndef g():\n    a.f()\n")  # path-ignore
    (tmp_path / "c.py").write_text("def f():\n    pass\n")  # path-ignore

    graph = build_module_graph(tmp_path)
    clusters = cluster_modules(graph)
    mapping = {mod: cid for cid, mods in clusters.items() for mod in mods}
    assert mapping["a"] == mapping["b"]
