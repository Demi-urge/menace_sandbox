from pathlib import Path

from module_synergy_grapher import ModuleSynergyGrapher, get_synergy_cluster


def _build_sample_graph(tmp_path: Path):
    (tmp_path / "a.py").write_text(
        "import b\n\n\ndef a_func():\n    return 1\n"
    )
    (tmp_path / "b.py").write_text(
        "def b_func():\n    return 2\n"
    )
    grapher = ModuleSynergyGrapher()
    graph = grapher.build_graph(tmp_path)
    path = tmp_path / "sandbox_data" / "module_synergy_graph.json"
    return graph, path


def test_build_graph_basic(tmp_path: Path):
    graph, path = _build_sample_graph(tmp_path)
    assert set(graph.nodes) == {"a", "b"}
    assert set(graph.edges) == {("a", "b")}
    assert graph["a"]["b"]["weight"] == 1.0
    assert path.exists()


def test_get_synergy_cluster_thresholds(tmp_path: Path):
    _, path = _build_sample_graph(tmp_path)
    assert get_synergy_cluster("a", threshold=0.5, path=path) == {"a", "b"}
    assert get_synergy_cluster("a", threshold=1.1, path=path) == {"a"}
    assert get_synergy_cluster("b", threshold=0.5, path=path) == {"b"}
