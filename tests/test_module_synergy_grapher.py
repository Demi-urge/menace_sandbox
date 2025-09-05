from __future__ import annotations

from pathlib import Path
import json

import networkx as nx
import pytest

import module_synergy_grapher as msg
from module_synergy_grapher import ModuleSynergyGrapher, get_synergy_cluster
from menace import synergy_history_db as shd


# ---------------------------------------------------------------------------
# Fixtures


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    """Replace heavy embedding machinery with fast stubs."""

    # Default embeddings disabled
    monkeypatch.setattr(msg, "governed_embed", lambda text: [])


# ---------------------------------------------------------------------------
# Helpers


def _build_sample_graph(tmp_path: Path):
    (tmp_path / "a.py").write_text(  # path-ignore
        "import b\n\n\ndef a_func():\n    return 1\n"
    )
    (tmp_path / "b.py").write_text(  # path-ignore
        "def b_func():\n    return 2\n"
    )
    grapher = ModuleSynergyGrapher()
    graph = grapher.build_graph(tmp_path)
    path = tmp_path / "sandbox_data" / "module_synergy_graph.json"
    return graph, path


# ---------------------------------------------------------------------------
# Graph construction


def test_build_graph_basic(tmp_path: Path):
    graph, path = _build_sample_graph(tmp_path)
    assert set(graph.nodes) == {"a", "b"}
    assert set(graph.edges) == {("a", "b")}
    assert graph["a"]["b"]["weight"] == 1.0
    assert path.exists()


def test_update_graph(tmp_path: Path):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("")  # path-ignore
    grapher = ModuleSynergyGrapher()
    grapher.build_graph(tmp_path)
    assert ("a", "b") in grapher.graph.edges

    # Change module ``a`` to remove import, edge should disappear
    (tmp_path / "a.py").write_text("")  # path-ignore
    grapher.update_graph(["a"])
    assert ("a", "b") not in grapher.graph.edges

    # Add new module and ensure connections are added
    (tmp_path / "c.py").write_text("import a\n")  # path-ignore
    grapher.update_graph(["c"])
    assert "c" in grapher.graph
    assert ("c", "a") in grapher.graph.edges


def test_update_graph_removed_module(tmp_path: Path):
    """Modules deleted from disk should be pruned from the graph."""

    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("")  # path-ignore
    grapher = ModuleSynergyGrapher()
    grapher.build_graph(tmp_path)
    assert "b" in grapher.graph

    # Remove module ``b`` entirely
    (tmp_path / "b.py").unlink()  # path-ignore
    grapher.update_graph(["b"])

    assert "b" not in grapher.graph
    assert ("a", "b") not in grapher.graph.edges


def test_build_save_load_cluster(tmp_path: Path):
    """End-to-end test covering graph build, save/load and clustering."""

    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("")  # path-ignore

    grapher = ModuleSynergyGrapher()
    graph = grapher.build_graph(tmp_path)
    pkl_path = tmp_path / "graph.pkl"
    grapher.save(graph, pkl_path)

    loader = ModuleSynergyGrapher()
    loaded = loader.load(pkl_path)

    assert set(loaded.nodes) == {"a", "b"}
    assert set(loaded.edges) == {("a", "b")}
    assert loaded["a"]["b"]["weight"] == pytest.approx(1.0)
    assert loader.get_synergy_cluster("a", threshold=0.5) == {"a", "b"}


# ---------------------------------------------------------------------------
# Configuration handling


def test_config_overrides(tmp_path: Path):
    cfg = {"coefficients": {"import": 0.2}}
    grapher = ModuleSynergyGrapher(config=cfg)
    assert grapher.coefficients["import"] == 0.2

    json_path = tmp_path / "cfg.json"
    json_path.write_text(json.dumps({"coefficients": {"structure": 0.3}}))
    grapher = ModuleSynergyGrapher(config=json_path)
    assert grapher.coefficients["structure"] == 0.3

    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text('coefficients = {cooccurrence = 0.4}')
    grapher = ModuleSynergyGrapher(config=toml_path)
    assert grapher.coefficients["cooccurrence"] == 0.4


# ---------------------------------------------------------------------------
# Coefficient learning


def test_learn_coefficients_from_history(tmp_path: Path):
    (tmp_path / "a.py").write_text("import b\n")  # path-ignore
    (tmp_path / "b.py").write_text("")  # path-ignore

    db_path = tmp_path / "synergy_history.db"
    conn = shd.connect(db_path)
    try:
        shd.insert_entry(conn, {"a": 1.0, "b": 1.0})
    finally:
        conn.close()

    grapher = ModuleSynergyGrapher(
        coefficients={"import": 0.0, "structure": 0.0, "cooccurrence": 0.0, "embedding": 0.0}
    )
    grapher.learn_coefficients(tmp_path)
    weights_path = tmp_path / "sandbox_data" / "synergy_weights.json"
    assert weights_path.exists()
    assert grapher.coefficients["import"] > 0


# ---------------------------------------------------------------------------
# Heuristic components


def test_structure_similarity(tmp_path: Path):
    (tmp_path / "c.py").write_text("x = 1\n")  # path-ignore
    (tmp_path / "d.py").write_text("x = 2\n")  # path-ignore
    graph = ModuleSynergyGrapher().build_graph(tmp_path)
    assert graph.has_edge("c", "d")
    assert graph["c"]["d"]["weight"] == pytest.approx(1 / 3)


def test_workflow_cooccurrence(tmp_path: Path, monkeypatch):
    (tmp_path / "wfa.py").write_text("print('a')\n")  # path-ignore
    (tmp_path / "wfb.py").write_text("print('b')\n")  # path-ignore

    db_path = tmp_path / "workflows.db"
    db_path.touch()

    class DummyWorkflowDB:
        def __init__(self, path):  # pragma: no cover - simple stub
            class Conn:
                def execute(self, _query):
                    class Cur:
                        def fetchall(self):
                            return [("wfa,wfb", "")]

                    return Cur()

            self.conn = Conn()

    monkeypatch.setattr(msg, "WorkflowDB", DummyWorkflowDB)
    graph = ModuleSynergyGrapher().build_graph(tmp_path)
    assert graph["wfa"]["wfb"]["weight"] == pytest.approx(1.0)


def test_embedding_similarity(tmp_path: Path, monkeypatch):
    (tmp_path / "e.py").write_text('"""alpha beta"""')  # path-ignore
    (tmp_path / "f.py").write_text('"""alpha beta"""')  # path-ignore

    def fake_embed(text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    monkeypatch.setattr(msg, "governed_embed", fake_embed)
    graph = ModuleSynergyGrapher(embedding_threshold=0.5).build_graph(tmp_path)
    assert graph["e"]["f"]["weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Synergy clustering


def test_get_synergy_cluster_thresholds(tmp_path: Path):
    _, path = _build_sample_graph(tmp_path)
    assert get_synergy_cluster("a", threshold=0.5, path=path) == {"a", "b"}
    assert get_synergy_cluster("a", threshold=1.1, path=path) == {"a"}
    assert get_synergy_cluster("b", threshold=0.5, path=path) == {"b"}


def test_get_synergy_cluster_synthetic():
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=0.4)
    g.add_edge("b", "c", weight=0.4)
    g.add_edge("a", "d", weight=0.2)

    grapher = ModuleSynergyGrapher()
    grapher.graph = g
    assert grapher.get_synergy_cluster("a", threshold=0.7) == {"a", "c"}
    assert grapher.get_synergy_cluster("a", threshold=0.7, bfs=True) == {"a", "c"}
    assert grapher.get_synergy_cluster("a", threshold=0.3) == {"a", "b", "c"}


def test_get_synergy_cluster_cycle_terminates():
    """Graphs with cycles should not cause infinite loops during search."""

    g = nx.DiGraph()
    g.add_edge("a", "b", weight=0.6)
    g.add_edge("b", "a", weight=0.6)  # cycle with positive weight

    grapher = ModuleSynergyGrapher()
    grapher.graph = g

    assert grapher.get_synergy_cluster("a", threshold=0.5) == {"a", "b"}
    assert grapher.get_synergy_cluster("a", threshold=0.5, bfs=True) == {"a", "b"}


# ---------------------------------------------------------------------------
# Additional scoring behaviour


def test_build_graph_scores_all_metrics(tmp_path: Path, monkeypatch):
    """Edges combine import, structure, co-occurrence and embeddings."""

    (tmp_path / "mod1.py").write_text(  # path-ignore
        '"""module a"""\nimport mod2\n\n'
        'def foo():\n    pass\n\n'
        'class K:\n    pass\n'
    )
    (tmp_path / "mod2.py").write_text(  # path-ignore
        '"""module b"""\n'
        'def foo():\n    pass\n\n'
        'class K:\n    pass\n'
    )
    (tmp_path / "mod3.py").write_text('"""module c"""')  # path-ignore

    embed_map = {
        "module a": [1.0, 0.0],
        "module b": [0.8, 0.2],
        "module c": [0.0, 1.0],
    }
    monkeypatch.setattr(msg, "governed_embed", lambda text: embed_map[text])

    def fake_workflow(self, root, modules):
        return {("mod1", "mod3"): 3, ("mod3", "mod1"): 3}

    def fake_history(self, root, modules):
        return {("mod1", "mod3"): 2, ("mod3", "mod1"): 2}

    monkeypatch.setattr(ModuleSynergyGrapher, "_workflow_pairs", fake_workflow)
    monkeypatch.setattr(ModuleSynergyGrapher, "_history_pairs", fake_history)

    graph = ModuleSynergyGrapher().build_graph(tmp_path)

    assert graph["mod1"]["mod2"]["weight"] == pytest.approx(2.6368091668)
    assert graph["mod2"]["mod1"]["weight"] == pytest.approx(1.6368091668)
    assert graph["mod1"]["mod3"]["weight"] == pytest.approx(1.0)
    assert graph["mod3"]["mod1"]["weight"] == pytest.approx(1.0)
    assert "mod3" not in graph["mod2"]


def test_update_graph_only_recomputes_affected_edges(tmp_path: Path, monkeypatch):
    (tmp_path / "mod1.py").write_text(  # path-ignore
        '"""module a"""\nimport mod2\n\n'
        'def foo():\n    pass\n\n'
        'class K:\n    pass\n'
    )
    (tmp_path / "mod2.py").write_text(  # path-ignore
        '"""module b"""\n'
        'def foo():\n    pass\n\n'
        'class K:\n    pass\n'
    )
    (tmp_path / "mod3.py").write_text('"""module c"""')  # path-ignore

    embed_map = {
        "module a": [1.0, 0.0],
        "module b": [0.8, 0.2],
        "module c": [0.0, 1.0],
        "module b new": [0.0, 1.0],
    }

    monkeypatch.setattr(msg, "governed_embed", lambda text: embed_map[text])

    def fake_workflow(self, root, modules):
        return {("mod1", "mod3"): 3, ("mod3", "mod1"): 3}

    def fake_history(self, root, modules):
        return {("mod1", "mod3"): 2, ("mod3", "mod1"): 2}

    monkeypatch.setattr(ModuleSynergyGrapher, "_workflow_pairs", fake_workflow)
    monkeypatch.setattr(ModuleSynergyGrapher, "_history_pairs", fake_history)

    grapher = ModuleSynergyGrapher()
    graph = grapher.build_graph(tmp_path)
    before = {(a, b): d["weight"] for a, b, d in graph.edges(data=True)}

    (tmp_path / "mod2.py").write_text(  # path-ignore
        '"""module b new"""\n'
        'def bar():\n    pass\n\n'
        'class L:\n    pass\n'
    )

    grapher.update_graph(["mod2"])
    after = {(a, b): d["weight"] for a, b, d in grapher.graph.edges(data=True)}

    assert after[("mod1", "mod3")] == before[("mod1", "mod3")]
    assert after[("mod1", "mod2")] == pytest.approx(1.0)
    assert ("mod2", "mod1") not in after

    changed = {
        e for e in set(before) | set(after) if before.get(e) != after.get(e)
    }
    assert all("mod2" in e for e in changed)


def test_get_synergy_cluster_known_graph():
    g = nx.DiGraph()
    g.add_edge("a", "b", weight=0.5)
    g.add_edge("b", "c", weight=0.3)
    g.add_edge("a", "c", weight=0.2)

    grapher = ModuleSynergyGrapher(graph=g)
    assert grapher.get_synergy_cluster("a", threshold=0.5) == {"a", "b", "c"}
    assert grapher.get_synergy_cluster("a", threshold=0.6) == {"a", "c"}
