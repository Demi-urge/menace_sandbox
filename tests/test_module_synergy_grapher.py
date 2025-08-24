from __future__ import annotations

from pathlib import Path
import json

import networkx as nx
import pytest

import module_synergy_grapher as msg
from module_synergy_grapher import ModuleSynergyGrapher, get_synergy_cluster


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


# ---------------------------------------------------------------------------
# Graph construction


def test_build_graph_basic(tmp_path: Path):
    graph, path = _build_sample_graph(tmp_path)
    assert set(graph.nodes) == {"a", "b"}
    assert set(graph.edges) == {("a", "b")}
    assert graph["a"]["b"]["weight"] == 1.0
    assert path.exists()


def test_update_graph(tmp_path: Path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("")
    grapher = ModuleSynergyGrapher()
    grapher.build_graph(tmp_path)
    assert ("a", "b") in grapher.graph.edges

    # Change module ``a`` to remove import, edge should disappear
    (tmp_path / "a.py").write_text("")
    grapher.update_graph(["a"])
    assert ("a", "b") not in grapher.graph.edges

    # Add new module and ensure connections are added
    (tmp_path / "c.py").write_text("import a\n")
    grapher.update_graph(["c"])
    assert "c" in grapher.graph
    assert ("c", "a") in grapher.graph.edges


def test_build_save_load_cluster(tmp_path: Path):
    """End-to-end test covering graph build, save/load and clustering."""

    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("")

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
# Heuristic components


def test_structure_similarity(tmp_path: Path):
    (tmp_path / "c.py").write_text("x = 1\n")
    (tmp_path / "d.py").write_text("x = 2\n")
    graph = ModuleSynergyGrapher().build_graph(tmp_path)
    assert graph.has_edge("c", "d")
    assert graph["c"]["d"]["weight"] == pytest.approx(1 / 3)


def test_workflow_cooccurrence(tmp_path: Path, monkeypatch):
    (tmp_path / "wfa.py").write_text("print('a')\n")
    (tmp_path / "wfb.py").write_text("print('b')\n")

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
    (tmp_path / "e.py").write_text('"""alpha beta"""')
    (tmp_path / "f.py").write_text('"""alpha beta"""')

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
