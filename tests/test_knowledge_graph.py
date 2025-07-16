import pytest
pytest.importorskip("networkx")
from menace.knowledge_graph import KnowledgeGraph


def test_add_and_query():
    kg = KnowledgeGraph()
    kg.add_memory_entry("k1", tags=["bot"])
    kg.add_code_snippet("snippet", bots=["bot"])
    kg.add_pathway("a->b")
    related = kg.related("bot:bot", depth=2)
    assert any(r.startswith("code:") for r in related)


def test_deep_root_cause_traversal():
    kg = KnowledgeGraph()
    # build a long dependency chain error->A->B->C->D->E->F->G
    prev = "bot:A"
    kg.graph.add_node(prev)
    kg.graph.add_node("error:1")
    kg.graph.add_edge("error:1", prev, type="bot")
    for n in ["B", "C", "D", "E", "F", "G"]:
        node = f"bot:{n}"
        kg.graph.add_node(node)
        kg.graph.add_edge(prev, node, type="depends")
        prev = node

    limited = kg.root_causes("G", hops=5)
    unlimited = kg.root_causes("G", hops=None)
    assert "error:1" not in limited
    assert "error:1" in unlimited


def test_suggest_root_cause_clusters():
    kg = KnowledgeGraph()
    kg.graph.add_node("bot:X")
    for i in range(5):
        enode = f"error:{i}"
        kg.graph.add_node(enode, message="boom")
        kg.graph.add_edge(enode, "bot:X", type="bot")

    clusters = kg.suggest_root_cause("X", hops=None, min_cluster_size=2)
    assert clusters
    assert all(isinstance(c, list) for c in clusters)
