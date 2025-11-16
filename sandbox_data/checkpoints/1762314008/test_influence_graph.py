import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.influence_graph import InfluenceGraph


def test_add_observation():
    g = InfluenceGraph()
    g.add_observation("apple", "buy", "happy", sentiment=0.8)
    assert "apple" in g.nodes
    assert ("apple", "buy") in g.edges
    edge = g.edges[("apple", "buy")]
    assert edge.count == 1 and edge.sentiment > 0.7


def test_merge_and_reweight():
    g = InfluenceGraph()
    g.add_observation("Apple", "purchase", "joy", sentiment=0.6)
    g.add_observation("apple", "purchase", "joy", sentiment=0.4)
    g.merge_redundant_nodes(threshold=0.8)
    names = [n for n in g.nodes if n.lower() == "apple"]
    assert len(names) == 1
    kept = names[0]
    g.reweight_links({(kept, "purchase"): 0.2}, decay=0.5)
    edge = g.edges[(kept, "purchase")]
    assert edge.sentiment > 0.2


def test_nightly_job():
    g = InfluenceGraph()
    for i in range(6):
        g.add_observation("u", f"t{i}", "e", sentiment=0.5)
    g.nightly_job({})
    assert any(name.endswith("_1") for name in g.nodes)
