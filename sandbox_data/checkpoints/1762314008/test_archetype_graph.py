import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.archetype_graph import ArchetypeGraph


def test_add_and_connect():
    g = ArchetypeGraph()
    g.add_archetype("A", influence=1.0, reputation=["trustworthy"], beliefs=[1.0, 0.0])
    g.add_archetype("B", influence=0.5)
    g.connect("A", "B", weight=0.8)
    assert "A" in g.nodes and g.nodes["A"].influence == 1.0
    assert g.edges["A"]["B"].weight == 0.8
    assert g.ledger


def test_decay_and_update():
    g = ArchetypeGraph(decay=0.9)
    g.add_archetype("X")
    g.add_archetype("Y")
    g.connect("X", "Y", weight=1.0)
    g.edges["X"]["Y"].last_updated -= 2
    g.decay_edges()
    old = g.edges["X"]["Y"].weight
    g.update_relationship("X", "Y", alignment=1.0, frequency=2.0, betrayal=True, validation=0.5)
    new = g.edges["X"]["Y"].weight
    assert new != old
    assert len(g.ledger) >= 2
