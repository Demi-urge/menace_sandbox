import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.faction_influence import FactionInfluenceEngine


def test_alliance_propagation():
    eng = FactionInfluenceEngine()
    eng.add_faction("A")
    eng.add_faction("B")
    eng.add_faction("C")
    eng.form_alliance("A", "B")
    eng.record_interaction("A", "C")
    assert eng.factions["A"].rating > 1500
    assert eng.factions["B"].rating > 1500
    assert eng.factions["C"].rating < 1500


def test_instability_and_coalition_boost():
    eng = FactionInfluenceEngine()
    eng.add_faction("X", rating=2500)
    eng.add_faction("Y")
    eng.add_faction("Z")
    eng.form_coalition(["Y", "Z"], "X")
    eng.record_interaction("X", "Y")
    prev = eng.factions["X"].rating
    eng.record_interaction("Y", "X", triggers=["vmPFC", "NAcc"])
    assert eng.factions["Y"].rating > 1500
    assert eng.factions["X"].rating < prev
