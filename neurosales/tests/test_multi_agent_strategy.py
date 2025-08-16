import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.multi_agent_strategy import MultiAgentStrategy
from neurosales.faction_influence import FactionInfluenceEngine


def test_alliance_voting_and_memory():
    engine = FactionInfluenceEngine()
    members = {"A": ["a1", "a2"], "B": ["b1"]}
    mas = MultiAgentStrategy(members, engine)
    formed = mas.propose_alliance("A", "B")
    assert formed
    hist = mas.memory.history("A", "B")
    assert hist and hist[0].action == "proposed_alliance"


def test_betrayal_discourages_alliance():
    engine = FactionInfluenceEngine()
    members = {"X": ["x1"], "Y": ["y1"]}
    mas = MultiAgentStrategy(members, engine)
    assert mas.propose_alliance("X", "Y")
    mas.record_betrayal("Y", "X")
    refused = mas.propose_alliance("X", "Y")
    assert not refused
