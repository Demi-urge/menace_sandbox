import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.trigger_graph import TriggerEffectGraph


def test_transition_and_next():
    graph = TriggerEffectGraph(["reward", "ego"], ttl_seconds=100)
    graph.record_transition("u1", "reward", "ego", True)
    graph.record_transition("u1", "reward", "none", False)
    nxt = graph.next_method("u1", "reward")
    assert nxt == "ego"


def test_reset_user():
    graph = TriggerEffectGraph(["a", "b"], ttl_seconds=100)
    graph.record_transition("u1", "a", "b", True)
    graph.reset_user("u1")
    nxt = graph.next_method("u1", "a")
    assert nxt == ""

