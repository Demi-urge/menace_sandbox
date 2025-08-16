import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.adaptive_influence_propagation import AdaptiveInfluencePropagator


def test_time_lagged_propagation():
    edges = {"A": {"B": 1.0}, "B": {"C": 1.0, "D": 1.0}}
    prop = AdaptiveInfluencePropagator(edges, base_decay=0.99, lag_factor=0.5)
    prop.record_shift("A", 10.0)
    prop.propagate()
    assert prop.influence.get("B", 0.0) > 0.0
    first = prop.influence["B"]
    prop.events[0].timestamp -= 3600
    prop.propagate()
    assert prop.influence.get("C", 0.0) > 0.0
    assert prop.influence.get("D", 0.0) > 0.0
    assert prop.influence["B"] > first
