import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.rlhf import RLHFPolicyManager


def test_record_updates_weight_and_selects_best():
    mgr = RLHFPolicyManager(exploration_rate=0.0)
    mgr.record_result("a", ctr=0.1, sentiment=0.2, session=0.2)
    mgr.record_result("b", ctr=0.0, sentiment=0.0, session=0.0)
    best = mgr.best_response(["a", "b"])
    assert best == "a"


def test_prune_underperformers():
    mgr = RLHFPolicyManager(exploration_rate=0.0)
    mgr.record_result("x", ctr=-0.2, sentiment=-0.2, session=-0.2)
    mgr.record_result("y", ctr=0.1, sentiment=0.1, session=0.1)
    mgr.record_result("x", ctr=-0.2, sentiment=-0.2, session=-0.2)
    assert "x" not in mgr.weights
    best = mgr.best_response(["y"])
    assert best == "y"
