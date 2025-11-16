import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.hierarchical_reward import HierarchicalRewardLearner


def test_kiln_and_audit_adjustment():
    hr = HierarchicalRewardLearner(alpha=1.0)
    bal = hr.record_interaction(
        "u1",
        engagement=1.0,
        sentiment=0.8,
        personalization=0.5,
        correct=True,
        context="emotional",
        profile="explorer",
    )
    assert bal.green == 2.0  # doubled for explorer
    assert bal.violet > 1.0  # emotional context boosts violet

    hr.record_interaction(
        "u2",
        engagement=0.9,
        sentiment=0.9,
        personalization=0.2,
        correct=False,
        context="emotional",
        profile="analyst",
    )
    before = hr.context_weights["emotional"]["iron"]
    hr.nightly_audit()
    after = hr.context_weights["emotional"]["iron"]
    assert after > before

