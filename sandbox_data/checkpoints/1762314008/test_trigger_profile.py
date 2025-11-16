import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.trigger_profile import TriggerProfileScorer


def test_score_interaction_updates_profile_and_returns_best():
    scorer = TriggerProfileScorer(["novelty", "authority", "tribal"])
    best = scorer.score_interaction("u1", {"novelty": 0.3, "authority": 0.6})
    assert best == "authority"
    profile = scorer.get_profile("u1")
    assert profile.scores["novelty"] == 0.3
    assert profile.scores["authority"] == 0.6
