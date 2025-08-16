import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.human_feedback import HumanFeedbackManager


def test_ranking_and_fine_tune_queue():
    mgr = HumanFeedbackManager(decay_rate=1.0)
    mgr.register_user("power", weight=2.0)
    mgr.record_feedback("a", rating=1, user_id="power", certainty=0.5)
    ranked = mgr.adjust_ranking({"a": 0.4, "b": 0.5})
    assert ranked[0] == "a"
    mgr.record_feedback("a", rating=-1, user_id="power", certainty=1.0, correction="fix")
    ranked2 = mgr.adjust_ranking({"a": 0.4, "b": 0.5})
    assert ranked2[0] == "b"
    batch = mgr.mini_fine_tune(1)
    assert batch == [("a", "fix")]


def test_experience_decay():
    mgr = HumanFeedbackManager(decay_rate=0.5)
    mgr.record_feedback("x", rating=1, user_id="u", certainty=0.0)
    mgr.decay_scores()
    mgr.record_feedback("x", rating=0, user_id="u", certainty=0.0)
    assert mgr.response_scores["x"] < 1.0

