import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.social_power import SocialPowerRanker


def test_rankings_with_citations_and_wins():
    engine = SocialPowerRanker()
    engine.record_interaction("A", "B", cited_by=["B", "C"])
    engine.record_interaction("A", "C")
    scores = engine.rankings()
    assert scores["A"] > scores["B"] and scores["A"] > scores.get("C", 0)


def test_contradiction_and_decay():
    engine = SocialPowerRanker()
    engine.record_interaction("X", "Y")
    before = engine.rankings()["X"]
    engine.record_interaction("Y", "X", contradiction=True, behavioral_response=False)
    after = engine.rankings()["X"]
    assert after < before


def test_user_faction_assignment():
    engine = SocialPowerRanker()
    fac = engine.assign_user_faction("u1", {"red": 0.2, "blue": 0.5})
    assert fac == "blue"
    empty = engine.assign_user_faction("u1", {})
    assert empty == ""
