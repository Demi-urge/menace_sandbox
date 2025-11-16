import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.user_preferences import PreferenceEngine, PerformanceTracker, RoleplayCoach
from unittest.mock import patch


def test_preference_decay():
    engine = PreferenceEngine(window_seconds=1)
    engine.add_message("u1", "I love cats")
    assert engine.get_profile("u1").keyword_freq.get("cats", 0) > 0
    # force expiration
    for rec in list(engine.messages["u1"]):
        rec.timestamp -= 2
    engine.add_message("u1", "hi")
    assert "cats" not in engine.get_profile("u1").keyword_freq


def test_archetype_assignment():
    with patch("neurosales.embedding.embed_text") as et:
        et.side_effect = lambda t: [1.0] if "cats" in t else [0.0]
        engine = PreferenceEngine()
        engine.add_message("u1", "I like cats")
        engine.add_message("u2", "I like dogs")
        engine.assign_archetypes(k=2)
        arche1 = engine.get_profile("u1").archetype
        arche2 = engine.get_profile("u2").archetype
        assert arche1 != "" and arche2 != ""
        assert arche1 != arche2


def test_roleplay_and_performance():
    engine = PreferenceEngine()
    tracker = PerformanceTracker()
    coach = RoleplayCoach(engine, tracker)
    resp, score = coach.interact("u1", "hello there")
    assert "archetype" in resp
    assert score > 0
    history = tracker.history("u1")
    assert len(history) == 1
    assert history[0].score == score
