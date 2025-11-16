import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.profiles import ProfileManager


def test_update_and_decay():
    manager = ProfileManager(decay_factor=0.5)
    manager.update_profile("u1", 1.0, 2.0, "type", 0.5)
    assert "u1" in manager.profiles
    profile = manager.profiles["u1"]
    assert profile.emotional_tone == 1.0
    assert profile.preference_scores["emotional_tone"] == 1.0

    manager.decay_profiles()
    assert profile.preference_scores["emotional_tone"] == 0.5
