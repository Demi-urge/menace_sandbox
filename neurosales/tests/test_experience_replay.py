import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.experience_replay import ExperienceReplayBuffer


def test_priority_sampling():
    buf = ExperienceReplayBuffer(max_size=10, decay=1.0)
    buf.add_exchange("hi", [0.1], [0.0], category="chat")
    buf.add_exchange("boom", [1.0], [0.0], flags={"rage_quit": True}, category="chat")
    counts = {"hi": 0, "boom": 0}
    for _ in range(20):
        sample = buf.sample(1)[0]
        counts[sample.text] += 1
    assert counts["boom"] > counts["hi"]


def test_reward_retrofit_and_confidence():
    buf = ExperienceReplayBuffer(max_size=5, decay=1.0)
    idx = buf.add_exchange("oops", [0.0], [0.0], flags={"error": True}, category="tech")
    assert buf.confidence_scale("tech") < 1.0
    buf.retrofit_reward(idx, 2.0)
    sample = buf.sample(1)[0]
    assert sample.reward_stack[-1] == 2.0
