import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
from neurosales.adaptive_exploration import AdaptiveExplorer


def test_choose_line_prefers_fresh_when_quota_high():
    random.seed(0)
    engine = AdaptiveExplorer(base_quota=0.6)
    choice = engine.choose_line(
        "u1",
        "newcomer",
        "sales",
        ["fresh", "old"],
        [True, False],
        confidence=0.5,
        engagement=0.5,
    )
    assert choice == "fresh"


def test_quota_adjustments_with_confidence_and_engagement():
    random.seed(1)
    engine = AdaptiveExplorer(base_quota=0.5)
    choice = engine.choose_line(
        "u2",
        "veteran",
        "tech",
        ["fresh", "old"],
        [True, False],
        confidence=0.8,
        engagement=0.7,
    )
    assert choice == "old"
    engine.record_feedback("tech", choice, success=True, engagement=0.9)
    low = engine.topic_quota["tech"]
    engine.record_feedback("tech", choice, success=True, engagement=0.0)
    engine.record_feedback("tech", choice, success=True, engagement=0.0)
    engine.record_feedback("tech", choice, success=True, engagement=0.0)
    assert engine.topic_quota["tech"] > low

