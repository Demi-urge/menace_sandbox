import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.adaptive_ranking import AdaptiveRanker


def test_rerank_updates_with_feedback():
    ranker = AdaptiveRanker(alpha=1.0, exploration=0.0)
    scores = {"a": 0.6, "b": 0.5, "c": 0.4}
    first = ranker.rerank("u1", scores)
    assert first[0] == "a"
    ranker.update_with_feedback("u1", "a", ignored=True)
    second = ranker.rerank("u1", scores)
    assert second[0] != "a"


def test_exploration_injects_diversity():
    ranker = AdaptiveRanker(alpha=0.5, exploration=1.0)
    scores = {"a": 0.9, "b": 0.2, "c": 0.3, "d": 0.25, "e": 0.1, "f": 0.05}
    res = ranker.rerank("u2", scores, history=["a old message"], top_n=5)
    assert "f" in res

