import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.engagement_graph import EngagementGraph, pagerank, shortest_path


def test_record_and_best_next():
    g = EngagementGraph()
    g.record_interaction("forum", "neutral", "build_trust", True)
    g.record_interaction("forum", "neutral", "spam", False)
    nxt = g.best_next_strategy("forum", "neutral")
    assert nxt == "build_trust"


def test_pagerank_scores():
    graph = {
        "A": {"B": 1.0},
        "B": {"C": 1.0},
        "C": {"B": 1.0},
    }
    scores = pagerank(graph, max_iter=50)
    assert scores["B"] > scores["A"] and scores["B"] > scores["C"]


def test_shortest_path():
    graph = {
        "neutral": {"comment": 1.0, "alt": 2.0},
        "comment": {"success": 1.0},
        "alt": {"success": 2.0},
    }
    path = shortest_path(graph, "neutral", "success")
    assert path == ["neutral", "comment", "success"]
