import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.scoring import ResponsePriorityQueue


def test_heap_order():
    queue = ResponsePriorityQueue()
    queue.add_response("r1", {"novelty": 0.5, "urgency": 0.1})
    queue.add_response("r2", {"novelty": 0.2, "urgency": 0.8})
    queue.add_response("r3", {"novelty": 0.9, "urgency": 0.2})
    best = queue.pop_best()
    assert best == "r3"


def test_update_metrics_changes_priority():
    queue = ResponsePriorityQueue()
    queue.add_response("a", {"novelty": 0.5})
    queue.add_response("b", {"novelty": 0.4})
    queue.update_metrics("b", ctr=0.3)
    best = queue.pop_best()
    assert best == "b"
