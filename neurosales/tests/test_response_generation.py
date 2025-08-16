import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.response_generation import ResponseCandidateGenerator, redundancy_filter


def test_redundancy_filter_removes_duplicates():
    res = redundancy_filter(["hello world", "hello world", "hello there"], threshold=0.5)
    assert len(res) == 2
    assert res[0] == "hello world"


def test_generate_candidates_pool():
    gen = ResponseCandidateGenerator()
    gen.add_past_response("Sure, I can assist you.")
    gen.add_past_response("Let me show you how to proceed.")
    candidates = gen.generate_candidates("I need help", ["Hi"], "helper")
    assert candidates
    assert len(candidates) == len(set(candidates))
