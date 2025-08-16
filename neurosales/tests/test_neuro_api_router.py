import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.neuro_api_router import NeuroAPIRouter
from unittest.mock import patch


def test_scores_degrade_on_failure_and_latency():
    def good():
        return "ok"

    def bad():
        raise RuntimeError("fail")

    router = NeuroAPIRouter({"good": good, "bad": bad})

    router.fetch("good")
    assert router.telemetry["good"].score == 1.0

    try:
        router.fetch("bad")
    except RuntimeError:
        pass
    assert router.telemetry["bad"].score < 1.0

    # simulate slow endpoint
    router.endpoints["slow"] = good
    router.telemetry["slow"] = router.telemetry["good"].__class__()
    with patch("time.time", side_effect=[0.0, 2.5]):
        router.fetch("slow")
    assert router.telemetry["slow"].score < 1.0


def test_audit_promotes_and_demotes():
    def good():
        return "ok"

    router = NeuroAPIRouter({"e1": good})
    router.telemetry["e1"].failures = 3
    router.telemetry["e1"].score = 1.0
    router.audit()
    assert router.telemetry["e1"].score < 1.0
    router.telemetry["e1"].score = 0.5
    router.audit()
    assert router.telemetry["e1"].score > 0.5


def test_model_routing_on_cost_spike():
    def local(prompt: str) -> str:
        return "local"

    def gpt4(prompt: str) -> str:
        return "gpt4"

    cost = [0.01]

    def hook() -> float:
        return cost[0]

    router = NeuroAPIRouter({}, gpt4_cost_hook=hook, local_model=local, gpt4_model=gpt4)
    assert router.route_model("hi") == "gpt4"
    cost[0] = 0.05
    assert router.route_model("hi") == "local"

