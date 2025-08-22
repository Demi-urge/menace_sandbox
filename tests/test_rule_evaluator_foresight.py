import menace.deployment_governance as dg
import pytest


class DummyTracker:
    pass


class DummyForecaster:
    def __init__(self, tracker):
        self.tracker = tracker

    def forecast(self, workflow_id, patch):
        return dg.ForecastResult([], 1.0, "u")


class DummyGraph:
    def simulate_impact_wave(self, *args, **kwargs):
        return {}


def _make_scorecard():
    return {
        "alignment_status": "pass",
        "security_status": "pass",
        "raroi": 1.3,
        "confidence": 0.9,
        "scenario_scores": {"s": 0.9},
    }


def test_gate_pass(monkeypatch):
    called = {}

    def fake_gate(workflow_id, patch, *, forecaster, tracker, graph, roi_threshold=0.0):
        called["wf"] = workflow_id
        called["patch"] = patch
        return True, [], dg.ForecastResult([], 0.0, "fid")

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "UpgradeForecaster", lambda tracker: DummyForecaster(tracker))
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: DummyGraph())

    result = dg.evaluate_scorecard(
        _make_scorecard(),
        patch=["diff"],
        foresight_tracker=DummyTracker(),
        workflow_id="wf1",
    )

    assert result["decision"] == "promote"
    assert result["reason_codes"] == ["meets_promotion_criteria"]
    assert result["foresight"]["forecast"] == {
        "projections": [],
        "confidence": 0.0,
        "upgrade_id": "fid",
    }
    assert result["foresight"]["reason_codes"] == []
    assert called["wf"] == "wf1"
    assert called["patch"] == ["diff"]


def test_gate_failure_borderline(monkeypatch, tmp_path):
    def fake_gate(workflow_id, patch, *, forecaster, tracker, graph, roi_threshold=0.0):
        return False, ["r1", "r2"], dg.ForecastResult([], 0.0, "fid")

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "UpgradeForecaster", lambda tracker: DummyForecaster(tracker))
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: DummyGraph())

    bucket = dg.BorderlineBucket(tmp_path / "b.jsonl")
    result = dg.evaluate_scorecard(
        _make_scorecard(),
        patch="p",
        foresight_tracker=DummyTracker(),
        workflow_id="wf1",
        borderline_bucket=bucket,
    )

    assert result["decision"] == "borderline"
    assert "r1" in result["reason_codes"] and "r2" in result["reason_codes"]
    assert result["foresight"]["reason_codes"] == ["r1", "r2"]


def test_gate_failure_pilot(monkeypatch):
    def fake_gate(workflow_id, patch, *, forecaster, tracker, graph, roi_threshold=0.0):
        return False, ["bad"], dg.ForecastResult([], 0.0, "fid")

    monkeypatch.setattr(dg, "is_foresight_safe_to_promote", fake_gate)
    monkeypatch.setattr(dg, "UpgradeForecaster", lambda tracker: DummyForecaster(tracker))
    monkeypatch.setattr(dg, "WorkflowGraph", lambda: DummyGraph())

    result = dg.evaluate_scorecard(
        _make_scorecard(),
        patch="p",
        foresight_tracker=DummyTracker(),
        workflow_id="wf1",
    )

    assert result["decision"] == "pilot"
    assert "bad" in result["reason_codes"]
    assert result["foresight"]["reason_codes"] == ["bad"]
