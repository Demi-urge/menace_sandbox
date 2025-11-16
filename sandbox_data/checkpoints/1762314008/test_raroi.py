import pytest
from typing import Mapping
import menace_sandbox.roi_tracker as rt
from menace_sandbox.roi_tracker import ROITracker
import menace_sandbox.self_test_service as sts
from menace_sandbox.borderline_bucket import BorderlineBucket


def test_raroi_formula(monkeypatch):
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 10
    base_roi = 2.0
    monkeypatch.setattr(rt.np, "std", lambda arr: 0.0)
    monkeypatch.setattr(rt, "get_impact_severity", lambda wf: 0.4)
    failing = ["security"]
    sts.set_failed_critical_tests(failing)

    called: dict[str, Mapping[str, float]] = {}

    def fake_estimate(metrics: Mapping[str, float]) -> float:
        called["metrics"] = metrics
        return 0.25

    monkeypatch.setattr(rt, "_estimate_rollback_probability", fake_estimate)

    base, raroi, _ = tracker.calculate_raroi(
        base_roi,
        workflow_type="standard",
        metrics={"errors_per_minute": 2.5},
    )
    expected = base_roi * (1 - 0.25 * 0.4)
    penalty = 1.0
    for k in failing:
        penalty *= rt.CRITICAL_TEST_PENALTIES.get(k, 1.0)
    expected *= penalty
    assert base == base_roi
    assert raroi == pytest.approx(expected)
    assert called["metrics"] == {"errors_per_minute": 2.5, "instability": 0.0}


def test_raroi_ranking_order(monkeypatch):
    tracker = ROITracker()
    monkeypatch.setattr(rt.np, "std", lambda arr: 0.0)
    tracker.update(
        0.0,
        2.0,
        modules=["a"],
        metrics={"errors_per_minute": 0.0, "security": True},
    )
    tracker.update(
        0.0,
        3.0,
        modules=["b"],
        metrics={"errors_per_minute": 8.0, "security": True},
    )
    sts.set_failed_critical_tests([])
    ranking = tracker.rankings()
    assert ranking[0][0] == "a"
    assert ranking[0][2] < ranking[1][2]
    assert ranking[0][1] > ranking[1][1]

def test_raroi_high_risk_instability_and_failures(monkeypatch):
    tracker = ROITracker()
    tracker.roi_history = [1.0, 2.0, 3.0]
    base_roi = 2.0
    # High instability
    monkeypatch.setattr(rt.np, "std", lambda arr: 0.8)
    # High impact severity and rollback probability via metrics
    called: dict[str, Mapping[str, float]] = {}
    def fake_estimate(metrics: Mapping[str, float]) -> float:
        called["metrics"] = metrics
        return 0.9
    monkeypatch.setattr(rt, "_estimate_rollback_probability", fake_estimate)
    monkeypatch.setattr(rt, "get_impact_severity", lambda wf: 0.9)
    failing = ["security", "alignment"]
    base, raroi, _ = tracker.calculate_raroi(
        base_roi,
        metrics={"errors_per_minute": 10.0},
        failing_tests=failing,
    )
    catastrophic_risk = 0.9 * 0.9
    stability_factor = 1 - 0.8
    penalty = rt.CRITICAL_TEST_PENALTIES["security"] * rt.CRITICAL_TEST_PENALTIES["alignment"]
    expected = base_roi * (1 - catastrophic_risk) * stability_factor * penalty
    assert base == base_roi
    assert raroi == pytest.approx(expected)
    assert called["metrics"]["errors_per_minute"] == 10.0
    assert called["metrics"]["instability"] == 0.8


@pytest.fixture()
def tracker_with_candidate(tmp_path):
    bucket = BorderlineBucket(str(tmp_path / "b.jsonl"))
    tracker = ROITracker(
        raroi_borderline_threshold=0.1, borderline_bucket=bucket
    )
    tracker.workflow_confidence_scores["wf1"] = 0.9
    tracker.score_workflow("wf1", 0.05)
    return tracker, bucket


def test_borderline_promotion(tracker_with_candidate):
    tracker, bucket = tracker_with_candidate
    tracker.process_borderline_candidates(lambda wf, info: 0.2)
    cand = bucket.get_candidate("wf1")
    assert cand["status"] == "promoted"
    assert cand["raroi"][-1] == 0.2


def test_borderline_termination(tracker_with_candidate):
    tracker, bucket = tracker_with_candidate
    tracker.process_borderline_candidates(lambda wf, info: 0.01)
    cand = bucket.get_candidate("wf1")
    assert cand["status"] == "terminated"
    assert cand["raroi"][-1] == 0.01


def test_low_raroi_or_confidence_added_to_bucket(tmp_path):
    bucket = BorderlineBucket(str(tmp_path / "b.jsonl"))
    tracker = ROITracker(
        raroi_borderline_threshold=0.1,
        confidence_threshold=0.8,
        borderline_bucket=bucket,
    )
    tracker.workflow_confidence_scores["wf_low_raroi"] = 0.9
    _final, review, _conf = tracker.score_workflow("wf_low_raroi", 0.05)
    assert not review
    assert bucket.get_candidate("wf_low_raroi") is not None

    tracker.workflow_confidence_scores["wf_low_conf"] = 0.5
    _final, review, _conf = tracker.score_workflow("wf_low_conf", 0.5)
    assert review
    cand = bucket.get_candidate("wf_low_conf")
    assert cand is not None
    assert cand["confidence"] == 0.5
