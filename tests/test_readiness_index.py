import types

from menace_sandbox.readiness_index import (
    ReadinessIndex,
    bind_tracker,
    compute_readiness,
    evaluate_cycle,
    military_grade_readiness,
    readiness_summary,
)
from menace_sandbox.roi_tracker import ROITracker
from menace_sandbox.telemetry_backend import TelemetryBackend


def _setup_tracker() -> ROITracker:
    tracker = ROITracker()
    tracker.last_raroi = 0.5
    tracker.predicted_roi = [1.0]
    tracker.actual_roi = [1.0]
    tracker.metrics_history["synergy_safety_rating"] = [0.9]
    tracker.metrics_history["synergy_resilience"] = [0.8]
    tracker.synergy_reliability = lambda window=None: 0.7  # type: ignore
    tracker.last_confidence = 0.8
    return tracker


def test_compute_readiness_product():
    assert compute_readiness(0.5, 0.8, 0.7, 0.6) == 0.5 * 0.8 * 0.7 * 0.6


def test_evaluate_cycle_uses_tracker_raroi():
    tracker = _setup_tracker()
    score = evaluate_cycle(tracker, 0.9, 0.8, 0.7)
    assert score == compute_readiness(0.5, 0.9, 0.8, 0.7)


def test_readiness_summary_returns_metrics():
    tracker = _setup_tracker()
    bind_tracker(tracker)
    summary = readiness_summary("wf1")
    expected = compute_readiness(0.5, 1.0, 0.9, 0.8)
    assert summary["readiness"] == expected
    assert summary["workflow_id"] == "wf1"


def test_military_grade_matches_compute():
    assert military_grade_readiness(0.5, 0.8, 0.7, 0.6) == compute_readiness(
        0.5, 0.8, 0.7, 0.6
    )


def test_military_grade_readiness_product():
    assert military_grade_readiness(0.5, 0.8, 0.7, 0.6) == 0.5 * 0.8 * 0.7 * 0.6


def test_readiness_index_snapshot_scales_reliability():
    tracker = _setup_tracker()
    idx = ReadinessIndex(tracker)
    snap = idx.snapshot()
    expected = compute_readiness(0.5, 0.7 * 0.8, 0.9, 0.8)
    assert snap["readiness"] == expected
    assert snap["reliability"] == 0.7 * 0.8


def test_readiness_reduces_on_confidence_drop(tmp_path):
    backend = TelemetryBackend(str(tmp_path / "tel.db"))
    deltas = {
        "synergy_reliability": 0.9,
        "synergy_safety_rating": 1.0,
        "synergy_resilience": 1.0,
    }
    backend.log_prediction(
        "wf1", 1.0, 1.0, 1.0, deltas, False, None, scenario="base"
    )
    idx = ReadinessIndex(telemetry=backend)
    baseline = idx.readiness("wf1")
    assert baseline == military_grade_readiness(1.0, 0.9, 1.0, 1.0)

    backend.log_prediction(
        "wf1", 1.0, 1.0, 0.5, deltas, True, None, scenario="drift"
    )
    updated = idx.readiness("wf1")
    assert updated == military_grade_readiness(1.0, 0.9 * 0.5, 1.0, 1.0)
    assert updated < baseline
