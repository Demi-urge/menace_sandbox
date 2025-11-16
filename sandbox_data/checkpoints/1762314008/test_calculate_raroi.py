import numpy as np
import pytest
from typing import Mapping

import menace_sandbox.roi_tracker as rt
from menace_sandbox.roi_tracker import ROITracker
import menace_sandbox.self_test_service as sts


@pytest.mark.parametrize(
    "workflow_type, base_roi, roi_history, errors, test_status",
    [
        ("standard", 2.0, [1.0] * 10, 5.0, {"security": False, "alignment": False}),
        (
            "experimental",
            1.5,
            [1.0, 1.5, 0.5, 1.0],
            2.0,
            {"security": False, "alignment": True},
        ),
        ("critical", 1.0, [0.3, 0.4, 0.5, 0.6, 0.7], 8.0, {}),
    ],
)
def test_calculate_raroi_formula(
    workflow_type: str,
    base_roi: float,
    roi_history: list[float],
    errors: float,
    test_status: dict[str, bool],
    monkeypatch,
) -> None:
    tracker = ROITracker()
    tracker.roi_history = roi_history
    tracker._last_errors_per_minute = errors
    sts.set_failed_critical_tests([k for k, v in test_status.items() if not v])

    called: dict[str, Mapping[str, float]] = {}
    orig = rt._estimate_rollback_probability

    def fake_estimate(metrics: Mapping[str, float]) -> float:
        called["metrics"] = metrics
        return orig(metrics)

    monkeypatch.setattr(rt, "_estimate_rollback_probability", fake_estimate)

    base, raroi, _ = tracker.calculate_raroi(
        base_roi,
        workflow_type=workflow_type,
        metrics={"errors_per_minute": errors},
    )

    recent = tracker.roi_history[-tracker.window :]
    instability = np.std(recent) if recent else 0.0
    metrics_map = {"errors_per_minute": errors, "instability": instability}
    rollback_probability = orig(metrics_map)
    impact = tracker.impact_severity(workflow_type)
    stability_factor = max(0.0, 1.0 - instability)
    failing = [k for k, v in test_status.items() if not v]
    penalty = 1.0
    for k in failing:
        penalty *= rt.CRITICAL_TEST_PENALTIES.get(k, 1.0)

    expected = (
        base_roi
        * (1.0 - rollback_probability * impact)
        * stability_factor
        * penalty
    )

    assert base == base_roi
    assert raroi == pytest.approx(expected)
    assert called["metrics"] == metrics_map


@pytest.mark.parametrize(
    "impact_input, expected_clamped",
    [
        (1.5, 1.0),
        (-0.5, 0.0),
    ],
)
def test_calculate_raroi_clamps_impact_severity(
    impact_input: float, expected_clamped: float
) -> None:
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 5
    base, raroi, _ = tracker.calculate_raroi(
        1.0, rollback_prob=0.5, impact_severity=impact_input, metrics={}
    )

    expected = 1.0 * (1.0 - 0.5 * expected_clamped)
    assert base == 1.0
    assert raroi == pytest.approx(expected)


def test_multiple_failing_suites_penalties(monkeypatch) -> None:
    tracker = ROITracker()
    tracker.roi_history = [1.0] * 5
    sts.set_failed_critical_tests(["security", "alignment"])

    called: dict[str, Mapping[str, float]] = {}
    orig = ROITracker._safety_factor

    def fake_safety(self, metrics: Mapping[str, float]) -> float:
        called["metrics"] = dict(metrics)
        return orig(self, metrics)

    monkeypatch.setattr(ROITracker, "_safety_factor", fake_safety)

    base, raroi, _ = tracker.calculate_raroi(2.0, rollback_prob=0.0, metrics={})

    penalty = (
        rt.CRITICAL_TEST_PENALTIES["security"]
        * rt.CRITICAL_TEST_PENALTIES["alignment"]
    )

    assert base == 2.0
    assert raroi == pytest.approx(2.0 * penalty)
    assert called["metrics"]["security_failures"] == 1.0
    assert called["metrics"]["alignment_failures"] == 1.0


@pytest.mark.parametrize(
    "failing",
    [
        ["security"],
        {"security": False, "alignment": True},
    ],
)
def test_explicit_failing_tests(failing) -> None:
    tracker = ROITracker()
    tracker.roi_history = [0.1, 0.1, 0.1]
    sts.set_failed_critical_tests([])
    base, raroi, _ = tracker.calculate_raroi(
        1.0,
        workflow_type="standard",
        rollback_prob=0.0,
        failing_tests=failing,
        metrics={},
    )
    penalty = rt.CRITICAL_TEST_PENALTIES["security"]
    expected = 1.0 * (1 - 0.0 * 0.5) * (1 - np.std([0.1, 0.1, 0.1])) * penalty
    assert base == 1.0
    assert raroi == pytest.approx(expected)


def test_calculate_raroi_returns_bottlenecks(monkeypatch) -> None:
    tracker = ROITracker(raroi_borderline_threshold=0.8)
    monkeypatch.setattr(rt, "propose_fix", lambda m, p: [("profitability", "improve")])
    base, raroi, suggestions = tracker.calculate_raroi(
        1.0,
        workflow_type="standard",
        metrics={"profitability": 0.0},
        rollback_prob=1.0,
        impact_severity=1.0,
    )
    assert raroi == 0.0
    assert suggestions == [("profitability", "improve")]
