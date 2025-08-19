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

    base, raroi = tracker.calculate_raroi(
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
