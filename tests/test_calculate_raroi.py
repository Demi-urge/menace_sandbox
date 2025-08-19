import numpy as np
import pytest

import menace_sandbox.roi_tracker as rt
from menace_sandbox.roi_tracker import ROITracker


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
) -> None:
    tracker = ROITracker()
    tracker.roi_history = roi_history
    tracker._last_errors_per_minute = errors
    tracker._last_test_failures = [k for k, v in test_status.items() if not v]

    base, raroi = tracker.calculate_raroi(
        base_roi,
        workflow_type=workflow_type,
    )

    recent = tracker.roi_history[-tracker.window :]
    instability = np.std(recent) if recent else 0.0
    error_prob = max(0.0, min(1.0, errors / 10.0))
    rollback_probability = min(1.0, max(instability, error_prob))
    impact = tracker.impact_severity(workflow_type)
    stability_factor = max(0.0, 1.0 - instability)
    failing = [k for k, v in test_status.items() if not v]
    safety_factor = 0.5 if any(k in failing for k in rt.CRITICAL_SUITES) else 1.0

    expected = (
        base_roi
        * (1.0 - rollback_probability * impact)
        * stability_factor
        * safety_factor
    )

    assert base == base_roi
    assert raroi == pytest.approx(expected)
