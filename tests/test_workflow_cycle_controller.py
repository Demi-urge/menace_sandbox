from __future__ import annotations

from menace_sandbox.workflow_evolution_manager import WorkflowCycleController


def test_controller_halts_after_threshold() -> None:
    events: list[tuple[str, float]] = []

    def _callback(workflow_id: str, status: str, stats: dict[str, float]) -> None:
        events.append((status, float(stats.get("non_positive_streak", 0))))

    controller = WorkflowCycleController(
        roi_db=None,
        threshold=0.0,
        streak_required=2,
        status_callback=_callback,
    )

    active, stats = controller.record("wf", roi_delta=-0.1)
    assert active is True
    assert float(stats["non_positive_streak"]) == 1.0

    active, stats = controller.record("wf", roi_delta=-0.1)
    assert active is False
    assert events[-1][0] == "halted"
    assert events[-1][1] == 2.0


def test_controller_resumes_when_roi_recovers() -> None:
    events: list[str] = []

    controller = WorkflowCycleController(
        roi_db=None,
        threshold=0.0,
        streak_required=2,
        status_callback=lambda _wf, status, _stats: events.append(status),
    )

    controller.record("wf", roi_delta=-0.1)
    controller.record("wf", roi_delta=-0.2)  # triggers halt
    active, _ = controller.record("wf", roi_delta=0.5)

    assert active is True
    assert events == ["halted", "active"]
