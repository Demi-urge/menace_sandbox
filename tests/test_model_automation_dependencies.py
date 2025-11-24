"""Tests for :mod:`model_automation_dependencies`."""

from __future__ import annotations

import time


def test_planning_components_timeout_falls_back(monkeypatch) -> None:
    """Ensure planning components degrade gracefully when imports hang."""

    import model_automation_dependencies as mad

    mad._planning_components.cache_clear()

    def _slow_import():
        time.sleep(mad._PLANNING_IMPORT_TIMEOUT + 0.1)
        raise AssertionError("import should time out before completion")

    monkeypatch.setattr(mad, "_import_planning_classes", _slow_import)
    monkeypatch.setattr(mad, "_PLANNING_IMPORT_TIMEOUT", 0.01)

    planner_cls, planning_task_cls, bot_plan_cls = mad._planning_components()

    assert planner_cls.__name__ == "_StubPlanner"
    assert planning_task_cls.__name__ == "_StubPlanningTask"
    assert bot_plan_cls.__name__ == "_StubBotPlan"

