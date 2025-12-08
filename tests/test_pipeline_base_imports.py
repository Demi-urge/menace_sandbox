"""Import smoke tests for lightweight bootstrap modules."""

from __future__ import annotations


def test_pipeline_base_imports_task_handoff_components(monkeypatch):
    """Task handoff dependencies should import cleanly during bootstrap."""

    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    from shared.pipeline_base import TaskHandoffBot, TaskInfo, TaskPackage, WorkflowDB

    assert TaskHandoffBot
    assert TaskInfo
    assert TaskPackage
    assert WorkflowDB
