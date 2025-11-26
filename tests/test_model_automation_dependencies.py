"""Tests for :mod:`model_automation_dependencies`."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path
from typing import cast


def _load_dependencies_module():
    package_name = "menace_sandbox"
    package_path = Path(__file__).resolve().parents[1]

    sys.modules.pop("model_automation_dependencies", None)
    sys.modules.pop(f"{package_name}.model_automation_dependencies", None)

    package_spec = importlib.util.spec_from_loader(
        package_name, loader=None, origin=str(package_path)
    )
    if package_spec is not None:
        package_spec.submodule_search_locations = [str(package_path)]
    package = sys.modules.get(package_name)
    if package is None or package.__spec__ is None:
        package = importlib.util.module_from_spec(cast(importlib.machinery.ModuleSpec, package_spec))
        package.__path__ = [str(package_path)]
        package.__spec__ = package_spec
        sys.modules[package_name] = package

    module_spec = importlib.util.spec_from_file_location(
        f"{package_name}.model_automation_dependencies",
        package_path / "model_automation_dependencies.py",
        submodule_search_locations=[str(package_path)],
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    cast(importlib.machinery.SourceFileLoader, module_spec.loader).exec_module(module)
    return module


def test_planning_components_timeout_falls_back(monkeypatch) -> None:
    """Ensure planning components degrade gracefully when imports hang."""

    mad = _load_dependencies_module()
    mad._planning_components.cache_clear()
    mad._reset_planning_import_state()

    def _slow_import():
        time.sleep(mad._PLANNING_IMPORT_TIMEOUT + 0.1)
        raise AssertionError("import should time out before completion")

    monkeypatch.setattr(mad, "_import_planning_classes", _slow_import)
    monkeypatch.setattr(mad, "_PLANNING_IMPORT_TIMEOUT", 0.01)

    planner_cls, planning_task_cls, bot_plan_cls = mad._planning_components()

    assert planner_cls.__name__ == "_StubPlanner"
    assert planning_task_cls.__name__ == "_StubPlanningTask"
    assert bot_plan_cls.__name__ == "_StubBotPlan"


def test_planning_components_timeout_does_not_leak_threads(monkeypatch) -> None:
    """A timed-out import should not spawn additional helper threads on retries."""

    mad = _load_dependencies_module()
    mad._planning_components.cache_clear()
    mad._reset_planning_import_state()

    stuck_event = threading.Event()
    loader_calls = 0

    def _stuck_import():
        nonlocal loader_calls
        loader_calls += 1
        stuck_event.wait()
        raise AssertionError("stuck import should not complete")

    monkeypatch.setattr(mad, "_import_planning_classes", _stuck_import)
    monkeypatch.setattr(mad, "_PLANNING_IMPORT_TIMEOUT", 0.01)

    def _helper_threads() -> list[threading.Thread]:
        return [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("planning-import-loader")
        ]

    initial_threads = len(_helper_threads())

    planner_cls, planning_task_cls, bot_plan_cls = mad._planning_components()

    assert planner_cls.__name__ == "_StubPlanner"
    assert planning_task_cls.__name__ == "_StubPlanningTask"
    assert bot_plan_cls.__name__ == "_StubBotPlan"
    first_threads = len(_helper_threads())
    assert first_threads == initial_threads + 1

    mad._planning_components.cache_clear()
    planner_cls, planning_task_cls, bot_plan_cls = mad._planning_components()

    assert planner_cls.__name__ == "_StubPlanner"
    assert planning_task_cls.__name__ == "_StubPlanningTask"
    assert bot_plan_cls.__name__ == "_StubBotPlan"
    assert loader_calls == 1
    assert len(_helper_threads()) == first_threads

    stuck_event.set()

