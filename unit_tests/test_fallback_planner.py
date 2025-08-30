"""Tests for the fallback planner implementation."""

from __future__ import annotations

import ast
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence
import pytest


class DummyResult:
    def __init__(self, roi_gain: float) -> None:
        self.roi_gain = roi_gain


class DummyROI:
    def __init__(self, data):
        self.data = data

    def fetch_results(self, wid):
        return [DummyResult(r) for r in self.data.get(wid, [])]


class DummyStability:
    def __init__(self, data):
        self.data = data

    def is_stable(self, wid, current_roi=None, threshold=None):
        return wid in self.data


def _load_fallback_planner():
    src = Path("self_improvement/meta_planning.py").read_text()
    tree = ast.parse(src)
    nodes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_FallbackPlanner"
    ]
    module = ast.Module(nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)
    ns: dict[str, Any] = {
        "Any": Any,
        "Callable": Callable,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "fmean": fmean,
        "ROIResultsDB": DummyROI,
        "WorkflowStabilityDB": DummyStability,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    return ns["_FallbackPlanner"]


def test_fallback_planner_uses_roi_and_stability():
    Fallback = _load_fallback_planner()
    planner = Fallback()
    planner.roi_db = DummyROI({"a": [0.2, 0.3], "b": [-0.1]})
    planner.stability_db = DummyStability({"a": {"failures": 1, "entropy": 0.1}})

    records = planner.discover_and_persist({"a": lambda: None, "b": lambda: None})
    assert records == [
        {
            "chain": ["a"],
            "roi_gain": pytest.approx(0.25),
            "failures": 1,
            "entropy": 0.1,
        }
    ]

