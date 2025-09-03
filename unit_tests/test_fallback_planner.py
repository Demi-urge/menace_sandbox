"""Tests for the fallback planner implementation."""

from __future__ import annotations

import ast
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence
from dynamic_path_router import resolve_path

import pytest


class DummyResult:
    def __init__(self, roi_gain: float) -> None:
        self.roi_gain = roi_gain


class DummyROI:
    def __init__(self, data: Mapping[str, Sequence[float]]):
        self.data = data
        self.logged: list[dict[str, Any]] = []

    def fetch_results(self, wid: str):
        return [DummyResult(r) for r in self.data.get(wid, [])]

    def log_result(self, **kw: Any) -> None:
        self.logged.append(kw)


class DummyStability:
    def __init__(self, data: Mapping[str, Mapping[str, Any]]):
        self.data = data
        self.logged: list[tuple[str, float, int, float, float | None]] = []

    def is_stable(
        self,
        wid: str,
        current_roi: float | None = None,
        threshold: float | None = None,
    ) -> bool:  # noqa: D401
        return wid in self.data

    def record_metrics(
        self,
        wid: str,
        roi: float,
        failures: int,
        entropy: float,
        roi_delta: float | None = None,
    ) -> None:  # noqa: D401
        self.logged.append((wid, roi, failures, entropy, roi_delta))


class DummyLogger:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.warnings: list[str] = []
        self.exceptions: list[str] = []

    def debug(self, msg: str, *args: Any, **_: Any) -> None:
        self.debugs.append(msg % args if args else msg)

    def warning(self, msg: str, *args: Any, **_: Any) -> None:
        self.warnings.append(msg % args if args else msg)

    def exception(self, msg: str, *args: Any, **_: Any) -> None:
        self.exceptions.append(msg % args if args else msg)


def _load_fallback_planner():
    src = resolve_path("self_improvement/meta_planning.py").read_text()
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_FallbackPlanner"]
    module = ast.Module(nodes, type_ignores=[])
    module = ast.fix_missing_locations(module)
    logger = DummyLogger()
    from contextlib import contextmanager, nullcontext
    import json
    import os
    import time

    class DummyLock:
        @contextmanager
        def acquire(self, timeout: float | None = None):
            yield

    ns: dict[str, Any] = {
        "Any": Any,
        "Callable": Callable,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "fmean": fmean,
        "ROIResultsDB": DummyROI,
        "WorkflowStabilityDB": DummyStability,
        "get_logger": lambda name: logger,
        "log_record": lambda **kw: kw,
        "contextmanager": contextmanager,
        "nullcontext": nullcontext,
        "SandboxLock": DummyLock,
        "LOCK_TIMEOUT": 1,
        "Timeout": Exception,
        "json": json,
        "os": os,
        "time": time,
    }
    exec(compile(module, "<ast>", "exec"), ns)
    Fallback = ns["_FallbackPlanner"]

    def simple_init(self):
        self.logger = logger
        self.state_path = Path("state.json")
        self.state_lock = DummyLock()
        self.cluster_map = {}
        self.mutation_rate = 1.0
        self.roi_weight = 1.0
        self.domain_transition_penalty = 1.0
        self.entropy_weight = 0.0
        self.stability_weight = 1.0
        self.state_prune_strategy = "recent"
        self.roi_window = 5
        self.state_capacity = 1000
        self.roi_db = None
        self.stability_db = None

    Fallback.__init__ = simple_init  # type: ignore
    return Fallback, logger


def test_mutate_pipeline_scores_with_weights_and_penalty():
    Fallback, logger = _load_fallback_planner()
    planner = Fallback()
    planner.roi_db = DummyROI(
        {
            "domA.1": [0.4],
            "domA.2": [0.5],
            "domB.1": [0.6],
        }
    )
    planner.stability_db = DummyStability(
        {
            "domA.1": {"failures": 0, "entropy": 0.1},
            "domA.2": {"failures": 1, "entropy": 0.1},
            "domB.1": {"failures": 0, "entropy": 0.1},
        }
    )
    planner.mutation_rate = 2
    planner.roi_weight = 2.0
    planner.domain_transition_penalty = 1.0
    planner.stability_weight = 0.5

    workflows = {wid: (lambda wid=wid: None) for wid in planner.roi_db.data}
    results = planner.mutate_pipeline(["domA.1"], workflows)

    assert results[0]["chain"] == ["domA.1", "domA.2"]
    assert results[0]["score"] == pytest.approx(0.4)
    assert any(len(r["chain"]) == 3 for r in results)
    assert planner.roi_db.logged  # integration logging
    assert planner.stability_db.logged


def test_discover_and_persist_handles_db_failures():
    Fallback, logger = _load_fallback_planner()
    planner = Fallback()

    class BadROI(DummyROI):
        def fetch_results(self, wid: str):
            raise RuntimeError("boom")

    planner.roi_db = BadROI({})
    planner.stability_db = DummyStability({"a": {"failures": 0, "entropy": 0.0}})
    records = planner.discover_and_persist({"a": lambda: None})
    assert records == []
    assert logger.warnings  # failure path logged


def test_evaluate_chain_rejects_duplicate_chain():
    Fallback, _ = _load_fallback_planner()
    planner = Fallback()
    planner.roi_db = DummyROI({"a": [0.1]})
    planner.stability_db = DummyStability({"a": {"failures": 0, "entropy": 0.1}})
    assert planner._evaluate_chain(["a", "a"]) is None
    assert not planner.cluster_map


def test_discover_and_persist_ranks_and_prunes():
    Fallback, _ = _load_fallback_planner()
    planner = Fallback()
    planner.roi_db = DummyROI(
        {"w1": [0.2], "w2": [0.9], "w3": [0.1]}
    )
    planner.stability_db = DummyStability(
        {
            "w1": {"failures": 0, "entropy": 0.1},
            "w2": {"failures": 0, "entropy": 0.1},
            "w3": {"failures": 0, "entropy": 0.1},
        }
    )
    planner.state_capacity = 2
    workflows = {wid: (lambda wid=wid: None) for wid in planner.roi_db.data}
    results = planner.discover_and_persist(workflows)
    assert results[0]["chain"] == ["w2"]
    assert len(planner.cluster_map) == 2


def test_state_persistence_roundtrip_and_pruning(tmp_path):
    Fallback, _ = _load_fallback_planner()
    planner = Fallback()
    planner.state_path = tmp_path / "state.json"
    planner.state_prune_strategy = "score"
    planner.state_capacity = 2
    planner.cluster_map = {
        ("a",): {"score": 0.1, "ts": 1.0},
        ("b",): {"score": 0.9, "ts": 2.0},
        ("c",): {"score": 0.5, "ts": 3.0},
    }
    planner._save_state()

    planner2 = Fallback()
    planner2.state_path = planner.state_path
    planner2.state_prune_strategy = "score"
    planner2.state_capacity = 2
    planner2._load_state()
    assert planner2.cluster_map == planner.cluster_map
    planner2._prune_state()
    assert set(planner2.cluster_map.keys()) == {("b",), ("c",)}
