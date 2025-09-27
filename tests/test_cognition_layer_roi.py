import sys
import types
from typing import Dict, List

import pytest
import time


class _StubTracker:
    def update_db_metrics(self, metrics):
        pass

    def update(self, *a, **k):
        pass


sys.modules.setdefault("roi_tracker", types.SimpleNamespace(ROITracker=_StubTracker))

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyContextBuilder:
    roi_tag_penalties: Dict[str, float] = {}

    def build_context(
        self,
        prompt,
        *,
        top_k=5,
        include_vectors=False,
        session_id="",
        return_stats=False,
        return_metadata=False,
        stack_preferences=None,
    ):
        vectors = [
            ("db1", "v1", 0.5),
            ("db1", "v2", 0.3),
            ("db2", "v3", 0.2),
        ]
        stats = {"tokens": 1, "wall_time_ms": 1.0, "prompt_tokens": len(prompt.split())}
        meta = {
            "misc": [
                {
                    "origin_db": o,
                    "vector_id": v,
                    "metadata": {"timestamp": time.time() - 30.0},
                }
                for o, v, _ in vectors
            ]
        }
        sid = session_id or "sid"
        if include_vectors:
            if return_metadata:
                if return_stats:
                    return "context", sid, vectors, meta, stats
                return "context", sid, vectors, meta
            if return_stats:
                return "context", sid, vectors, stats
            return "context", sid, vectors
        if return_metadata:
            if return_stats:
                return "context", meta, stats
            return "context", meta
        if return_stats:
            return "context", stats
        return "context"


class DummyTracker:
    def __init__(self):
        self.metrics = None
        self.update_args = None

    def update_db_metrics(self, metrics):
        self.metrics = metrics

    def update(self, roi_before, roi_after, *, retrieval_metrics=None, **_):
        self.update_args = {
            "roi_before": roi_before,
            "roi_after": roi_after,
            "retrieval_metrics": retrieval_metrics,
        }


def _make_layer(tracker):
    return CognitionLayer(
        context_builder=DummyContextBuilder(),
        vector_metrics=VectorMetricsDB(":memory:"),
        roi_tracker=tracker,
    )


def test_patch_outcome_updates_roi_tracker_success():
    tracker = DummyTracker()
    layer = _make_layer(tracker)

    _, sid = layer.query("hello")
    layer.record_patch_outcome(sid, True, contribution=1.0)

    assert tracker.metrics == {
        "db1": {"roi": 1.0, "win_rate": 1.0, "regret_rate": 0.0},
        "db2": {"roi": 1.0, "win_rate": 1.0, "regret_rate": 0.0},
    }
    assert tracker.update_args["roi_after"] == pytest.approx(3.0)
    assert len(tracker.update_args["retrieval_metrics"]) == 3


def test_patch_outcome_updates_roi_tracker_failure():
    tracker = DummyTracker()
    layer = _make_layer(tracker)

    _, sid = layer.query("hello")
    layer.record_patch_outcome(sid, False, contribution=1.0)

    assert tracker.metrics == {
        "db1": {"roi": 1.0, "win_rate": 0.0, "regret_rate": 1.0},
        "db2": {"roi": 1.0, "win_rate": 0.0, "regret_rate": 1.0},
    }
    assert tracker.update_args["roi_after"] == pytest.approx(3.0)
    assert len(tracker.update_args["retrieval_metrics"]) == 3


def test_failure_triggers_backfill_and_reliability(monkeypatch):
    tracker = DummyTracker()
    layer = _make_layer(tracker)
    _, sid = layer.query("hello")

    calls: Dict[str, object] = {}

    async def fake_schedule_backfill(*, dbs=None, **_):
        calls["backfill"] = dbs

    monkeypatch.setattr(
        "vector_service.cognition_layer.schedule_backfill", fake_schedule_backfill
    )

    def fake_reload() -> None:
        calls["reload"] = True

    monkeypatch.setattr(layer, "reload_reliability_scores", fake_reload)

    layer.record_patch_outcome(sid, False, contribution=1.0)

    assert set(calls.get("backfill") or []) == {"db1", "db2"}
    assert calls.get("reload") is True


class DropTracker(DummyTracker):
    origin_db_delta_history: Dict[str, List[float]] = {}

    def origin_db_deltas(self):
        return {
            db: vals[-1]
            for db, vals in self.origin_db_delta_history.items()
            if vals
        }

    def update(self, *a, **k):
        super().update(*a, **k)
        self.origin_db_delta_history = {"db1": [-0.5], "db2": [0.1]}


def test_roi_drop_triggers_backfill_and_reliability(monkeypatch):
    tracker = DropTracker()
    layer = _make_layer(tracker)
    _, sid = layer.query("hello")

    calls: Dict[str, object] = {}

    async def fake_schedule_backfill(*, dbs=None, **_):
        calls["backfill"] = dbs

    monkeypatch.setattr(
        "vector_service.cognition_layer.schedule_backfill", fake_schedule_backfill
    )

    def fake_reload() -> None:
        calls["reload"] = True

    monkeypatch.setattr(layer, "reload_reliability_scores", fake_reload)

    layer.record_patch_outcome(sid, True, contribution=1.0)

    assert set(calls.get("backfill") or []) == {"db1", "db2"}
    assert calls.get("reload") is True


def test_roi_stats_exposes_metrics():
    tracker = DummyTracker()
    layer = _make_layer(tracker)
    vm = layer.vector_metrics
    vm.log_retrieval("bots", tokens=0, wall_time_ms=0.0, hit=True, rank=1, session_id="b", vector_id="v")
    vm.update_outcome("b", [("bots", "v")], contribution=0.4, win=True)

    stats = layer.roi_stats()
    assert stats["bots"]["bots"]["roi_delta"] == 0.4
