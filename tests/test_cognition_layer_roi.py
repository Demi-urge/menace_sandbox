import sys
import types

import pytest


class _StubTracker:
    def update_db_metrics(self, metrics):
        pass

    def update(self, *a, **k):
        pass


sys.modules.setdefault("roi_tracker", types.SimpleNamespace(ROITracker=_StubTracker))

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyContextBuilder:
    def build_context(self, prompt, *, top_k=5, include_vectors=False, session_id="", return_stats=False):
        vectors = [
            ("db1", "v1", 0.5),
            ("db1", "v2", 0.3),
            ("db2", "v3", 0.2),
        ]
        stats = {"tokens": 1, "wall_time_ms": 1.0, "prompt_tokens": len(prompt.split())}
        if include_vectors:
            if return_stats:
                return "context", session_id or "sid", vectors, stats
            return "context", session_id or "sid", vectors
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
        "db1": {"roi": 2.0, "win_rate": 1.0, "regret_rate": 0.0},
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
        "db1": {"roi": 2.0, "win_rate": 0.0, "regret_rate": 1.0},
        "db2": {"roi": 1.0, "win_rate": 0.0, "regret_rate": 1.0},
    }
    assert tracker.update_args["roi_after"] == pytest.approx(3.0)
    assert len(tracker.update_args["retrieval_metrics"]) == 3
