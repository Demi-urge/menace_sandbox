import sys
import types


class _StubTracker:
    def update_db_metrics(self, metrics):
        pass


sys.modules.setdefault("roi_tracker", types.SimpleNamespace(ROITracker=_StubTracker))

from vector_service.cognition_layer import CognitionLayer


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


class DummyVectorMetrics:
    def log_retrieval(self, *a, **k):
        pass

    def update_outcome(self, *a, **k):
        pass

    def log_retrieval_feedback(self, *a, **k):
        pass

    def record_patch_ancestry(self, *a, **k):
        pass


class DummyTracker:
    def __init__(self):
        self.metrics = None

    def update_db_metrics(self, metrics):
        self.metrics = metrics


def test_patch_outcome_updates_roi_tracker():
    tracker = DummyTracker()
    layer = CognitionLayer(
        context_builder=DummyContextBuilder(),
        vector_metrics=DummyVectorMetrics(),
        roi_tracker=tracker,
    )

    _, sid = layer.query("hello")
    layer.record_patch_outcome(sid, True, contribution=1.0)

    assert tracker.metrics == {
        "db1": {"roi": 2.0, "win_rate": 1.0, "regret_rate": 0.0},
        "db2": {"roi": 1.0, "win_rate": 1.0, "regret_rate": 0.0},
    }
