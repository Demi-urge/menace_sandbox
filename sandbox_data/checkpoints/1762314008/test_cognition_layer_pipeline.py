import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import menace_sandbox.roi_tracker as roi_tracker_module
sys.modules["roi_tracker"] = roi_tracker_module

import menace_sandbox.cognition_layer as cl_module
from menace_sandbox.roi_tracker import ROITracker
from menace_sandbox.vector_metrics_db import VectorMetricsDB
from menace_sandbox.vector_service.cognition_layer import CognitionLayer
from menace_sandbox.vector_service.patch_logger import PatchLogger
from menace_sandbox.patch_safety import PatchSafety


class DummyRetriever:
    def search(self, query, top_k=5, session_id=""):
        return [("db1", "vec1", 1.0)]


class DummyContextBuilder:
    def __init__(self, retriever):
        self.retriever = retriever

    def build_context(
        self,
        prompt,
        *,
        top_k=5,
        include_vectors=False,
        session_id="",
        return_stats=False,
        return_metadata=False,
    ):
        vectors = self.retriever.search(prompt, top_k=top_k, session_id=session_id)
        sid = session_id or "sid"
        meta = {"data": []}
        for origin, vec_id, _ in vectors:
            meta["data"].append(
                {"origin_db": origin, "vector_id": vec_id, "metadata": {"timestamp": 0}}
            )
        stats = {"tokens": 1, "wall_time_ms": 0.0, "prompt_tokens": 1}
        if include_vectors:
            if return_metadata:
                if return_stats:
                    return "ctx", sid, vectors, meta, stats
                return "ctx", sid, vectors, meta
            if return_stats:
                return "ctx", sid, vectors, stats
            return "ctx", sid, vectors
        if return_metadata:
            if return_stats:
                return "ctx", meta, stats
            return "ctx", meta
        if return_stats:
            return "ctx", stats
        return "ctx"


def test_cognition_layer_pipeline_feedback_updates_weights_and_roi():
    retriever = DummyRetriever()
    builder = DummyContextBuilder(retriever)
    metrics = VectorMetricsDB(":memory:")
    tracker = ROITracker()
    tracker.update = lambda *a, **k: (None, [], False, False)  # type: ignore
    ps = PatchSafety(storage_path=None, failure_db_path=None)
    logger = PatchLogger(vector_metrics=metrics, roi_tracker=tracker, patch_safety=ps)
    layer = CognitionLayer(
        context_builder=builder,
        patch_logger=logger,
        vector_metrics=metrics,
        roi_tracker=tracker,
    )

    setattr(builder, "_cognition_layer", layer)
    cl_module._roi_tracker = tracker

    _ctx, sid = cl_module.build_cognitive_context(
        "example query", top_k=1, context_builder=builder
    )
    cl_module.log_feedback(sid, True, context_builder=builder)

    weights = metrics.get_db_weights()
    assert weights.get("db1", 0.0) > 0.0

    roi_deltas = tracker.origin_db_deltas()
    assert roi_deltas.get("db1", 0.0) > 0.0
