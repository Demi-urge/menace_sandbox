from types import SimpleNamespace

import pytest

from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=5, session_id="", **kwargs):
        return list(self.hits)


class DummyBus:
    def __init__(self):
        self.published = []

    def publish(self, topic, event):
        self.published.append((topic, event))


def test_risk_scores_reduce_weights_and_emit_events(tmp_path, monkeypatch):
    monkeypatch.setattr("vector_service.cognition_layer.ROITracker", None)
    hits = [
        {
            "origin_db": "bot",
            "record_id": "b1",
            "score": 0.5,
            "text": "bot",
            "metadata": {"name": "b", "risk_score": 0.0},
        },
        {
            "origin_db": "workflow",
            "record_id": "w1",
            "score": 0.5,
            "text": "wf",
            "metadata": {"title": "w", "risk_score": 2.0},
        },
    ]
    retriever = DummyRetriever(hits)
    vec_db = VectorMetricsDB(tmp_path / "vec.db")
    bus = DummyBus()

    class DummyBuilder:
        def __init__(self, retriever):
            self.retriever = retriever

        def build_context(self, query, top_k=5, session_id="", **kwargs):
            hits = self.retriever.search(query, top_k=top_k, session_id=session_id)
            vectors = [(h["origin_db"], h["record_id"], h["score"]) for h in hits]
            meta = {f"{h['origin_db']}:{h['record_id']}": h for h in hits}
            stats = {"tokens": 0, "wall_time_ms": 0.0, "prompt_tokens": 0}
            return "", session_id or "sid", vectors, meta, stats

        def refresh_db_weights(self, *args, **kwargs):
            pass

    builder = DummyBuilder(retriever)
    patch_logger = SimpleNamespace(
        track_contributors=lambda *a, **k: {},
        roi_tracker=None,
        event_bus=bus,
    )
    layer = CognitionLayer(
        retriever=retriever,
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=vec_db,
        event_bus=bus,
        roi_tracker=None,
    )

    _ctx, sid = layer.query("q", top_k=2)
    layer.record_patch_outcome(sid, True, contribution=1.0)

    weights = vec_db.get_db_weights()
    assert weights["workflow"] < weights["bot"]

    feedback = {
        evt[1]["db"]: evt[1]
        for evt in bus.published
        if evt[0] == "retrieval:feedback"
    }
    assert feedback["workflow"]["risk"] == pytest.approx(2.0)
    assert feedback["bot"]["risk"] == pytest.approx(0.0)
