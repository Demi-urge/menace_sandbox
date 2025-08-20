import json

from vector_service.context_builder import ContextBuilder
from vector_metrics_db import VectorMetricsDB, VectorMetric


class DummyRetriever:
    def search(self, query, top_k=5, session_id=""):
        return [
            {
                "origin_db": "bot",
                "record_id": "risky",
                "score": 0.5,
                "metadata": {"name": "risky"},
            },
            {
                "origin_db": "bot",
                "record_id": "safe",
                "score": 0.5,
                "metadata": {"name": "safe"},
            },
        ]


def test_risky_vectors_rank_lower(monkeypatch, tmp_path):
    db = VectorMetricsDB(tmp_path / "vec.db")
    db.add(
        VectorMetric(
            event_type="retrieval",
            db="bot",
            tokens=0,
            wall_time_ms=0.0,
            hit=True,
            rank=0,
            win=False,
            regret=True,
            vector_id="risky",
        )
    )
    db.add(
        VectorMetric(
            event_type="retrieval",
            db="bot",
            tokens=0,
            wall_time_ms=0.0,
            hit=True,
            rank=0,
            win=True,
            regret=False,
            vector_id="safe",
        )
    )
    db.record_patch_ancestry("p", [("risky", 1.0, None, None, 0.9), ("safe", 1.0, None, None, 0.1)])

    import vector_service.context_builder as cb

    monkeypatch.setattr(cb, "_VEC_METRICS", db)

    builder = ContextBuilder(retriever=DummyRetriever())
    ctx = builder.build_context("hello", top_k=2)
    data = json.loads(ctx)
    bots = data["bots"]
    assert [b["id"] for b in bots] == ["safe", "risky"]


class SevRetriever:
    def search(self, query, top_k=5, session_id=""):
        return [
            {
                "origin_db": "bot",
                "record_id": "danger",
                "score": 0.5,
                "metadata": {"name": "danger", "alignment_severity": 0.9},
            },
            {
                "origin_db": "bot",
                "record_id": "ok",
                "score": 0.5,
                "metadata": {"name": "ok", "alignment_severity": 0.1},
            },
        ]


def test_alignment_severity_filter():
    builder = ContextBuilder(retriever=SevRetriever(), max_alignment_severity=0.5)
    ctx = builder.build_context("hi", top_k=2)
    data = json.loads(ctx)
    bots = data["bots"]
    assert [b["id"] for b in bots] == ["ok"]


class AlertRetriever:
    def search(self, query, top_k=5, session_id=""):
        return [
            {
                "origin_db": "bot",
                "record_id": "alert",
                "score": 0.5,
                "metadata": {"name": "alert", "semantic_alerts": ["a", "b"]},
            },
            {
                "origin_db": "bot",
                "record_id": "safe",
                "score": 0.5,
                "metadata": {"name": "safe"},
            },
        ]


def test_alert_count_filter():
    builder = ContextBuilder(retriever=AlertRetriever(), max_alerts=0)
    ctx = builder.build_context("hi", top_k=2)
    data = json.loads(ctx)
    bots = data["bots"]
    assert [b["id"] for b in bots] == ["safe"]
