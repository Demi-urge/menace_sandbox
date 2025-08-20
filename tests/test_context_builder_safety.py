import json

from vector_service.context_builder import ContextBuilder
from vector_metrics_db import VectorMetricsDB, VectorMetric


class DummyRetriever:
    def search(self, query, top_k=5, session_id="", **kwargs):
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
    def search(self, query, top_k=5, session_id="", **kwargs):
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
    def search(self, query, top_k=5, session_id="", **kwargs):
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


class LicenseRetriever:
    def search(self, query, top_k=5, session_id="", **kwargs):
        return [
            {
                "origin_db": "bot",
                "record_id": "bad",
                "score": 0.5,
                "metadata": {"name": "bad", "license": "GPL-3.0"},
            },
            {
                "origin_db": "bot",
                "record_id": "ok",
                "score": 0.5,
                "metadata": {"name": "ok"},
            },
        ]


def test_license_denylist_filter(monkeypatch):
    import vector_service.context_builder as cb

    class Gauge:
        def __init__(self):
            self.calls: list[tuple[str, object]] = []

        def labels(self, risk):
            self.calls.append(("labels", risk))
            return self

        def inc(self, amount=1.0):
            self.calls.append(("inc", amount))

    gauge = Gauge()
    monkeypatch.setattr(cb, "_VECTOR_RISK", gauge)
    builder = ContextBuilder(retriever=LicenseRetriever(), license_denylist={"GPL-3.0"})
    ctx = builder.build_context("hi", top_k=2)
    data = json.loads(ctx)
    bots = data["bots"]
    assert [b["id"] for b in bots] == ["ok"]
    assert ("labels", "filtered") in gauge.calls
