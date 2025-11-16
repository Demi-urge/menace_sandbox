import json
import logging

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


class DummyPatchSafety:
    def __init__(self) -> None:
        self.max_alert_severity = 1.0
        self.max_alerts = 5
        self.license_denylist = set()

    def evaluate(self, *_, **__):
        return True, 0.0, {}

    def load_failures(self, *_, **__):
        return None


class NoRiskRetriever:
    def search(self, query, top_k=5, session_id="", **kwargs):
        return [
            {
                "origin_db": "bot",
                "record_id": "norisk",
                "score": 0.5,
                "metadata": {"name": "norisk"},
            }
        ]


def test_missing_risk_logs_default(caplog):
    builder = ContextBuilder(
        retriever=NoRiskRetriever(),
        patch_safety=DummyPatchSafety(),
        risk_penalty=1.0,
    )
    with caplog.at_level(logging.WARNING):
        ctx = builder.build_context("hi", top_k=1)
    data = json.loads(ctx)
    bots = data["bots"]
    assert bots[0]["risk_score"] == 0.0
    assert bots[0]["risk_score_defaulted"] is True
    assert any("risk_score missing" in rec.message for rec in caplog.records)


class RiskRetriever:
    def search(self, query, top_k=5, session_id="", **kwargs):
        return [
            {
                "origin_db": "bot",
                "record_id": "low",
                "score": 0.5,
                "metadata": {"name": "low", "risk_score": 0.1},
            },
            {
                "origin_db": "bot",
                "record_id": "high",
                "score": 0.5,
                "metadata": {"name": "high", "risk_score": 0.9},
            },
        ]


def test_risk_score_affects_ranking():
    builder = ContextBuilder(
        retriever=RiskRetriever(),
        patch_safety=DummyPatchSafety(),
        risk_penalty=1.0,
        ranking_weight=1.0,
        roi_weight=1.0,
        safety_weight=1.0,
    )
    ctx = builder.build_context("hi", top_k=2)
    data = json.loads(ctx)
    bots = data["bots"]
    assert [b["id"] for b in bots] == ["low", "high"]
    assert bots[0]["risk_score"] == 0.1
    assert bots[1]["risk_score"] == 0.9
