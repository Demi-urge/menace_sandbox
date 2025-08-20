import types
import time

from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer
from vector_metrics_db import VectorMetricsDB


class DummyRetriever:
    def search(self, query, top_k=5, session_id=""):
        time.sleep(0.001)
        return [
            {
                "origin_db": "bot",
                "record_id": 1,
                "score": 0.9,
                "metadata": {"name": "alpha", "timestamp": time.time() - 30.0},
            }
        ]


def test_retrieval_metrics_persist(tmp_path):
    db = VectorMetricsDB(tmp_path / "vec.db")
    builder = ContextBuilder(retriever=DummyRetriever())
    layer = CognitionLayer(
        context_builder=builder,
        vector_metrics=db,
        patch_logger=types.SimpleNamespace(
            track_contributors=lambda *a, **k: None,
            track_contributors_async=lambda *a, **k: None,
        ),
    )

    layer.query("hello world")

    rows = db.conn.execute(
        "SELECT tokens, wall_time_ms, prompt_tokens, age FROM vector_metrics WHERE event_type='retrieval'"
    ).fetchall()
    assert rows
    for tokens, wall_ms, prompt_tokens, age in rows:
        assert tokens > 0
        assert wall_ms > 0.0
        assert prompt_tokens > 0
        assert age >= 0


def test_retriever_filters_unsafe_vectors(monkeypatch):
    from vector_service import retriever as rmod
    from types import SimpleNamespace

    license_fp = next(iter(rmod._LICENSE_DENYLIST))

    monkeypatch.setattr(
        rmod,
        "govern_retrieval",
        lambda text, meta, reason=None, max_alert_severity=1.0: (meta, reason),
    )
    monkeypatch.setattr(rmod, "pii_redact_dict", lambda x: x)
    monkeypatch.setattr(rmod, "redact_dict", lambda x: x)

    calls: list[int] = []

    class DummyGauge:
        def inc(self, value):
            calls.append(value)

    monkeypatch.setattr(rmod, "_FILTERED_RESULTS", DummyGauge())

    hits = [
        SimpleNamespace(
            metadata={"redacted": True, "license_fingerprint": license_fp},
            score=1.0,
            record_id="lic",
            text="",
        ),
        SimpleNamespace(
            metadata={"redacted": True, "alignment_severity": 0.9},
            score=1.0,
            record_id="sev",
            text="",
        ),
        SimpleNamespace(metadata={"redacted": True}, score=1.0, record_id="ok", text=""),
    ]

    ret = rmod.Retriever()
    res = ret._parse_hits(hits, max_alert_severity=0.5)
    assert [r["record_id"] for r in res] == ["ok"]
    assert calls == [2]


def test_bundle_to_entry_surfaces_flags(monkeypatch):
    from vector_service import context_builder as cmod

    monkeypatch.setattr(cmod, "_VEC_METRICS", None)

    cb = ContextBuilder()
    bundle = {
        "origin_db": "information",
        "record_id": "1",
        "text": "sample",
        "license": "GPL-3.0",
        "license_fingerprint": "fp",
        "semantic_alerts": [("x", "y", 0.9)],
        "alignment_severity": 0.9,
        "metadata": {"title": "t"},
    }
    _, scored = cb._bundle_to_entry(bundle, "q")
    flags = scored.entry.get("flags")
    assert flags["license"] == "GPL-3.0"
    assert flags["license_fingerprint"] == "fp"
    assert flags["semantic_alerts"] == [("x", "y", 0.9)]
    assert flags["alignment_severity"] == 0.9
