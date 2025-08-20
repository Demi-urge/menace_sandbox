from types import SimpleNamespace

import pytest

from vector_service.retriever import Retriever
from vector_service.context_builder import ContextBuilder


class DummyRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=5, **kwargs):  # pragma: no cover - simple stub
        return list(self._hits)


def test_high_severity_vector_ranks_lower():
    hits = [
        {
            "origin_db": "information",
            "record_id": "safe",
            "score": 1.0,
            "text": "safe text",
        },
        {
            "origin_db": "information",
            "record_id": "bad",
            "score": 1.0,
            "alignment_severity": 0.9,
            "text": "bad",
        },
    ]
    cb = ContextBuilder(retriever=DummyRetriever(hits))
    _, _, vectors = cb.build_context("q", include_vectors=True)
    # Safe record should rank before high severity one
    assert vectors[0][1] == "safe"
    assert vectors[0][2] > vectors[1][2]


def test_disallowed_license_ranks_lower():
    hits = [
        {
            "origin_db": "information",
            "record_id": "ok",
            "score": 1.0,
            "license": "MIT",
            "text": "ok",
        },
        {
            "origin_db": "information",
            "record_id": "gpl",
            "score": 1.0,
            "license": "GPL-3.0",
            "text": "gpl",
        },
    ]
    cb = ContextBuilder(retriever=DummyRetriever(hits))
    _, _, vectors = cb.build_context("q", include_vectors=True)
    assert vectors[0][1] == "ok"
    assert vectors[0][2] > vectors[1][2]


def test_retriever_excludes_above_threshold(monkeypatch):
    from vector_service import retriever as rmod

    def fake_govern(text, meta, reason=None, max_alert_severity=1.0):
        if meta.get("alignment_severity", 0.0) > max_alert_severity:
            return None
        return meta, reason

    monkeypatch.setattr(rmod, "govern_retrieval", fake_govern)
    monkeypatch.setattr(rmod, "pii_redact_dict", lambda x: x)
    monkeypatch.setattr(rmod, "redact_dict", lambda x: x)

    hit = SimpleNamespace(
        metadata={"redacted": True, "alignment_severity": 0.9},
        score=1.0,
        record_id=1,
        text="hit",
    )
    ret = Retriever()
    assert ret._parse_hits([hit], max_alert_severity=0.5) == []
