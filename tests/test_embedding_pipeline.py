import argparse
import json
from pathlib import Path

import pytest

from menace_cli import handle_embed
from vector_service.context_builder import ContextBuilder
from vector_service.embedding_backfill import EmbeddingBackfill


class MockVectorService:
    def __init__(self):
        self.store = {}

    def vectorise(self, kind, record):
        text = record.get("text", "")
        return [float(len(text))]

    def vectorise_and_store(self, kind, record_id, record, origin_db=None, metadata=None):
        vec = self.vectorise(kind, record)
        self.store[str(record_id)] = (vec, origin_db, record.get("text", ""))
        return vec

    def search(self, query_vec, top_k=5):
        results = []
        for rid, (vec, origin, text) in self.store.items():
            dist = abs(vec[0] - query_vec[0])
            results.append((rid, dist, origin, text))
        results.sort(key=lambda x: x[1])
        return results[:top_k]


@pytest.fixture()
def vector_service(monkeypatch):
    svc = MockVectorService()

    # patch SharedVectorService methods used by databases
    monkeypatch.setattr(
        "vector_service.vectorizer.SharedVectorService.vectorise_and_store",
        lambda self, kind, record_id, record, origin_db=None, metadata=None: svc.vectorise_and_store(kind, record_id, record, origin_db, metadata),
    )
    monkeypatch.setattr(
        "vector_service.vectorizer.SharedVectorService.vectorise",
        lambda self, kind, record: svc.vectorise(kind, record),
    )
    return {"svc": svc, "records": {}}


# Only include databases with buckets recognised by ContextBuilder
DB_NAMES = [
    "bot",
    "workflow",
    "enhancement",
    "error",
    "information",
    "code",
    "discrepancy",
]


@pytest.mark.parametrize("db_name", DB_NAMES)
def test_embedding_pipeline(vector_service, db_name, monkeypatch):
    svc = vector_service["svc"]
    record_map = vector_service["records"]

    class SyntheticDB:
        records = {f"{db_name}_1": f"sample {db_name} text"}

        def __init__(self, vector_backend="annoy"):
            self.vector_backend = vector_backend

        def iter_records(self):
            for rid, text in self.records.items():
                yield rid, text, "text"

        def needs_refresh(self, record_id, record):
            return True

        def add_embedding(self, record_id, record, kind):
            svc.vectorise_and_store("text", record_id, {"text": record}, origin_db=db_name, metadata={"redacted": True})
            record_map[str(record_id)] = {"origin_db": db_name, "text": record}

    monkeypatch.setattr(
        EmbeddingBackfill, "_load_known_dbs", lambda self, names=None: [SyntheticDB]
    )

    args = argparse.Namespace(
        dbs=[db_name],
        batch_size=None,
        all=False,
        backend="annoy",
        log_file=None,
        verify=False,
        verify_only=False,
    )
    assert handle_embed(args) == 0

    class SimpleRetriever:
        max_alert_severity = 1.0
        max_alerts = 5
        license_denylist: set[str] = set()

        def search(self, query, top_k=5, session_id="", **_):
            vec = svc.vectorise("text", {"text": query})
            results = svc.search(vec, top_k)
            hits = []
            for rid, dist, origin, text in results:
                hits.append(
                    {
                        "origin_db": origin,
                        "record_id": rid,
                        "score": 1 - dist,
                        "text": text,
                        "metadata": {"redacted": True},
                    }
                )
            return hits

    builder = ContextBuilder(retriever=SimpleRetriever())
    context = builder.build(f"sample {db_name} text", top_k=1)
    assert context and context != "{}"
