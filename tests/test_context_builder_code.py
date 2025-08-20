import json
import sys
import types

# Minimal stubs for heavy dependencies required by CodeDB
sys.modules.setdefault(
    "auto_link", types.SimpleNamespace(auto_link=lambda *_a, **_k: (lambda f: f))
)
sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "retry_utils",
    types.SimpleNamespace(
        publish_with_retry=lambda *a, **k: True,
        with_retry=lambda func, **k: func(),
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={}),
)

from vector_service.context_builder import ContextBuilder
from vector_service import EmbeddableDBMixin
from code_database import CodeDB, CodeRecord


class _CodeRetriever:
    def __init__(self, db: CodeDB):
        self.db = db

    def search(self, query, top_k=5, **_):  # pragma: no cover - simple stub
        vec = self.db.encode_text(query)
        rows = self.db.search_by_vector(vec, top_k)
        hits = []
        for row in rows:
            hits.append(
                {
                    "origin_db": "code",
                    "record_id": row["id"],
                    "score": 1.0,
                    "text": row["code"],
                    "metadata": {"redacted": True, "summary": row["summary"]},
                }
            )
        return hits


def test_context_builder_returns_code_record(tmp_path, monkeypatch):
    monkeypatch.setattr(
        EmbeddableDBMixin,
        "encode_text",
        lambda self, text: [float(len(text))],
    )

    db = CodeDB(path=tmp_path / "code.db")
    cid = db.add(CodeRecord(code="print('hi')", summary="greet"))

    builder = ContextBuilder(retriever=_CodeRetriever(db))
    context = builder.build_context("hi")
    data = json.loads(context)
    assert data["code"][0]["id"] == cid
    assert data["code"][0]["desc"]

