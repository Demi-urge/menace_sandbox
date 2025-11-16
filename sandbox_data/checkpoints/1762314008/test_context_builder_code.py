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


class _CodeRecord:
    def __init__(self, code: str = "", summary: str = ""):
        self.code = code
        self.summary = summary


class _CodeDB:
    def __init__(self, path=None):
        self._rows = []

    def add(self, record: _CodeRecord) -> int:  # pragma: no cover - simple stub
        self._rows.append({"id": 1, "code": record.code, "summary": record.summary})
        return 1

    def encode_text(self, text: str):  # pragma: no cover - simple stub
        return [float(len(text))]

    def search_by_vector(self, _vec, _k):  # pragma: no cover - simple stub
        return self._rows


sys.modules.setdefault(
    "code_database", types.SimpleNamespace(CodeDB=_CodeDB, CodeRecord=_CodeRecord)
)

from vector_service.context_builder import ContextBuilder  # noqa: E402
import vector_service.context_builder as cb  # noqa: E402
from vector_service import EmbeddableDBMixin  # noqa: E402
from code_database import CodeDB, CodeRecord  # noqa: E402


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
    if hasattr(EmbeddableDBMixin, "encode_text"):
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


def test_context_builder_includes_patch_details(monkeypatch):
    class DummyPatchDB:
        def get(self, pid):  # pragma: no cover - simple stub
            return types.SimpleNamespace(
                outcome="win", roi_delta=1.2, lines_changed=5, tests_passed=True
            )

    monkeypatch.setattr(cb, "PatchHistoryDB", DummyPatchDB)

    class DummyRetriever:
        def search(self, *_a, **_k):
            return [
                {
                    "origin_db": "patch",
                    "record_id": 1,
                    "score": 1.0,
                    "text": "",
                    "metadata": {"patch_id": 1, "summary": "patched", "diff": "diff"},
                }
            ]

    builder = ContextBuilder(retriever=DummyRetriever())
    data = json.loads(builder.build_context("q"))
    entry = data["patches"][0]
    assert entry["summary"] == "patched"
    assert entry["diff"] == "diff"
    assert entry["roi_delta"] == 1.2
    assert entry["lines_changed"] == 5
    assert entry["tests_passed"] is True
    assert entry["outcome"] == "win"


def test_context_builder_summarises_patch_text(monkeypatch):
    long_text = "line" * 40

    class DummyPatchDB:
        def get(self, pid):  # pragma: no cover - simple stub
            return types.SimpleNamespace(
                description=long_text,
                diff=long_text,
                outcome=long_text,
            )

    monkeypatch.setattr(cb, "PatchHistoryDB", DummyPatchDB)

    class DummyRetriever:
        def search(self, *_a, **_k):
            return [
                {
                    "origin_db": "patch",
                    "record_id": 1,
                    "score": 1.0,
                    "text": "",
                    "metadata": {"patch_id": 1},
                }
            ]

    builder = ContextBuilder(retriever=DummyRetriever())
    data = json.loads(builder.build_context("q"))
    entry = data["patches"][0]
    assert entry["summary"].endswith("...")
    assert entry["diff"].endswith("...")
    assert entry["outcome"].endswith("...")
