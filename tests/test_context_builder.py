import json
import types
import sys

import pytest

# Provide minimal modules so that context_builder imports succeed without the
# heavy real implementations.
sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
_code_stub = types.ModuleType("code_database")
_code_stub.__spec__ = types.SimpleNamespace()
_code_stub.__file__ = "code_database_stub.py"
_code_stub.CodeDB = object
sys.modules.setdefault("code_database", _code_stub)
sys.modules.setdefault("menace.code_database", _code_stub)
sys.modules.setdefault(
    "failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object)
)

from menace.context_builder import ContextBuilder


class TinyVecDB:
    """In-memory vector store used for testing the builder.

This stub mimics the minimal interface of ErrorDB, BotDB,
WorkflowDB and CodeDB so tests can seed sample records without
requiring the full database implementations."""

    def __init__(self, entries: list[dict]):
        self.entries = []
        for e in entries:
            vec = 0.0 if "alpha" in e["text"] else 1.0
            self.entries.append({"vec": vec, **e})

    def encode_text(self, text: str) -> list[float]:
        return [0.0 if "alpha" in text else 1.0]

    def search_by_vector(self, vector: list[float], top_k: int):
        q = vector[0]
        items = []
        for e in self.entries:
            dist = abs(e["vec"] - q)
            meta = dict(e["meta"])
            meta["_distance"] = dist
            meta.setdefault("id", e["meta"].get("id"))
            items.append(meta)
        items.sort(key=lambda m: m["_distance"])
        return items[:top_k]

    def get_vector(self, rec_id: int):
        for e in self.entries:
            if e["meta"].get("id") == rec_id:
                return [e["vec"]]
        return [1.0]


@pytest.fixture
def builder(monkeypatch):
    bot_db = TinyVecDB(
        [
            {"text": "alpha bot helps", "meta": {"id": 1, "name": "AlphaBot", "roi": 2.0}},
            {"text": "other bot", "meta": {"id": 2, "name": "BetaBot", "roi": 1.0}},
        ]
    )
    workflow_db = TinyVecDB(
        [
            {"text": "alpha workflow", "meta": {"id": 10, "title": "AlphaFlow", "roi": 0.5}}
        ]
    )
    error_db = TinyVecDB(
        [
            {"text": "alpha crash", "meta": {"id": 100, "message": "alpha crash", "frequency": 5}}
        ]
    )
    code_db = TinyVecDB(
        [
            {
                "text": "alpha code snippet",
                "meta": {"id": 42, "summary": "alpha code", "complexity_score": 0.2},
            }
        ]
    )
    discrepancy_db = TinyVecDB(
        [
            {
                "text": "alpha discrepancy",
                "meta": {"id": 700, "message": "alpha discrepancy", "severity": 0.9},
            }
        ]
    )
    mm = types.SimpleNamespace(_summarise_text=lambda text: text[:7])
    builder = ContextBuilder(
        error_db=error_db,
        bot_db=bot_db,
        workflow_db=workflow_db,
        discrepancy_db=discrepancy_db,
        code_db=code_db,
        memory_manager=mm,
    )
    monkeypatch.setattr(builder.retriever, "_context_score", lambda kind, rec: (0.0, {}))
    monkeypatch.setattr(builder.retriever, "_related_boost", lambda scored, multiplier=1.1: {})
    return builder
def test_build_context(builder):
    ctx_json = builder.build_context("alpha", limit_per_db=2)
    ctx = json.loads(ctx_json)
    assert ctx["errors"][0]["id"] == 100
    assert ctx["errors"][0]["desc"] == "alpha c"
    assert pytest.approx(ctx["errors"][0]["metric"]) == 1.0 / 6.0
    assert [b["id"] for b in ctx["bots"]] == [1, 2]
    assert ctx["workflows"][0]["id"] == 10
    assert ctx["code"][0]["id"] == 42
    assert ctx["discrepancies"][0]["id"] == 700

