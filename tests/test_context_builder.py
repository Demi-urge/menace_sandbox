from typing import List

import pytest
import sys
import types

sys.modules.setdefault("bot_database", types.SimpleNamespace(BotDB=object))
sys.modules.setdefault("task_handoff_bot", types.SimpleNamespace(WorkflowDB=object))
sys.modules.setdefault("error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault("failure_learning_system", types.SimpleNamespace(DiscrepancyDB=object))
sys.modules.setdefault("code_database", types.SimpleNamespace(CodeDB=object))

from menace.context_builder import ContextBuilder


class TinyVecDB:
    """Minimal in-memory vector database for testing."""

    def __init__(self, entries: List[dict]):
        self.entries = []
        for e in entries:
            vec = 0.0 if "alpha" in e["text"] else 1.0
            self.entries.append({"vec": vec, **e})

    def encode_text(self, text: str) -> List[float]:
        return [0.0 if "alpha" in text else 1.0]

    def search_by_vector(self, vector: List[float], top_k: int):
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
def context_builder(monkeypatch):
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
    builder = ContextBuilder(
        bot_db=bot_db,
        workflow_db=workflow_db,
        error_db=error_db,
        code_db=code_db,
    )
    monkeypatch.setattr(builder, "_discrepancy_items", lambda q, limit=5: [])
    monkeypatch.setattr(builder.retriever, "_context_score", lambda kind, rec: (0.0, {}))
    monkeypatch.setattr(builder.retriever, "_related_boost", lambda scored, multiplier=1.1: {})
    return builder


def test_build_context_across_databases(context_builder):
    ctx = context_builder.build_context({"query": "alpha"})
    assert ctx["errors"] and ctx["errors"][0]["id"] == 100
    assert [b["id"] for b in ctx["bots"]] == [1, 2]
    assert ctx["workflows"][0]["id"] == 10
    assert ctx["code"][0]["id"] == 42


def test_collapse_and_format(context_builder):
    ctx = context_builder.build_context({"query": "alpha"})
    collapsed = context_builder.collapse_context(ctx, max_per_section=1)
    formatted = context_builder.format_collapsible(collapsed)
    assert "bots:" in formatted
    assert "AlphaBot" in formatted
    assert "... 1 more" in formatted
