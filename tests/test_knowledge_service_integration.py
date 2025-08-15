from __future__ import annotations

import types

from log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX
from tests.test_self_improvement_logging import _load_engine


class DummyKnowledgeService:
    def __init__(self) -> None:
        self.data = {
            FEEDBACK: "past feedback", 
            IMPROVEMENT_PATH: "past improvements", 
            ERROR_FIX: "recent fixes"
        }

    def get_recent_insights(self, tag: str) -> str:
        return self.data.get(tag, "")


def test_memory_summaries_include_insights(monkeypatch):
    sie = _load_engine()
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.gpt_memory = types.SimpleNamespace(search_context=lambda *a, **k: [])
    engine.knowledge_service = DummyKnowledgeService()
    summary = engine._memory_summaries("modX")
    assert "past feedback" in summary
    assert "past improvements" in summary
    assert "recent fixes" in summary
