import pytest
import types
import sys

vec_mod = types.ModuleType("vector_service")


class _EmbeddableDBMixin:
    def __init__(self, *a, **k):
        pass


vec_mod.EmbeddableDBMixin = _EmbeddableDBMixin
sys.modules.setdefault("vector_service", vec_mod)

from patch_suggestion_db import PatchSuggestionDB, SuggestionRecord  # noqa: E402
from patch_safety import PatchSafety  # noqa: E402
from dataclasses import dataclass  # noqa: E402


@dataclass
class EnhancementSuggestion:
    path: str
    score: float
    rationale: str


def test_add_safe_suggestion(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    db.add(SuggestionRecord(module="m.py", description="add logging"))
    assert db.history("m.py") == ["add logging"]


def test_add_unsafe_suggestion(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    with pytest.raises(ValueError):
        db.add(SuggestionRecord(module="m.py", description="# eval data"))


def test_failure_similarity_rejected(tmp_path):
    safety = PatchSafety()
    safety.record_failure({"category": "fail", "module": "m.py"})
    db = PatchSuggestionDB(tmp_path / "s.db", safety=safety)
    with pytest.raises(ValueError):
        db.add(SuggestionRecord(module="m.py", description="fail"))


def test_failed_strategy_storage(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    db.add_failed_strategy("abc")
    assert db.failed_strategy_tags() == ["abc"]


def test_queue_and_top_suggestions(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    suggs = [
        EnhancementSuggestion(path="a.py", score=1.0, rationale="a"),
        EnhancementSuggestion(path="b.py", score=5.0, rationale="b"),
        EnhancementSuggestion(path="c.py", score=3.0, rationale="c"),
    ]
    db.queue_suggestions(suggs)
    top = db.top_suggestions(2)
    assert [s.module for s in top] == ["b.py", "c.py"]
    assert top[0].rationale == "b"
