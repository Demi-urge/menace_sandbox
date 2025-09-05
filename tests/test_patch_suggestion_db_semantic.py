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
    patch_count: int = 0
    module_id: str = ""


def test_add_safe_suggestion(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    db.add(SuggestionRecord(module="m.py", description="add logging"))  # path-ignore
    assert db.history("m.py") == ["add logging"]  # path-ignore


def test_add_unsafe_suggestion(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    with pytest.raises(ValueError):
        db.add(SuggestionRecord(module="m.py", description="# eval data"))  # path-ignore


def test_failure_similarity_rejected(tmp_path):
    safety = PatchSafety(failure_db_path=None)
    safety.record_failure({"category": "fail", "module": "m.py"})  # path-ignore
    db = PatchSuggestionDB(tmp_path / "s.db", safety=safety)
    with pytest.raises(ValueError):
        db.add(SuggestionRecord(module="m.py", description="fail"))  # path-ignore


def test_failed_strategy_storage(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    db.add_failed_strategy("abc")
    assert db.failed_strategy_tags() == ["abc"]


def test_queue_and_top_suggestions(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    suggs = [
        EnhancementSuggestion(path="a.py", score=1.0, rationale="a", patch_count=1, module_id="a1"),  # path-ignore
        EnhancementSuggestion(path="b.py", score=5.0, rationale="b", patch_count=2, module_id="b1"),  # path-ignore
        EnhancementSuggestion(path="c.py", score=3.0, rationale="c", patch_count=3, module_id="c1"),  # path-ignore
    ]
    db.queue_suggestions(suggs)
    top = db.top_suggestions(2)
    assert [s.module for s in top] == ["b.py", "c.py"]  # path-ignore
    assert top[0].rationale == "b"
    assert top[0].patch_count == 2
    assert top[0].module_id == "b1"


def test_queue_skips_duplicates(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    suggs = [
        EnhancementSuggestion(
            path="a.py", score=1.0, rationale="r", patch_count=1, module_id="m"  # path-ignore
        ),
        EnhancementSuggestion(
            path="a.py", score=1.05, rationale="r", patch_count=2, module_id="m"  # path-ignore
        ),
    ]
    db.queue_suggestions(suggs)
    top = db.top_suggestions(10)
    assert len(top) == 1


def test_outcome_adjusts_scoring(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    db.log_outcome("m.py", "prev", True, 1.0)  # path-ignore
    suggs = [
        EnhancementSuggestion(path="m.py", score=1.0, rationale="m"),  # path-ignore
        EnhancementSuggestion(path="o.py", score=1.0, rationale="o"),  # path-ignore
    ]
    db.queue_suggestions(suggs)
    top = db.top_suggestions(2)
    assert top[0].module == "m.py"  # path-ignore
    with db._lock:
        count = db.conn.execute(
            "SELECT COUNT(*) FROM suggestion_outcomes"
        ).fetchone()[0]
    assert count == 1
