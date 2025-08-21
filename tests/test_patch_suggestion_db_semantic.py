import pytest

from patch_suggestion_db import PatchSuggestionDB, SuggestionRecord
from patch_safety import PatchSafety


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
