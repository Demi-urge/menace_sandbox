import pytest

from patch_suggestion_db import PatchSuggestionDB, SuggestionRecord


def test_add_safe_suggestion(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    db.add(SuggestionRecord(module="m.py", description="add logging"))
    assert db.history("m.py") == ["add logging"]


def test_add_unsafe_suggestion(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    with pytest.raises(ValueError):
        db.add(SuggestionRecord(module="m.py", description="# eval data"))
