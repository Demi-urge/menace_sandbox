import os
import types
os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
from sandbox_runner.cycle import _choose_suggestion
import patch_suggestion_db as psdb


def test_choose_suggestion_uses_db(tmp_path):
    db = psdb.PatchSuggestionDB(tmp_path / "s.db")
    db.add(psdb.SuggestionRecord(module="mod.py", description="add logging"))
    db.add(psdb.SuggestionRecord(module="mod.py", description="add logging"))
    db.add(psdb.SuggestionRecord(module="mod.py", description="improve error"))
    ctx = types.SimpleNamespace(suggestion_cache={"mod.py": "fallback"}, suggestion_db=db)
    assert _choose_suggestion(ctx, "mod.py") == "add logging"


def test_choose_suggestion_fallback():
    ctx = types.SimpleNamespace(suggestion_cache={"mod.py": "fallback"}, suggestion_db=None)
    assert _choose_suggestion(ctx, "mod.py") == "fallback"

