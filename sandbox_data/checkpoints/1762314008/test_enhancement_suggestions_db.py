import types
from patch_suggestion_db import PatchSuggestionDB, EnhancementSuggestionRecord


def test_upsert_and_fetch(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")

    Suggestion = types.SimpleNamespace
    suggs = [
        Suggestion(path="mod1.py", score=1.0, rationale="r1"),  # path-ignore
        Suggestion(path="mod1.py", score=2.0, rationale="r1"),  # path-ignore
        Suggestion(path="mod2.py", score=3.0, rationale="r2"),  # path-ignore
    ]
    db.queue_enhancement_suggestions(suggs)

    rows = db.fetch_top_enhancement_suggestions(5)
    by_module = {r.module: r for r in rows}
    assert by_module["mod1.py"].occurrences == 2  # path-ignore
    assert by_module["mod1.py"].score == 2.0  # path-ignore
    assert by_module["mod2.py"].occurrences == 1  # path-ignore
    # After fetching they should be removed
    assert db.fetch_top_enhancement_suggestions(5) == []
