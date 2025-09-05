import pytest
from patch_suggestion_db import PatchSuggestionDB, SuggestionRecord


def test_enhancement_outcome_reporting(tmp_path):
    db = PatchSuggestionDB(tmp_path / "s.db")
    # Add suggestion and capture ID
    rec = SuggestionRecord(module="mod.py", description="refactor")  # path-ignore
    sugg_id = db.add(rec)
    assert sugg_id
    # Log two outcomes for same suggestion
    db.log_enhancement_outcome(sugg_id, 1, roi_delta=0.5, error_delta=-1.0)
    db.log_enhancement_outcome(sugg_id, 2, roi_delta=1.0, error_delta=-2.0)
    report = db.enhancement_report()
    assert report[0]["suggestion_id"] == sugg_id
    assert report[0]["patch_count"] == 2
    assert report[0]["avg_roi_delta"] == pytest.approx(0.75)
    assert report[0]["avg_error_delta"] == pytest.approx(-1.5)
