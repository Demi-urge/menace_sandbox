import types
import sys

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from patch_provenance_service import create_app
from code_database import PatchHistoryDB
from patch_suggestion_db import PatchSuggestionDB
from patch_safety import PatchSafety


def test_rejects_semantic_risk(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    app = create_app(db, PatchSuggestionDB(tmp_path / "s.db"), PatchSafety(failure_db_path=None))
    client = app.test_client()
    res = client.post(
        "/patches",
        json={
            "filename": "m.py",  # path-ignore
            "description": "eval('data')",
            "roi_before": 0.0,
            "roi_after": 0.0,
        },
    )
    assert res.status_code == 400
