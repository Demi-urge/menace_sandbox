import importlib
import os


def test_basic_flow(tmp_path, monkeypatch):
    db_path = tmp_path / "bb.db"
    monkeypatch.setenv("BORDERLINE_BUCKET_DB", str(db_path))
    if "borderline_bucket" in list(importlib.sys.modules.keys()):
        del importlib.sys.modules["borderline_bucket"]
    bb = importlib.import_module("borderline_bucket")
    bb.add_candidate("wf1", 0.1, 0.9)
    bb.record_outcome("wf1", True)
    bb.promote("wf1")
    data = bb.list_candidates()
    assert data == [
        {
            "workflow_id": "wf1",
            "raroi": 0.1,
            "confidence": 0.9,
            "status": "promoted",
            "outcomes": [True],
        }
    ]
