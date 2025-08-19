import sys
import types
import json

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB, PatchRecord
from vector_service.patch_logger import PatchLogger
from patch_provenance import get_patch_provenance


def test_patch_logger_logs_ancestry(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))
    pl = PatchLogger(patch_db=db)
    pl.track_contributors({"o1:v1": 0.2, "o2:v2": 0.9, "o3:v3": 0.5}, True, patch_id=str(pid), session_id="s")
    rows = db.get_ancestry(pid)
    assert [v for _, v, *_ in rows] == ["v2", "v3", "v1"]
    assert [round(i, 1) for _, _, i, *_ in rows] == [0.9, 0.5, 0.2]
    contribs = db.get_contributors(pid)
    assert [v for v, _, _ in contribs] == ["o2:v2", "o3:v3", "o1:v1"]
    assert [round(i, 1) for _, i, _ in contribs] == [0.9, 0.5, 0.2]
    assert all(s == "s" for _, _, s in contribs)


def test_patch_logger_records_provenance(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))
    pl = PatchLogger(patch_db=db)
    meta = {"o1:v1": {"license": "MIT", "semantic_alerts": ["unsafe"]}}
    pl.track_contributors({"o1:v1": 0.8}, True, patch_id=str(pid), session_id="s", retrieval_metadata=meta)
    rows = db.get_ancestry(pid)
    assert rows[0][3] == "MIT"
    assert json.loads(rows[0][5]) == ["unsafe"]
    prov = get_patch_provenance(pid, patch_db=db)
    assert prov == [
        {
            "origin": "o1",
            "vector_id": "v1",
            "influence": 0.8,
            "license": "MIT",
            "license_fingerprint": None,
            "semantic_alerts": ["unsafe"],
        }
    ]
