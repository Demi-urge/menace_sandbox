import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB, PatchRecord
from vector_service.patch_logger import PatchLogger


def test_patch_logger_logs_ancestry(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))
    pl = PatchLogger(patch_db=db)
    pl.track_contributors({"o1:v1": 0.2, "o2:v2": 0.9, "o3:v3": 0.5}, True, patch_id=str(pid), session_id="s")
    rows = db.get_ancestry(pid)
    assert [v for _, v, _ in rows] == ["v2", "v3", "v1"]
    assert [round(i, 1) for _, _, i in rows] == [0.9, 0.5, 0.2]
