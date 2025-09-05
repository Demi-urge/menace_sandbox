import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB, PatchRecord


def test_find_patches_by_vector(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    pid1 = db.add(PatchRecord("a.py", "desc1", 1.0, 2.0))  # path-ignore
    pid2 = db.add(PatchRecord("b.py", "desc2", 1.0, 2.0))  # path-ignore
    db.log_ancestry(pid1, [("o", "v1", 0.5)])
    db.log_ancestry(pid2, [("o", "v1", 0.2), ("o", "v2", 0.7)])

    rows = db.find_patches_by_vector("v1")
    assert [r[0] for r in rows] == [pid1, pid2]
    assert [round(r[1], 1) for r in rows] == [0.5, 0.2]
