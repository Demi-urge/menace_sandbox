import os
import types
import sys

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB, PatchRecord
from patch_provenance import get_patch_provenance
from vector_service.patch_logger import PatchLogger


def test_store_and_retrieve_alerts(tmp_path):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    db = PatchHistoryDB()
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))  # path-ignore
    logger = PatchLogger(patch_db=db)
    vec_id = "db1:vec1"
    meta = {vec_id: {"license": "mit", "semantic_alerts": ["unsafe"]}}
    logger.track_contributors([vec_id], True, patch_id=str(pid), session_id="s", retrieval_metadata=meta)
    prov = get_patch_provenance(pid, patch_db=db)
    assert prov[0]["license"] == "mit"
    assert prov[0]["semantic_alerts"] == ["unsafe"]
