import sys
import types
import json
import pytest

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from menace_sandbox.code_database import PatchHistoryDB, PatchRecord
from menace_sandbox.vector_service.patch_logger import PatchLogger
from menace_sandbox.patch_provenance import PatchProvenanceService
from menace_sandbox.vector_metrics_db import VectorMetricsDB


def test_patch_logger_logs_ancestry(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))  # path-ignore
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
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))  # path-ignore
    pl = PatchLogger(patch_db=db)
    meta = {"o1:v1": {"license": "MIT", "semantic_alerts": ["unsafe"]}}
    pl.track_contributors({"o1:v1": 0.8}, True, patch_id=str(pid), session_id="s", retrieval_metadata=meta)
    rows = db.get_ancestry(pid)
    assert rows[0][3] == "MIT"
    assert json.loads(rows[0][5]) == ["unsafe"]
    service = PatchProvenanceService(patch_db=db)
    prov = service.get_provenance(pid)
    assert prov[0]["origin"] == "o1"
    assert prov[0]["vector_id"] == "v1"
    assert prov[0]["influence"] == pytest.approx(0.8)
    assert prov[0]["license"] == "MIT"
    assert prov[0]["semantic_alerts"] == ["unsafe"]
    assert prov[0]["roi_before"] == pytest.approx(1.0)
    assert prov[0]["roi_after"] == pytest.approx(2.0)
    assert prov[0]["roi_delta"] == pytest.approx(1.0)


def test_vector_metrics_records_alignment_severity(tmp_path):
    db = PatchHistoryDB(tmp_path / "p.db")
    vm = VectorMetricsDB(tmp_path / "v.db")
    pid = db.add(PatchRecord("a.py", "desc", 1.0, 2.0))  # path-ignore
    pl = PatchLogger(patch_db=db, vector_metrics=vm, max_alert_severity=5.0)
    meta = {"o1:v1": {"alignment_severity": 2}}
    pl.track_contributors({"o1:v1": 0.5}, True, patch_id=str(pid), session_id="s", retrieval_metadata=meta)
    rows = vm.conn.execute("SELECT vector_id, alignment_severity FROM patch_ancestry").fetchall()
    assert rows == [("v1", 2.0)]
