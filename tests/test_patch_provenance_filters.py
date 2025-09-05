import json
import os
import subprocess
import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB, PatchRecord
from patch_provenance_service import create_app
from vector_service.patch_logger import PatchLogger


def _setup_db(path):
    db = PatchHistoryDB(path)
    pid1 = db.add(PatchRecord("a.py", "desc1", 1.0, 2.0))  # path-ignore
    pid2 = db.add(PatchRecord("b.py", "desc2", 1.0, 2.0))  # path-ignore
    db.log_ancestry(pid1, [("o", "v1", 0.5, "GPL", "fp1", ["malware"])])
    db.log_ancestry(pid2, [("o", "v2", 0.7, "MIT", "fp2", ["trojan"])])
    return db, pid1, pid2


def test_service_filters(tmp_path):
    db, pid1, pid2 = _setup_db(tmp_path / "p.db")
    app = create_app(db)
    client = app.test_client()
    res = client.get("/patches", query_string={"license": "GPL"})
    assert res.get_json() == [
        {"id": pid1, "filename": "a.py", "description": "desc1"}  # path-ignore
    ]
    res = client.get("/patches", query_string={"semantic_alert": "trojan"})
    assert res.get_json() == [
        {"id": pid2, "filename": "b.py", "description": "desc2"}  # path-ignore
    ]
    res = client.get("/patches", query_string={"license_fingerprint": "fp2"})
    assert res.get_json() == [
        {"id": pid2, "filename": "b.py", "description": "desc2"}  # path-ignore
    ]
    res = client.get(
        "/patches",
        query_string={"license": "GPL", "semantic_alert": "malware"},
    )
    assert res.get_json() == [
        {"id": pid1, "filename": "a.py", "description": "desc1"}  # path-ignore
    ]


def test_cli_filters(tmp_path):
    db, pid1, pid2 = _setup_db(tmp_path / "c.db")
    env = os.environ.copy()
    env["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "c.db")
    out = subprocess.run(
        [sys.executable, "-m", "tools.patch_provenance_cli", "search", "--license", "MIT"],
        capture_output=True,
        text=True,
        env=env,
    )
    patches = json.loads(out.stdout)
    assert [p["id"] for p in patches] == [pid2]
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.patch_provenance_cli",
            "search",
            "--semantic-alert",
            "malware",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    patches = json.loads(out.stdout)
    assert [p["id"] for p in patches] == [pid1]
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.patch_provenance_cli",
            "search",
            "--license-fingerprint",
            "fp1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    patches = json.loads(out.stdout)
    assert [p["id"] for p in patches] == [pid1]
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.patch_provenance_cli",
            "search",
            "--license",
            "GPL",
            "--semantic-alert",
            "malware",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    patches = json.loads(out.stdout)
    assert [p["id"] for p in patches] == [pid1]


def test_track_contributors_sorted(tmp_path):
    db = PatchHistoryDB(tmp_path / "t.db")
    pid = db.add(PatchRecord("c.py", "desc", 0.0, 0.0))  # path-ignore
    logger = PatchLogger(patch_db=db)
    vectors = [("o:v2", 0.8), ("o:v1", 0.2)]
    meta = {
        "o:v1": {"license": "MIT", "semantic_alerts": ["trojan"]},
        "o:v2": {"license": "GPL", "semantic_alerts": ["malware"]},
    }
    logger.track_contributors(
        vectors,
        True,
        patch_id=str(pid),
        session_id="s",
        retrieval_metadata=meta,
    )
    contributors = db.get_contributors(pid)
    assert [v for v, _, _ in contributors] == ["o:v2", "o:v1"]
    ancestry = db.get_ancestry(pid)
    assert ancestry[0][3] == "GPL" and json.loads(ancestry[0][5]) == ["malware"]
    assert ancestry[1][3] == "MIT" and json.loads(ancestry[1][5]) == ["trojan"]

