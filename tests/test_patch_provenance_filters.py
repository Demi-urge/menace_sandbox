import json
import os
import subprocess
import sys
import types

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from code_database import PatchHistoryDB, PatchRecord
from patch_provenance_service import create_app


def _setup_db(path):
    db = PatchHistoryDB(path)
    pid1 = db.add(PatchRecord("a.py", "desc1", 1.0, 2.0))
    pid2 = db.add(PatchRecord("b.py", "desc2", 1.0, 2.0))
    db.log_ancestry(pid1, [("o", "v1", 0.5, "GPL", ["malware"])])
    db.log_ancestry(pid2, [("o", "v2", 0.7, "MIT", ["trojan"])])
    return db, pid1, pid2


def test_service_filters(tmp_path):
    db, pid1, pid2 = _setup_db(tmp_path / "p.db")
    app = create_app(db)
    client = app.test_client()
    res = client.get("/patches", query_string={"license": "GPL"})
    assert res.get_json() == [
        {"id": pid1, "filename": "a.py", "description": "desc1"}
    ]
    res = client.get("/patches", query_string={"semantic_alert": "trojan"})
    assert res.get_json() == [
        {"id": pid2, "filename": "b.py", "description": "desc2"}
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

