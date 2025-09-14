import json
import os
import subprocess
import sys
import types
from pathlib import Path

ueb = types.ModuleType("unified_event_bus")
ueb.UnifiedEventBus = object
sys.modules.setdefault("unified_event_bus", ueb)

from menace_sandbox.code_database import PatchHistoryDB, PatchRecord
from patch_provenance_service import create_app


def _setup_db(tmp_path):
    os.environ["PATCH_HISTORY_DB_PATH"] = str(tmp_path / "patch_history.db")
    db = PatchHistoryDB()
    p1 = db.add(
        PatchRecord(filename="a.py", description="first patch", roi_before=0, roi_after=1),  # path-ignore
        [("origin:vec1", 0.9)],
    )
    db.log_ancestry(p1, [("origin", "vec1", 0.9)])
    p2 = db.add(
        PatchRecord(filename="b.py", description="second patch", roi_before=1, roi_after=2),  # path-ignore
        [("vec2", 0.5)],
    )
    db.log_ancestry(p2, [("", "vec2", 0.5)])
    return db, p1, p2


def _run_cli(args, cwd):
    proc = subprocess.run(
        [sys.executable, str(Path(cwd) / "patch_provenance_cli.py"), *args],  # path-ignore
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip())


def test_cli_and_service(tmp_path):
    root = Path(__file__).resolve().parents[2]
    db, p1, p2 = _setup_db(tmp_path)

    data = _run_cli(["list"], root)
    ids = {entry["id"] for entry in data}
    assert {p1, p2}.issubset(ids)

    show = _run_cli(["show", str(p1)], root)
    assert show["id"] == p1
    assert show["provenance"][0]["vector_id"] == "vec1"
    assert show["chain"][0]["patch_id"] == p1

    chain = _run_cli(["chain", str(p1)], root)
    assert chain[0]["patch_id"] == p1

    res = _run_cli(["search", "vec2"], root)
    assert res[0]["id"] == p2

    app = create_app(db)
    client = app.test_client()
    resp = client.get("/patches")
    assert resp.status_code == 200
    body = resp.get_json()
    assert any(entry["id"] == p1 for entry in body)

    resp = client.get(f"/patches/{p1}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["provenance"][0]["vector_id"] == "vec1"
    assert body["chain"][0]["patch_id"] == p1

    resp = client.get("/vectors/vec1")
    body = resp.get_json()
    assert body and body[0]["id"] == p1

    resp = client.get("/search", query_string={"q": "vec1"})
    body = resp.get_json()
    assert body and body[0]["id"] == p1
