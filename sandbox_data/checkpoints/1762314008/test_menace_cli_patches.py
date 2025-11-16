import json
import sys
import types

ueb = types.ModuleType("unified_event_bus")
ueb.UnifiedEventBus = object
sys.modules.setdefault("unified_event_bus", ueb)

from menace_sandbox.code_database import PatchHistoryDB, PatchRecord
import importlib
import menace_cli
menace_cli = importlib.reload(menace_cli)


def setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "p.db"
    monkeypatch.setenv("PATCH_HISTORY_DB_PATH", str(db_path))
    db = PatchHistoryDB(db_path)
    pid1 = db.add(PatchRecord("a.py", "first", 0.0, 0.0))  # path-ignore
    pid2 = db.add(PatchRecord("b.py", "second", 0.0, 0.0))  # path-ignore
    db.log_ancestry(pid1, [("o", "vec1", 0.6, "MIT", [])])
    db.log_ancestry(pid2, [("o", "vec2", 0.4, "Apache-2.0", [])])
    return pid1, pid2


def test_list_and_ancestry(monkeypatch, tmp_path, capsys):
    pid1, _ = setup_db(tmp_path, monkeypatch)
    menace_cli.main(["patches", "list"])
    out = json.loads(capsys.readouterr().out)
    assert out[0]["filename"] == "b.py"  # path-ignore
    menace_cli.main(["patches", "ancestry", str(pid1)])
    out = json.loads(capsys.readouterr().out)
    assert out[0]["patch_id"] == pid1
    assert out[0]["vectors"][0]["vector_id"] == "vec1"


def test_search(monkeypatch, tmp_path, capsys):
    pid1, pid2 = setup_db(tmp_path, monkeypatch)
    menace_cli.main(["patches", "search", "--vector", "vec2"])
    out = json.loads(capsys.readouterr().out)
    assert [p["id"] for p in out] == [pid2]
    menace_cli.main(["patches", "search", "--license", "MIT"])
    out = json.loads(capsys.readouterr().out)
    assert [p["id"] for p in out] == [pid1]
