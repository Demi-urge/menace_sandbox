import json
import types
import sys
from tests.test_visual_agent_auto_recover import _setup_va
from visual_agent_queue import VisualAgentQueue


def _stub_deps(monkeypatch):
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_mod.FastAPI = type("App", (), {})
    fastapi_mod.HTTPException = type("HTTPException", (Exception,), {})
    fastapi_mod.Header = lambda default=None: default
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)
    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = type("BaseModel", (), {})
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)
    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.Config = type("Config", (), {})
    uvicorn_mod.Server = type("Server", (), {})
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "x")


def test_queue_operations(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    q = va.task_queue
    q.clear()

    q.append({"id": "a"})
    q.append({"id": "b"})
    assert len(q) == 2
    ids = [item["id"] for item in q.load_all()]
    assert ids == ["a", "b"]

    popped = q.popleft()
    assert popped["id"] == "a"
    assert len(q) == 1

    q2 = VisualAgentQueue(tmp_path / "visual_agent_queue.db")
    assert len(q2) == 1


def test_migrate_legacy_queue(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    qfile = tmp_path / "visual_agent_queue.jsonl"
    qfile.write_text(json.dumps({"id": "x", "prompt": "p"}) + "\n")
    db = tmp_path / "visual_agent_queue.db"

    VisualAgentQueue.migrate_from_jsonl(db, qfile)
    assert qfile.with_suffix(qfile.suffix + ".bak").exists()

    q = VisualAgentQueue(db)
    assert [item["id"] for item in q.load_all()] == ["x"]


def test_recover_from_corrupt_db(monkeypatch, tmp_path):
    _stub_deps(monkeypatch)
    va = _setup_va(monkeypatch, tmp_path)
    va.task_queue.append({"id": "a"})

    db_path = tmp_path / "visual_agent_queue.db"
    db_path.write_text("bad")
    db_path.unlink()

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()
    assert not list(va2.task_queue)
    assert not va2.job_status
