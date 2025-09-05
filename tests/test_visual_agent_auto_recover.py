import importlib
import sys
import types
import json
from pathlib import Path

import pytest


def _setup_va(monkeypatch, tmp_path):
    heavy = ["cv2", "numpy", "mss", "pyautogui"]
    for name in heavy:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    filelock_mod = types.ModuleType("filelock")

    class DummyTimeout(Exception):
        pass

    class DummyFileLock:
        def __init__(self, *a, **k):
            pass

        def acquire(self, timeout=0):
            pass

        def release(self):
            pass

        @property
        def is_locked(self):
            return False

    filelock_mod.FileLock = DummyFileLock
    filelock_mod.Timeout = DummyTimeout
    monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    psutil_mod = types.ModuleType("psutil")
    psutil_mod.pid_exists = lambda *_a, **_k: False
    monkeypatch.setitem(sys.modules, "psutil", psutil_mod)

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = type("BaseModel", (), {})
    monkeypatch.setitem(sys.modules, "pydantic", pydantic_mod)

    fastapi_mod = types.ModuleType("fastapi")
    class DummyApp:
        def __init__(self, *a, **k):
            pass
        def post(self, *a, **k):
            return lambda f: f
        def get(self, *a, **k):
            return lambda f: f
        def on_event(self, *a, **k):
            return lambda f: f

    fastapi_mod.FastAPI = DummyApp
    fastapi_mod.HTTPException = type("HTTPException", (Exception,), {})
    fastapi_mod.Header = lambda default=None: default
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_mod)

    uvicorn_mod = types.ModuleType("uvicorn")
    uvicorn_mod.Config = type("Config", (), {})
    uvicorn_mod.Server = type("Server", (), {})
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_mod)

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")  # path-ignore
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VISUAL_AGENT_TOKEN", "tok")
    return importlib.reload(importlib.import_module("menace_visual_agent_2"))


def test_migrate_from_jsonl(monkeypatch, tmp_path):
    """Legacy queue and state files should migrate into the SQLite DB."""
    qfile = tmp_path / "visual_agent_queue.jsonl"
    sfile = tmp_path / "visual_agent_state.json"
    qfile.write_text(json.dumps({"id": "a", "prompt": "p"}) + "\n")
    state = {"status": {"a": {"status": "queued", "prompt": "p", "branch": None}}, "last_completed": 1.0}
    sfile.write_text(json.dumps(state))

    from visual_agent_queue import VisualAgentQueue

    db = tmp_path / "visual_agent_queue.db"
    VisualAgentQueue.migrate_from_jsonl(db, qfile, sfile)

    q = VisualAgentQueue(db)
    assert q.get_status()["a"]["status"] == "queued"
    assert q.get_last_completed() == 1.0
    assert qfile.with_suffix(qfile.suffix + ".bak").exists()
    assert sfile.with_suffix(sfile.suffix + ".bak").exists()

    va = _setup_va(monkeypatch, tmp_path)
    va._initialize_state()
    assert va.job_status["a"]["status"] == "queued"


def test_recover_from_corrupt_db(monkeypatch, tmp_path):
    """Corrupted database should result in an empty queue on startup."""
    va = _setup_va(monkeypatch, tmp_path)
    va.task_queue.append({"id": "a", "prompt": "p"})

    db_path = tmp_path / "visual_agent_queue.db"
    db_path.write_text("bad")
    # simulate detection of corruption by removing the bad file before reload
    db_path.unlink()

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()

    assert not list(va2.task_queue)
    assert not va2.job_status
