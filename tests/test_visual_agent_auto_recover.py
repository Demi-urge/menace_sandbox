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

    pt_mod = types.ModuleType("pytesseract")
    pt_mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    pt_mod.image_to_string = lambda *a, **k: ""
    pt_mod.image_to_data = lambda *a, **k: {}
    pt_mod.Output = types.SimpleNamespace(DICT=0)
    monkeypatch.setitem(sys.modules, "pytesseract", pt_mod)

    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    return importlib.reload(importlib.import_module("menace_visual_agent_2"))


def test_restore_from_backup(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()

    for p in (va.QUEUE_FILE, va.STATE_FILE, va.HASH_FILE):
        p.rename(p.with_suffix(p.suffix + ".bak1"))

    va.QUEUE_FILE.write_text("bad")
    va.STATE_FILE.write_text("{")
    va.HASH_FILE.write_text("bad")

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()

    assert list(va2.task_queue)[0]["id"] == "a"
    assert va2.job_status["a"]["status"] == "queued"


def test_clear_when_backups_invalid(monkeypatch, tmp_path):
    va = _setup_va(monkeypatch, tmp_path)
    va.job_status["a"] = {"status": "queued", "prompt": "p", "branch": None}
    va.task_queue.append({"id": "a", "prompt": "p", "branch": None})
    va._persist_state()

    for p in (va.QUEUE_FILE, va.STATE_FILE, va.HASH_FILE):
        bak = p.with_suffix(p.suffix + ".bak1")
        p.rename(bak)
        bak.write_text("bad")

    va.QUEUE_FILE.write_text("bad2")
    va.STATE_FILE.write_text("bad2")
    va.HASH_FILE.write_text("bad2")

    va2 = _setup_va(monkeypatch, tmp_path)
    va2._initialize_state()

    assert not va2.task_queue
    assert not va2.job_status
